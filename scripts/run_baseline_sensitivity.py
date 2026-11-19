#!/usr/bin/env python3
"""Do the quantum wins survive retuned classical comparators? (reviewer R1.4)

Four classical comparators either stop classifying on some cohort (pooled
sensitivity or specificity below 0.30, or |MCC| < 0.02) or fail to train:
the suite's 1-D CNN and Trainable-RBF, and Plan C's C-CNN and C-MLP. Each is
retuned here *inside every training fold*: a stratified 20% of the training
subjects is held out, every configuration of a small grid is trained on the
rest and scored by balanced accuracy on the held-out part, and the winner is
retrained on the whole training fold and tested once on the held-out site.
The test site is never seen during selection. The original configuration is
first in each grid, so ties keep it.

The quantum arms are not retuned; their per-fold records are read from
results/rerun22. The analysis is therefore deliberately conservative toward
quantum: a win that survives a classical comparator tuned on its own training
data cannot be blamed on a weak classical implementation.

Examples:
    python scripts/run_baseline_sensitivity.py --data-root data/ABIDE-I --tag abide
    python scripts/run_baseline_sensitivity.py --data-root data/UCLA-CNP-cc200 \\
        --tag ucla --cv stratified
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from run_quantum_models import Classical1DCNN as SuiteCNN
from run_quantum_models import metrics, select_features, train_torch_classifier
from run_reupload import Classical1DCNN as PlanCCNN
from run_reupload import train_model as train_planc
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.trainable_kernel import TrainableClassicalKernel, gram_matrix, train_kernel

N = 8
METRICS = ("accuracy", "f1", "auc", "specificity")


def _mlp(width):
    return torch.nn.Sequential(torch.nn.Linear(N, width), torch.nn.ReLU(),
                               torch.nn.Dropout(0.2), torch.nn.Linear(width, 2))


def fit_suite_cnn(cfg, xtr, ytr, xte, seed):
    return train_torch_classifier(SuiteCNN(N, channels=cfg["channels"]), xtr, ytr, xte,
                                  epochs=cfg["epochs"], lr=cfg["lr"], seed=seed)


def fit_planc_cnn(cfg, xtr, ytr, xte, seed):
    return train_planc(PlanCCNN(N, channels=cfg["channels"]), xtr, ytr, xte,
                       cfg["epochs"], cfg["lr"], seed)


def fit_planc_mlp(cfg, xtr, ytr, xte, seed):
    return train_planc(_mlp(cfg["width"]), xtr, ytr, xte, cfg["epochs"], cfg["lr"], seed)


def fit_trainable_rbf(cfg, xtr, ytr, xte, seed):
    tck = TrainableClassicalKernel(n_features=N, seed=seed)
    train_kernel(tck, xtr, ytr, steps=cfg["steps"], lr=cfg["lr"], seed=seed)
    k_train, k_test = gram_matrix(tck, xtr, xtr), gram_matrix(tck, xte, xtr)
    model = SVC(kernel="precomputed", C=cfg["C"], class_weight="balanced").fit(k_train, ytr)
    raw = model.decision_function(k_test)
    train_acc = float(((model.decision_function(k_train) > 0) == ytr).mean())
    return (raw > 0).astype(int), raw, train_acc


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


# comparator -> (plan, results file, quantum arms matched to it, trainer, grid)
TARGETS = {
    "CNN (1-D)": ("qmodels", "quantum_models_results.json", ["QCNN"], fit_suite_cnn,
                  grid(lr=[0.05, 0.01, 0.002], channels=[8, 16], epochs=[150, 400])),
    "Trainable-RBF": ("qmodels", "quantum_models_results.json", ["TQEK"], fit_trainable_rbf,
                      grid(steps=[60, 200], lr=[0.05, 0.01], C=[1.0, 0.1, 10.0])),
    "C-CNN": ("planC", "reupload_results.json", ["Q-Plain", "Q-Reup1", "Q-Reup2"],
              fit_planc_cnn, grid(lr=[0.02, 0.005, 0.001], channels=[8, 16], epochs=[40, 120])),
    "C-MLP": ("planC", "reupload_results.json", ["Q-Plain", "Q-Reup1", "Q-Reup2"],
              fit_planc_mlp, grid(lr=[0.02, 0.005, 0.001], width=[32, 64], epochs=[40, 120])),
}


def pooled(folds):
    tn, fp, fn, tp = (sum(f[k] for f in folds) for k in ("tn", "fp", "fn", "tp"))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    degenerate = not math.isfinite(mcc) or abs(mcc) < 0.02 or min(sens, spec) < 0.30
    return {"sensitivity": sens, "specificity": spec, "mcc": mcc, "degenerate": degenerate}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="cohort tag used in results/rerun22")
    ap.add_argument("--cv", default="leave-site-out", choices=["leave-site-out", "stratified"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    ap.add_argument("--rerun-dir", type=Path, default=Path("results/rerun22"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_baseline_sensitivity_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_abide(root=args.data_root, pipeline="cpac", strategy="filt_noglobal",
                         atlas="rois_cc200", n_components=N)
    series_dir = args.data_root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    conn = build_descriptors([np.loadtxt(series_dir / f"{s.file_id}_rois_cc200.1D")
                              for s in dataset.subjects], kind="correlation")
    labels, sites = dataset.labels, dataset.sites
    print(f"{args.tag}: {conn.shape}", flush=True)

    if args.cv == "stratified":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = [(f"fold{i+1}", tr, te) for i, (tr, te) in enumerate(splitter.split(conn, labels))]
    else:
        splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                  for s in sorted(set(sites.tolist()))]

    tuned = {name: [] for name in args.only}
    chosen_cfg = {name: [] for name in args.only}
    seconds = dict.fromkeys(args.only, 0.0)
    for site, tr, te in splits:
        if len(te) < args.min_test_size or len(np.unique(labels[te])) < 2:
            continue
        ytr, yte = labels[tr], labels[te]
        cols = select_features(conn[tr], ytr, N)
        scaler = StandardScaler().fit(conn[tr][:, cols])
        xtr, xte = scaler.transform(conn[tr][:, cols]), scaler.transform(conn[te][:, cols])
        inner_fit, inner_val = next(StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=args.seed).split(xtr, ytr))

        for name in args.only:
            _, _, _, fit, configs = TARGETS[name]
            t0 = time.perf_counter()
            scores = []
            for cfg in configs:
                preds, _, _ = fit(cfg, xtr[inner_fit], ytr[inner_fit], xtr[inner_val], args.seed)
                scores.append(balanced_accuracy_score(ytr[inner_val], preds))
            best = configs[int(np.argmax(scores))]
            preds, raw, train_acc = fit(best, xtr, ytr, xte, args.seed)
            seconds[name] += time.perf_counter() - t0
            tuned[name].append({"site": site, "n": len(te), "train_accuracy": train_acc,
                                **metrics(yte, preds, raw)})
            chosen_cfg[name].append({"site": site, **best, "inner_bacc": float(max(scores))})
        print(f"  {site:<12} " + "  ".join(
            f"{n}={tuned[n][-1]['accuracy']:.3f}" for n in args.only), flush=True)

    report = {}
    for name in args.only:
        plan, fname, quantum_arms, _, _ = TARGETS[name]
        path = args.rerun_dir / f"{plan}_{args.tag}" / fname
        original = json.loads(path.read_text())["per_fold"]
        order = [f["site"] for f in tuned[name]]
        by_site = {arm: {f["site"]: f for f in original[arm]} for arm in quantum_arms + [name]}
        if any(s not in by_site[name] for s in order):
            raise SystemExit(f"fold mismatch between {path} and this run")
        entry = {"pooled_tuned": pooled(tuned[name]),
                 "mean_tuned": {m: float(np.nanmean([f.get(m, np.nan) for f in tuned[name]]))
                                for m in METRICS + ("train_accuracy",)},
                 "mean_original": {m: float(np.nanmean([by_site[name][s].get(m, np.nan)
                                                        for s in order])) for m in METRICS},
                 "comparisons": {}}
        for arm in quantum_arms:
            for m in METRICS:
                q = np.array([by_site[arm][s].get(m, np.nan) for s in order], float)
                for label, c in (("original", [by_site[name][s].get(m, np.nan) for s in order]),
                                 ("tuned", [f.get(m, np.nan) for f in tuned[name]])):
                    c = np.array(c, float)
                    ok = ~(np.isnan(q) | np.isnan(c))
                    d = q[ok] - c[ok]
                    p = float(wilcoxon(q[ok], c[ok])[1]) if ok.sum() > 2 and np.any(d) else 1.0
                    entry["comparisons"][f"{arm} vs {label} {name} [{m}]"] = {
                        "median_diff": float(np.median(d)), "p_value": p,
                        "wins": int((d > 0).sum()), "folds": int(ok.sum())}
        report[name] = entry
        print(f"\n{name}: pooled tuned {entry['pooled_tuned']}")
        for k, v in entry["comparisons"].items():
            print(f"  {k:<58} {v['median_diff']:+.3f}  p={v['p_value']:.4f}")

    (out_dir / "baseline_sensitivity.json").write_text(json.dumps(
        {"config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir),
                                 "rerun_dir": str(args.rerun_dir)},
         "grids": {n: TARGETS[n][4] for n in args.only}, "seconds": seconds,
         "tuned_per_fold": tuned, "chosen": chosen_cfg, "report": report,
         "timestamp": stamp}, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'baseline_sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
