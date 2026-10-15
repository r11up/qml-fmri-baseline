#!/usr/bin/env python3
"""Label-permutation null for the matched kernel comparisons.

Section IX of the manuscript specifies a permutation null and reports that it
was implemented but not run at scale. This runs it for the kernel arm, which is
where the manuscript's strongest quantum claim sits (TQEK against its trainable
classical twin).

Labels are shuffled *within site*, not globally. Global shuffling destroys the
per-site class prevalence, which would let a model that merely tracks site
composition look good against the null; within-site shuffling preserves
prevalence and is the stricter test. Feature selection, scaling and bandwidth
choice are redone inside every fold on the permuted labels, so nothing leaks
from the true labelling into the null.

    cd github && PYTHONPATH=src ../.venv/bin/python \
        scripts/run_permutation_null.py --permutations 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported rather than re-implemented: the null must use exactly the feature
# selection and metric definitions the observed run used, or the comparison is
# between two different procedures.
from run_quantum_models import metrics, select_features  # noqa: E402

from qagta.data.abide import load_abide  # noqa: E402
from qagta.data.descriptors import build_descriptors  # noqa: E402
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix  # noqa: E402
from qagta.quantum.trainable_kernel import (  # noqa: E402
    TrainableClassicalKernel,
    TrainableQuantumKernel,
    gram_matrix,
    train_kernel,
)

MODELS = ["QSVM-fixed", "TQEK", "SVM-RBF", "Trainable-RBF"]


def one_pass(connectivity, labels, sites, splits, qubits, bandwidths,
             kernel_steps, seed):
    """One full LSO sweep of the four kernel models; returns mean acc and AUC."""
    acc = {m: [] for m in MODELS}
    auc = {m: [] for m in MODELS}
    for _site, train_idx, test_idx in splits:
        y_train, y_test = labels[train_idx], labels[test_idx]
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue
        chosen = select_features(connectivity[train_idx], y_train, qubits)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        x_train = scaler.transform(connectivity[train_idx][:, chosen])
        x_test = scaler.transform(connectivity[test_idx][:, chosen])
        angle = MinMaxScaler((0, np.pi)).fit(x_train)
        a_train = angle.transform(x_train)
        a_test = np.clip(angle.transform(x_test), 0, np.pi)

        def keep(name, preds, scores, y_test=y_test):
            m = metrics(y_test, preds, scores)
            acc[name].append(m["accuracy"])
            if m.get("auc") is not None:
                auc[name].append(m["auc"])

        # QSVM-fixed: bandwidth by kernel-target alignment, inside the fold
        y_signed = np.where(y_train > 0, 1.0, -1.0)
        ideal = np.outer(y_signed, y_signed)
        best = None
        for bw in bandwidths:
            fm = QuantumFeatureMap(n_qubits=qubits, reps=2,
                                   entanglement="linear", bandwidth=bw)
            k = quantum_kernel_matrix(a_train, a_train, fm)
            den = np.linalg.norm(k) * np.linalg.norm(ideal)
            score = float((k * ideal).sum() / den) if den > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, fm, k)
        _, fm, k_train = best
        k_test = quantum_kernel_matrix(a_test, a_train, fm)
        svc = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(k_train, y_train)
        raw = svc.decision_function(k_test)
        keep("QSVM-fixed", (raw > 0).astype(int), raw)

        # TQEK
        tqek = TrainableQuantumKernel(n_qubits=qubits, layers=2, seed=seed)
        train_kernel(tqek, a_train, y_train, steps=kernel_steps, seed=seed)
        svc = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(
            gram_matrix(tqek, a_train, a_train), y_train)
        raw = svc.decision_function(gram_matrix(tqek, a_test, a_train))
        keep("TQEK", (raw > 0).astype(int), raw)

        # Trainable-RBF, identical budget
        tck = TrainableClassicalKernel(n_features=qubits, seed=seed)
        train_kernel(tck, x_train, y_train, steps=kernel_steps, seed=seed)
        svc = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(
            gram_matrix(tck, x_train, x_train), y_train)
        raw = svc.decision_function(gram_matrix(tck, x_test, x_train))
        keep("Trainable-RBF", (raw > 0).astype(int), raw)

        # tuned classical RBF
        svc = SVC(kernel="rbf", C=1.0, gamma="scale",
                  class_weight="balanced").fit(x_train, y_train)
        raw = svc.decision_function(x_test)
        keep("SVM-RBF", (raw > 0).astype(int), raw)

    return ({m: float(np.mean(v)) for m, v in acc.items()},
            {m: (float(np.mean(v)) if v else None) for m, v in auc.items()})


def shuffle_within_site(labels, sites, rng):
    out = labels.copy()
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        out[idx] = labels[rng.permutation(idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--kernel-steps", type=int, default=60)
    ap.add_argument("--bandwidths", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.25, 0.5])
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index; gives this process its own permutation stream")
    ap.add_argument("--skip-observed", action="store_true",
                    help="shards other than 0 need not recompute the observed pass")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_permutation_null"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_abide(root=args.data_root, pipeline=args.pipeline,
                         strategy=args.strategy, atlas=args.atlas,
                         n_components=args.qubits)
    series_dir = args.data_root / "ABIDE_pcp" / args.pipeline / args.strategy
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D")
         for s in dataset.subjects], kind="correlation")
    labels = np.asarray(dataset.labels)
    sites = np.asarray(dataset.sites)
    splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
              for s in sorted(set(sites.tolist()))]
    splits = [(s, tr, te) for s, tr, te in splits
              if len(te) >= args.min_test_size and len(np.unique(labels[te])) > 1]
    print(f"{len(labels)} subjects, {len(splits)} usable site folds -> {out_dir}",
          flush=True)

    if args.skip_observed:
        obs_acc = {m: float("nan") for m in MODELS}
        obs_auc = {m: float("nan") for m in MODELS}
        per_pass = float("nan")
    else:
        t0 = time.perf_counter()
        obs_acc, obs_auc = one_pass(connectivity, labels, sites, splits, args.qubits,
                                    args.bandwidths, args.kernel_steps, args.seed)
        per_pass = time.perf_counter() - t0
    if not args.skip_observed:
        print(f"observed ({per_pass:.1f}s): " +
              "  ".join(f"{m} acc={obs_acc[m]:.3f}" for m in MODELS), flush=True)
        print(f"estimated total for {args.permutations} permutations: "
              f"{per_pass * args.permutations / 60:.0f} min", flush=True)

    rng = np.random.default_rng(args.seed + 10_000 * args.shard)
    null_acc = {m: [] for m in MODELS}
    null_auc = {m: [] for m in MODELS}
    for i in range(args.permutations):
        y = shuffle_within_site(labels, sites, rng)
        a, u = one_pass(connectivity, y, sites, splits, args.qubits,
                        args.bandwidths, args.kernel_steps, args.seed)
        for m in MODELS:
            null_acc[m].append(a[m])
            if u[m] is not None:
                null_auc[m].append(u[m])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.permutations}  " +
                  "  ".join(f"{m} null-acc={np.mean(null_acc[m]):.3f}"
                            for m in MODELS), flush=True)

    payload = {"n_permutations": args.permutations, "seed": args.seed,
               "shuffle": "within-site", "observed_accuracy": obs_acc,
               "observed_auc": obs_auc, "results": {}}
    if args.skip_observed:
        payload["null_accuracy"] = {m: null_acc[m] for m in MODELS}
        payload["null_auc"] = {m: null_auc[m] for m in MODELS}
        (out_dir / "permutation_null.json").write_text(json.dumps(payload, indent=2))
        print(f"shard {args.shard}: {args.permutations} permutations -> "
              f"{out_dir/'permutation_null.json'}", flush=True)
        return
    print("\n" + "=" * 68)
    for m in MODELS:
        na = np.array(null_acc[m])
        p_acc = float((na >= obs_acc[m]).sum() + 1) / (args.permutations + 1)
        entry = {"observed_accuracy": obs_acc[m],
                 "null_accuracy_mean": float(na.mean()),
                 "null_accuracy_sd": float(na.std(ddof=1)),
                 "p_accuracy": p_acc}
        line = (f"{m:<15} acc {obs_acc[m]:.3f} vs null "
                f"{na.mean():.3f}+-{na.std(ddof=1):.3f}  p={p_acc:.4f}")
        if obs_auc[m] is not None and null_auc[m]:
            nu = np.array(null_auc[m])
            p_auc = float((nu >= obs_auc[m]).sum() + 1) / (len(nu) + 1)
            entry.update({"observed_auc": obs_auc[m],
                          "null_auc_mean": float(nu.mean()),
                          "null_auc_sd": float(nu.std(ddof=1)),
                          "p_auc": p_auc})
            line += f" | auc {obs_auc[m]:.3f} vs {nu.mean():.3f}  p={p_auc:.4f}"
        payload["results"][m] = entry
        print(line)
    payload["null_accuracy"] = {m: null_acc[m] for m in MODELS}
    payload["null_auc"] = {m: null_auc[m] for m in MODELS}
    (out_dir / "permutation_null.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir/'permutation_null.json'}")


if __name__ == "__main__":
    main()
