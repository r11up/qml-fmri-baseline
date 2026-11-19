#!/usr/bin/env python3
"""Tensor-train-guided quantum models against parameter-matched classical twins.

Adds the two structured variational designs of Qi et al. to the matched
comparison, on the full 2,000-feature budget their tensor-train (TT) front-end
is built to compress:

TTN-VQC          TT layer (2,000 -> 8) -> RY encoding -> six blocks of
                 [CNOT ring, RX, RY, RZ] -> Pauli-Z -> linear head
                 (npj Quantum Inf. 9:4, 2023).
TT-MLP           Its twin: the same TT layer and head, with the circuit replaced
                 by two 8x8 tanh layers holding the same 144 parameters.
TensorHyper-VQC  A TT hypernetwork (2,000 -> 144) writes every rotation angle of
                 a six-block ring circuit whose RY encoding takes the top eight
                 features (npj Quantum Inf. 12:70, 2026).
TT-HyperMLP      Its twin: the same hypernetwork writes the weights of two 8x8
                 tanh layers applied to the same eight features.

Each quantum model and its twin have identical parameter counts (530 and 594),
see identical features and folds, and share the suite's training loop (full
batch, Adam, learning rate 0.05, 150 epochs, class-weighted cross-entropy).
Linear and RBF SVMs on the same 2,000 features are recorded for reference.

With --inner-select, every arm instead chooses its learning rate and stopping
epoch inside each training fold: a stratified 20% of the training subjects is
held out, each learning rate is trained for up to --epochs epochs and scored by
balanced accuracy every ten epochs, and the best setting is retrained on the
whole training fold. The test site takes no part in the choice, and all four
arms are tuned identically.

Features are selected inside each training fold by Welch t-test, exactly as in
the model suite; the first eight are the suite's eight, so the encoder of
TensorHyper-VQC sees the same angles as the suite's QCNN and VQC.

Examples:
    python scripts/run_tensor_vqc.py --data-root data/ABIDE-I
    python scripts/run_tensor_vqc.py --data-root data/UCLA-CNP-cc200 --cv stratified
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from run_quantum_models import metrics, select_features, train_torch_classifier
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.tensor_vqc import (
    TTMLP,
    TTNVQC,
    TensorHyperVQC,
    TTHyperMLP,
    count_parameters,
)

N_FEATURES = 2000
N_QUBITS = 8
DEPTH = 6
COMPRESS = dict(in_modes=[10, 10, 20], out_modes=[2, 2, 2], ranks=[1, 3, 3, 1])
HYPER = dict(in_modes=[5, 8, 5, 10], out_modes=[4, 4, 3, 3], ranks=[1, 2, 2, 2, 1])

BUILDERS = {
    "TTN-VQC": lambda seed: TTNVQC(**COMPRESS, depth=DEPTH, seed=seed),
    "TT-MLP": lambda seed: TTMLP(**COMPRESS, depth=DEPTH, seed=seed),
    "TensorHyper-VQC": lambda seed: TensorHyperVQC(N_QUBITS, **HYPER, depth=DEPTH, seed=seed),
    "TT-HyperMLP": lambda seed: TTHyperMLP(N_QUBITS, **HYPER, depth=DEPTH, seed=seed),
}
HYPER_INPUT = {"TensorHyper-VQC", "TT-HyperMLP"}
PAIRS = [("TTN-VQC", "TT-MLP"), ("TensorHyper-VQC", "TT-HyperMLP")]
METRICS = ("accuracy", "f1", "auc", "specificity")


def select_then_train(build, x_train, y_train, x_test, seed, lrs, max_epochs, every=10):
    """Pick the learning rate and stopping epoch on an inner split, then retrain."""
    fit, val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                           random_state=seed).split(x_train, y_train))
    xf = torch.as_tensor(x_train[fit], dtype=torch.float32)
    yf = torch.as_tensor(y_train[fit], dtype=torch.long)
    xv = torch.as_tensor(x_train[val], dtype=torch.float32)
    counts = torch.bincount(yf, minlength=2).float()
    weight = counts.sum() / (2 * counts.clamp_min(1))
    best = (-1.0, lrs[0], max_epochs)
    for lr in lrs:
        torch.manual_seed(seed)
        model = build(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            F.cross_entropy(model(xf), yf, weight=weight).backward()
            optimizer.step()
            if epoch % every == 0:
                model.eval()
                with torch.no_grad():
                    score = balanced_accuracy_score(y_train[val], model(xv).argmax(1).numpy())
                if score > best[0]:
                    best = (float(score), lr, epoch)
    score, lr, epochs = best
    preds, probs, train_acc = train_torch_classifier(build(seed), x_train, y_train, x_test,
                                                     epochs=epochs, lr=lr, seed=seed)
    return preds, probs, train_acc, {"lr": lr, "epochs": epochs, "inner_bacc": score}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--inner-select", action="store_true",
                        help="choose learning rate and stopping epoch inside each fold")
    parser.add_argument("--inner-lrs", type=float, nargs="+", default=[0.05, 0.005, 0.001])
    parser.add_argument("--cv", default="leave-site-out",
                        choices=["leave-site-out", "stratified"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_tensor_vqc"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"tensor-train quantum models -> {out_dir}", flush=True)

    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=N_QUBITS, limit=args.limit,
    )
    print(dataset.summary(), flush=True)
    series_dir = args.data_root / "ABIDE_pcp" / args.pipeline / args.strategy
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D") for s in dataset.subjects],
        kind="correlation",
    )
    labels, sites = dataset.labels, dataset.sites
    print(f"connectivity: {connectivity.shape}\n", flush=True)

    params = {name: count_parameters(build(args.seed)) for name, build in BUILDERS.items()}
    print("trainable parameters: " + "  ".join(f"{n}={c}" for n, c in params.items()), flush=True)

    names = list(BUILDERS) + ["SVM-linear (2,000)", "SVM-RBF (2,000)"]
    per_fold: dict[str, list[dict]] = {n: [] for n in names}
    timing: dict[str, float] = dict.fromkeys(names, 0.0)

    if args.cv == "stratified":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = [(f"fold{i+1}", tr, te) for i, (tr, te)
                  in enumerate(splitter.split(connectivity, labels))]
    else:
        splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                  for s in sorted(set(sites.tolist()))]

    for site, train_idx, test_idx in splits:
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue
        y_train, y_test = labels[train_idx], labels[test_idx]

        chosen = select_features(connectivity[train_idx], y_train, N_FEATURES)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        x_train = scaler.transform(connectivity[train_idx][:, chosen])
        x_test = scaler.transform(connectivity[test_idx][:, chosen])

        # The suite's variational convention for the eight encoded features.
        enc = MinMaxScaler((0, np.pi / 2)).fit(x_train[:, :N_QUBITS])
        e_train = enc.transform(x_train[:, :N_QUBITS])
        e_test = np.clip(enc.transform(x_test[:, :N_QUBITS]), 0, np.pi / 2)
        h_train, h_test = np.hstack([e_train, x_train]), np.hstack([e_test, x_test])

        n_test = len(test_idx)

        def record(name, preds, scores, elapsed, train_acc, chosen_cfg=None,
                   _site=site, _n=n_test, _y=y_test):
            per_fold[name].append({"site": _site, "n": _n, "train_accuracy": train_acc,
                                   **({"chosen": chosen_cfg} if chosen_cfg else {}),
                                   **metrics(_y, preds, scores)})
            timing[name] += elapsed

        for name, build in BUILDERS.items():
            a, b = (h_train, h_test) if name in HYPER_INPUT else (x_train, x_test)
            t0 = time.perf_counter()
            chosen_cfg = None
            if args.inner_select:
                preds, scores, train_acc, chosen_cfg = select_then_train(
                    build, a, y_train, b, args.seed, args.inner_lrs, args.epochs)
            else:
                preds, scores, train_acc = train_torch_classifier(
                    build(args.seed), a, y_train, b,
                    epochs=args.epochs, lr=args.lr, seed=args.seed,
                )
            record(name, preds, scores, time.perf_counter() - t0, train_acc, chosen_cfg)

        for name, kernel in (("SVM-linear (2,000)", "linear"), ("SVM-RBF (2,000)", "rbf")):
            t0 = time.perf_counter()
            model = SVC(kernel=kernel, C=1.0, gamma="scale",
                        class_weight="balanced").fit(x_train, y_train)
            raw = model.decision_function(x_test)
            train_acc = float(((model.decision_function(x_train) > 0) == y_train).mean())
            record(name, (raw > 0).astype(int), raw, time.perf_counter() - t0, train_acc)

        print(f"  {site:<12} n={len(test_idx):<4} " + "  ".join(
            f"{n.split()[0]}={per_fold[n][-1]['accuracy']:.3f}" for n in names), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds
                           if f.get(key) is not None and np.isfinite(f[key])])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 92)
    print(f"TENSOR-TRAIN QUANTUM MODELS - {N_FEATURES} features, {N_QUBITS} qubits, depth {DEPTH}")
    print("=" * 92)
    print(f"{'model':<20}{'train':>8}{'accuracy':>15}{'F1':>15}{'AUC':>15}{'sec/fold':>10}")
    for name in names:
        row = f"{name:<20}{summarise(per_fold[name], 'train_accuracy')[0]:>8.3f}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(per_fold[name], key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        print(row + f"{timing[name] / max(len(per_fold[name]), 1):>10.1f}")

    tests = {}
    print()
    for quantum, classical in PAIRS:
        for metric in METRICS:
            q = np.array([f[metric] for f in per_fold[quantum]], float)
            c = np.array([f[metric] for f in per_fold[classical]], float)
            diff = q - c
            p = float(wilcoxon(q, c)[1]) if len(q) > 1 and np.any(diff != 0) else 1.0
            tests[f"{quantum} vs {classical} [{metric}]"] = {
                "median_diff": float(np.median(diff)), "p_value": p,
                "wins": int((diff > 0).sum()), "folds": int(len(q)),
            }
            print(f"{quantum} vs {classical} [{metric}]: median diff {np.median(diff):+.3f}, "
                  f"wins {int((diff > 0).sum())}/{len(q)}, p = {p:.4f}")

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir),
                                "n_features": N_FEATURES, "n_qubits": N_QUBITS,
                                "depth": DEPTH, "compress_tt": COMPRESS, "hyper_tt": HYPER},
        "n_subjects": len(dataset),
        "parameters": params,
        "summary": {n: {m: summarise(per_fold[n], m)
                        for m in ("train_accuracy", "accuracy", "f1", "auc", "specificity")}
                    for n in names},
        "seconds_per_fold": {n: timing[n] / max(len(per_fold[n]), 1) for n in names},
        "per_fold": per_fold, "paired_tests": tests, "timestamp": stamp,
    }
    (out_dir / "tensor_vqc_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'tensor_vqc_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
