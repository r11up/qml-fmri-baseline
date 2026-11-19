#!/usr/bin/env python3
"""Does effective dimension predict generalization? (reviewer R2.6)

Both kernels of the matched comparison, the fidelity kernel of QSVM-fixed and
the RBF kernel of SVM-RBF, are swept across their single width parameter
(encoding bandwidth; gamma) on every fold of a cohort, and the fidelity kernel
also across register width (4, 8 and 12 qubits on the top 4, 8 and 12
features). At every setting the training Gram matrix is summarised by

  effective dimension  participation ratio (sum lambda)^2 / sum lambda^2,
                       computed exactly as tr(K)^2 / ||K||_F^2
  concentration        variance of the off-diagonal kernel values; it vanishes
                       when every pair of subjects looks alike (Thanasilp et al.)
  expressibility       quantum only: KL divergence between the distribution of
                       off-diagonal fidelities and the Haar distribution
                       (N - 1)(1 - F)^(N - 2), N = 2^n, in 75 bins (Sim et al.);
                       lower means the data states fill Hilbert space more

and an SVM (C = 1, balanced classes) on that Gram matrix gives training
accuracy, test accuracy, test AUC and the train-test gap. Features are chosen
and scaled inside each training fold exactly as in the model suite.

The summary asks two questions. Within each kernel family, does effective
dimension track test performance and the gap (Spearman correlation across
settings)? And across families, does a quantum setting perform like an RBF
setting of the same effective dimension (the quantum value minus the RBF curve
interpolated in log effective dimension)?

Example:
    python scripts/run_kernel_generalization.py --data-root data/ABIDE-I --tag abide
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run_quantum_models import select_features
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix

BANDWIDTHS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
GAMMAS = [0.002, 0.005, 0.01, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0, 2.0]
QUBITS = [4, 8, 12]
HAAR_BINS = 75


def effective_dimension(k: np.ndarray) -> float:
    return float(np.trace(k) ** 2 / np.sum(k * k))


def off_diagonal(k: np.ndarray) -> np.ndarray:
    return k[np.triu_indices_from(k, k=1)]


def expressibility(fidelities: np.ndarray, n_qubits: int) -> float:
    """KL(P_data(F) || P_Haar(F)) over 75 equal bins of [0, 1]."""
    edges = np.linspace(0.0, 1.0, HAAR_BINS + 1)
    p, _ = np.histogram(np.clip(fidelities, 0.0, 1.0), bins=edges)
    p = p / p.sum()
    dim = 2.0**n_qubits
    q = (1.0 - edges[:-1]) ** (dim - 1) - (1.0 - edges[1:]) ** (dim - 1)
    eps = 1e-12
    return float(np.sum(np.where(p > 0, p * np.log((p + eps) / (q + eps)), 0.0)))


def evaluate(k_train, k_test, y_train, y_test):
    model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(k_train, y_train)
    train_acc = accuracy_score(y_train, (model.decision_function(k_train) > 0).astype(int))
    raw = model.decision_function(k_test)
    test_acc = accuracy_score(y_test, (raw > 0).astype(int))
    return {"train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
            "test_auc": float(roc_auc_score(y_test, raw)), "gap": float(train_acc - test_acc)}


def summarise(records):
    """Fold means per setting, Spearman correlations, and the one-curve test."""
    keys = sorted({(r["family"], r["n_features"], r["width"]) for r in records})
    table = []
    for family, n, width in keys:
        rows = [r for r in records
                if (r["family"], r["n_features"], r["width"]) == (family, n, width)]
        entry = {"family": family, "n_features": n, "width": width, "folds": len(rows)}
        for key in ("effective_dimension", "concentration", "expressibility", "train_accuracy",
                    "test_accuracy", "test_auc", "gap"):
            values = [r[key] for r in rows if r.get(key) is not None]
            entry[key] = float(np.mean(values)) if values else None
        table.append(entry)

    correlations = {}
    for family in ("quantum", "rbf"):
        rows = [t for t in table if t["family"] == family]
        for target in ("test_accuracy", "test_auc", "gap", "train_accuracy"):
            rho, p = spearmanr([t["effective_dimension"] for t in rows], [t[target] for t in rows])
            correlations[f"{family}: eff.dim vs {target}"] = {"rho": float(rho), "p": float(p),
                                                              "settings": len(rows)}
    quantum = [t for t in table if t["family"] == "quantum"]
    for target in ("test_auc", "test_accuracy"):
        for name, x in (("concentration", "concentration"), ("expressibility", "expressibility")):
            rho, p = spearmanr([t[x] for t in quantum], [t[target] for t in quantum])
            correlations[f"quantum: {name} vs {target}"] = {"rho": float(rho), "p": float(p),
                                                            "settings": len(quantum)}

    rbf = sorted((t for t in table if t["family"] == "rbf" and t["n_features"] == 8),
                 key=lambda t: t["effective_dimension"])
    x_rbf = np.log([t["effective_dimension"] for t in rbf])
    one_curve = {}
    for target in ("test_auc", "test_accuracy", "gap"):
        y_rbf = np.array([t[target] for t in rbf])
        deltas = []
        for t in quantum:
            x = np.log(t["effective_dimension"])
            if x_rbf[0] <= x <= x_rbf[-1]:
                deltas.append(t[target] - float(np.interp(x, x_rbf, y_rbf)))
        one_curve[target] = {"mean_delta": float(np.mean(deltas)),
                             "mean_abs_delta": float(np.mean(np.abs(deltas))),
                             "max_abs_delta": float(np.max(np.abs(deltas))),
                             "settings_in_range": len(deltas)}
    return table, correlations, one_curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cv", default="leave-site-out", choices=["leave-site-out", "stratified"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_kernel_generalization_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_abide(root=args.data_root, pipeline="cpac", strategy="filt_noglobal",
                         atlas="rois_cc200", n_components=8)
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

    records = []
    t_start = time.perf_counter()
    for site, tr, te in splits:
        if len(te) < args.min_test_size or len(np.unique(labels[te])) < 2:
            continue
        ytr, yte = labels[tr], labels[te]
        ranked = select_features(conn[tr], ytr, max(QUBITS))
        for n in QUBITS:
            cols = ranked[:n]
            scaler = StandardScaler().fit(conn[tr][:, cols])
            xtr, xte = scaler.transform(conn[tr][:, cols]), scaler.transform(conn[te][:, cols])
            angle = MinMaxScaler((0, np.pi)).fit(xtr)
            atr, ate = angle.transform(xtr), np.clip(angle.transform(xte), 0, np.pi)
            for bandwidth in BANDWIDTHS:
                fm = QuantumFeatureMap(n_qubits=n, reps=2, entanglement="linear",
                                       bandwidth=bandwidth)
                k_train = quantum_kernel_matrix(atr, atr, fm).astype(np.float64)
                k_test = quantum_kernel_matrix(ate, atr, fm).astype(np.float64)
                off = off_diagonal(k_train)
                records.append({"site": site, "family": "quantum", "n_features": n,
                                "width": bandwidth,
                                "effective_dimension": effective_dimension(k_train),
                                "concentration": float(off.var()),
                                "expressibility": expressibility(off, n),
                                **evaluate(k_train, k_test, ytr, yte)})
            sq_tr = cdist(xtr, xtr, "sqeuclidean")
            sq_te = cdist(xte, xtr, "sqeuclidean")
            for gamma in GAMMAS:
                k_train, k_test = np.exp(-gamma * sq_tr), np.exp(-gamma * sq_te)
                off = off_diagonal(k_train)
                records.append({"site": site, "family": "rbf", "n_features": n, "width": gamma,
                                "effective_dimension": effective_dimension(k_train),
                                "concentration": float(off.var()), "expressibility": None,
                                **evaluate(k_train, k_test, ytr, yte)})
        print(f"  {site:<12} done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    table, correlations, one_curve = summarise(records)
    print()
    for t in table:
        print(f"  {t['family']:<8} n={t['n_features']:<3} width={t['width']:<6} "
              f"effdim={t['effective_dimension']:8.1f}  conc={t['concentration']:.2e}  "
              f"train={t['train_accuracy']:.3f} test={t['test_accuracy']:.3f} "
              f"auc={t['test_auc']:.3f} gap={t['gap']:+.3f}")
    print()
    for k, v in correlations.items():
        print(f"  {k:<44} rho={v['rho']:+.2f}  p={v['p']:.3g}")
    print("  one curve (quantum minus RBF at equal effective dimension):", one_curve)

    (out_dir / "kernel_generalization.json").write_text(json.dumps(
        {"config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
         "bandwidths": BANDWIDTHS, "gammas": GAMMAS, "qubits": QUBITS,
         "records": records, "table": table, "correlations": correlations,
         "one_curve": one_curve, "timestamp": stamp}, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'kernel_generalization.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
