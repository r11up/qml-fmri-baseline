"""What a quantum kernel costs to run: shot budgets and projected kernels.

The exact-simulation results elsewhere in this project are upper bounds: every
kernel entry is computed from statevectors, as if each overlap had been
measured infinitely often. Hardware pays per shot, and a fidelity kernel needs
one circuit per *pair* of subjects.

This script measures what that costs:

1. **Shot budgets.** The fidelity kernel is estimated by the compute-uncompute
   routine, whose outcome is Bernoulli with p = |<psi_i|psi_j>|^2, so N shots
   give Binomial(N, p) / N. Sampling that for a range of N gives the accuracy
   of the pipeline as a function of the shot budget.
2. **PSD repair.** A sampled Gram matrix is not positive semi-definite. The
   script records the smallest eigenvalue and compares an SVM trained on the
   raw estimate with one trained after clipping negative eigenvalues.
3. **Projected kernels.** Local measurements avoid the pairwise cost: each
   subject needs 3n single-qubit expectations, not one circuit per pair. The
   projected kernel is exp(-gamma * sum_k ||rho_k(x) - rho_k(x')||_F^2), which
   for single-qubit reduced states is half the squared Bloch distance.
4. **Resolvability.** Kernel values concentrate as the register widens. Shot
   noise has scale 1/sqrt(N), so a register is only usable while the kernel's
   own spread exceeds that floor. The script reports both.

Everything runs under the same leave-site-out protocol, feature selection and
bandwidth choice as the exact-simulation pipeline, so the only thing that
changes is how the kernel is obtained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from qagta.data.descriptors import build_descriptors  # noqa: E402
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix  # noqa: E402

BANDWIDTHS = (0.01, 0.05, 0.1, 0.25, 0.5)
GAMMAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def load_abide(root: Path, pipeline: str, strategy: str, atlas: str, min_t: int):
    """Correlation vectors, labels and sites for every usable ABIDE I subject."""
    table = pd.read_csv(root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
    table = table[table["FILE_ID"] != "no_filename"]
    series_dir = root / "ABIDE_pcp" / pipeline / strategy
    series, labels, sites = [], [], []
    for row in table.itertuples():
        path = series_dir / f"{row.FILE_ID}_{atlas}.1D"
        if not path.exists():
            continue
        try:
            ts = np.loadtxt(path)
        except (ValueError, OSError):
            continue
        if ts.ndim != 2 or ts.shape[0] < min_t:
            continue
        series.append(ts)
        labels.append(1 if int(row.DX_GROUP) == 1 else 0)
        sites.append(str(row.SITE_ID))
    return build_descriptors(series, kind="correlation"), np.array(labels), np.array(sites), series


def select_features(x_train, y_train, k):
    from scipy import stats
    a, b = x_train[y_train == 0], x_train[y_train == 1]
    t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.argsort(-np.nan_to_num(np.abs(t)))[: min(k, x_train.shape[1])]


def alignment(k_matrix, y_signed):
    ideal = np.outer(y_signed, y_signed)
    denom = np.linalg.norm(k_matrix) * np.linalg.norm(ideal)
    return float((k_matrix * ideal).sum() / denom) if denom > 0 else 0.0


def bloch(states: torch.Tensor, n_qubits: int) -> np.ndarray:
    """Single-qubit Bloch vectors (<X>, <Y>, <Z> per qubit) from statevectors."""
    a = states.numpy() if isinstance(states, torch.Tensor) else np.asarray(states)
    dim = a.shape[1]
    idx = np.arange(dim)
    out = []
    for k in range(n_qubits):
        flip = idx ^ (1 << (n_qubits - k - 1))
        bit = (idx >> (n_qubits - k - 1)) & 1
        partner = a[:, flip]
        out.append(np.real(np.sum(np.conj(a) * partner, axis=1)))                       # <X>
        out.append(np.real(np.sum(np.conj(a) * (1j * np.where(bit == 0, 1.0, -1.0))
                                  * partner, axis=1)))                                  # <Y>
        out.append(np.sum(np.abs(a) ** 2 * (1 - 2 * bit), axis=1))                      # <Z>
    return np.stack(out, axis=1)


def projected_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma * sum_k ||rho_k(x) - rho_k(x')||_F^2); Bloch form is half the squared distance."""
    d2 = ((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * 0.5 * d2)


def psd_repair(k):
    """Clip negative eigenvalues; returns (repaired matrix, smallest eigenvalue)."""
    w, v = np.linalg.eigh((k + k.T) / 2)
    return (v * np.clip(w, 0, None)) @ v.T, float(w.min())


def scores(y_true, raw):
    pred = (raw > 0).astype(int)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "auc": (float(roc_auc_score(y_true, raw)) if len(np.unique(y_true)) == 2
                    else float("nan"))}


def fit_precomputed(k_train, y_train, k_test, y_test):
    model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(k_train, y_train)
    return scores(y_test, model.decision_function(k_test))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--shots", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--depolarising", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4],
                    help="global depolarising rate applied to each prepared state")
    ap.add_argument("--resolvability-subjects", type=int, default=400)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/shot_budget_abide.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading ABIDE I ...", flush=True)
    conn, labels, sites, _ = load_abide(args.data_root, args.pipeline, args.strategy,
                                        args.atlas, 60)
    print(f"{conn.shape[0]} subjects, {conn.shape[1]} connections, "
          f"{len(set(sites))} sites", flush=True)

    per_fold: dict[str, list] = {}
    started = time.perf_counter()
    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue
        y_train, y_test = labels[train_idx], labels[test_idx]
        y_signed = np.where(y_train > 0, 1.0, -1.0)

        chosen = select_features(conn[train_idx], y_train, args.top_k)
        angle_idx = chosen[: args.qubits]
        angle_scaler = MinMaxScaler((0, np.pi)).fit(conn[train_idx][:, angle_idx])
        train_ang = angle_scaler.transform(conn[train_idx][:, angle_idx])
        test_ang = np.clip(angle_scaler.transform(conn[test_idx][:, angle_idx]), 0, np.pi)

        # bandwidth by kernel-target alignment on the training split
        best = None
        for bw in BANDWIDTHS:
            fm = QuantumFeatureMap(n_qubits=args.qubits, reps=2, entanglement="linear",
                                   bandwidth=bw)
            k_tr = quantum_kernel_matrix(train_ang, train_ang, fm)
            a = alignment(k_tr, y_signed)
            if best is None or a > best[0]:
                best = (a, bw, fm, k_tr)
        _, bandwidth, fm, k_train_exact = best
        k_test_exact = quantum_kernel_matrix(test_ang, train_ang, fm)

        row = {"site": site, "n": len(test_idx), "bandwidth": bandwidth}
        row["exact"] = fit_precomputed(k_train_exact.astype(np.float64), y_train,
                                       k_test_exact.astype(np.float64), y_test)

        # classical matched control, identical features
        sc = StandardScaler().fit(conn[train_idx][:, angle_idx])
        rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(
            sc.transform(conn[train_idx][:, angle_idx]), y_train)
        row["rbf_matched"] = scores(
            y_test, rbf.decision_function(sc.transform(conn[test_idx][:, angle_idx])))

        # shot budgets on the fidelity kernel
        row["shots"] = {}
        for n_shots in args.shots:
            kt = rng.binomial(n_shots, np.clip(k_train_exact, 0, 1)).astype(np.float64) / n_shots
            kt = (kt + kt.T) / 2
            np.fill_diagonal(kt, 1.0)
            ks = rng.binomial(n_shots, np.clip(k_test_exact, 0, 1)).astype(np.float64) / n_shots
            raw_fit = fit_precomputed(kt, y_train, ks, y_test)
            repaired, min_eig = psd_repair(kt)
            rep_fit = fit_precomputed(repaired, y_train, ks, y_test)
            row["shots"][str(n_shots)] = {"raw": raw_fit, "repaired": rep_fit,
                                          "min_eigenvalue": min_eig}

        # why sampling survives: agreement of the sampled kernel and of the decision function
        from scipy.stats import spearmanr
        base = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(
            k_train_exact.astype(np.float64), y_train)
        exact_scores = base.decision_function(k_test_exact.astype(np.float64))
        row["agreement"] = {}
        for n_shots in args.shots:
            kt = rng.binomial(n_shots, np.clip(k_train_exact, 0, 1)).astype(np.float64) / n_shots
            kt = (kt + kt.T) / 2
            np.fill_diagonal(kt, 1.0)
            ks = rng.binomial(n_shots, np.clip(k_test_exact, 0, 1)).astype(np.float64) / n_shots
            m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(kt, y_train)
            row["agreement"][str(n_shots)] = {
                "kernel_spearman": float(spearmanr(kt.ravel(), k_train_exact.ravel()).statistic),
                "decision_pearson": float(np.corrcoef(m.decision_function(ks), exact_scores)[0, 1]),
                "mean_abs_entry_error": float(np.abs(kt - k_train_exact).mean())}

        # device noise: a global depolarising channel shrinks overlaps toward 1/2^n
        row["depolarising"] = {}
        floor = 1.0 / (2 ** args.qubits)
        for lam in args.depolarising:
            kt = (1 - lam) ** 2 * k_train_exact + (1 - (1 - lam) ** 2) * floor
            ks = (1 - lam) ** 2 * k_test_exact + (1 - (1 - lam) ** 2) * floor
            kt = rng.binomial(1024, np.clip(kt, 0, 1)).astype(np.float64) / 1024
            kt = (kt + kt.T) / 2
            np.fill_diagonal(kt, 1.0)
            ks = rng.binomial(1024, np.clip(ks, 0, 1)).astype(np.float64) / 1024
            row["depolarising"][str(lam)] = fit_precomputed(kt, y_train, ks, y_test)

        # projected kernel: exact, then shot-sampled local expectations
        states_tr = fm(torch.as_tensor(train_ang, dtype=torch.float32))
        states_te = fm(torch.as_tensor(test_ang, dtype=torch.float32))
        b_tr, b_te = bloch(states_tr, args.qubits), bloch(states_te, args.qubits)
        best_p = None
        for g in GAMMAS:
            kp = projected_kernel(b_tr, b_tr, g)
            a = alignment(kp, y_signed)
            if best_p is None or a > best_p[0]:
                best_p = (a, g, kp)
        _, gamma, kp_train = best_p
        row["projected"] = {"gamma": gamma,
                            "exact": fit_precomputed(kp_train, y_train,
                                                     projected_kernel(b_te, b_tr, gamma), y_test),
                            "shots": {}}
        for n_shots in args.shots:
            noisy_tr = 2.0 * rng.binomial(n_shots, np.clip((1 + b_tr) / 2, 0, 1)) / n_shots - 1.0
            noisy_te = 2.0 * rng.binomial(n_shots, np.clip((1 + b_te) / 2, 0, 1)) / n_shots - 1.0
            row["projected"]["shots"][str(n_shots)] = fit_precomputed(
                projected_kernel(noisy_tr, noisy_tr, gamma), y_train,
                projected_kernel(noisy_te, noisy_tr, gamma), y_test)

        row["projected"]["depolarising"] = {}
        for lam in args.depolarising:
            nt = 2.0 * rng.binomial(1024, np.clip((1 + (1 - lam) * b_tr) / 2, 0, 1)) / 1024 - 1.0
            ne = 2.0 * rng.binomial(1024, np.clip((1 + (1 - lam) * b_te) / 2, 0, 1)) / 1024 - 1.0
            row["projected"]["depolarising"][str(lam)] = fit_precomputed(
                projected_kernel(nt, nt, gamma), y_train, projected_kernel(ne, nt, gamma), y_test)

        per_fold.setdefault("folds", []).append(row)
        e = row["exact"]["accuracy"]
        print(f"  {site:<12} n={len(test_idx):<4} exact {e:.3f}  "
              f"rbf {row['rbf_matched']['accuracy']:.3f}  "
              f"1024 shots {row['shots']['1024']['repaired']['accuracy']:.3f}  "
              f"projected {row['projected']['exact']['accuracy']:.3f}", flush=True)

    # resolvability: kernel spread against the shot-noise floor, by register width
    print("\nresolvability by register width", flush=True)
    sub = rng.choice(conn.shape[0], size=min(args.resolvability_subjects, conn.shape[0]),
                     replace=False)
    top = select_features(conn, labels, args.top_k)
    resolvability = []
    for width in args.widths:
        idx = top[:width]
        ang = MinMaxScaler((0, np.pi)).fit_transform(conn[sub][:, idx])
        fmw = QuantumFeatureMap(n_qubits=width, reps=2, entanglement="linear", bandwidth=0.25)
        kw = quantum_kernel_matrix(ang, ang, fmw).astype(np.float64)
        off = kw[~np.eye(kw.shape[0], dtype=bool)]
        bw_ = bloch(fmw(torch.as_tensor(ang, dtype=torch.float32)), width)
        kp = projected_kernel(bw_, bw_, 0.5)
        offp = kp[~np.eye(kp.shape[0], dtype=bool)]
        entry = {"qubits": width, "fidelity_mean": float(off.mean()),
                 "fidelity_std": float(off.std()),
                 "projected_std": float(offp.std()),
                 "shots_to_resolve_fidelity": float(
                     off.mean() * (1 - off.mean()) / max(off.std(), 1e-12) ** 2)}
        resolvability.append(entry)
        print(f"  {width:2d} qubits: fidelity mean {entry['fidelity_mean']:.4f} "
              f"sd {entry['fidelity_std']:.4f}; "
              f"projected sd {entry['projected_std']:.4f}; shots to resolve one sd "
              f"{entry['shots_to_resolve_fidelity']:.0f}", flush=True)

    payload = {"config": {k: str(v) for k, v in vars(args).items()},
               "n_subjects": int(conn.shape[0]), "seconds": round(time.perf_counter() - started),
               "resolvability": resolvability, **per_fold}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
