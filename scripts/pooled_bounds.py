#!/usr/bin/env python3
"""Random-effects bounds on the three central matched differences (Fig. 10).

Per cohort, the mean paired accuracy difference (quantum minus classical) and
the variance of that mean are pooled with the DerSimonian-Laird random-effects
model, which lets the true difference vary between cohorts. Reported with all
four cohorts, as in the paper, and without UCLA-CNP, whose folds are
within-site rather than leave-site-out.

  quantum kernel vs matched RBF   results/*_benchmark (Fig. 4 runs)
  QCNN vs 1-D CNN                 results/rerun22/qmodels_*
  hybrid QCNN vs hybrid CNN       results/rerun22/planA_*

Usage:
    python scripts/pooled_bounds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BENCH = {"abide": "ABIDE_benchmark", "adhd": "ADHD200_benchmark",
         "mdd": "MDD_benchmark", "ucla": "UCLA_benchmark"}
PAIRS = {
    "quantum kernel vs matched RBF": (
        lambda t: Path("results") / BENCH[t] / "benchmark_results.json",
        "quantum kernel", "SVM RBF (matched)"),
    "QCNN vs 1-D CNN": (
        lambda t: Path("results/rerun22") / f"qmodels_{t}" / "quantum_models_results.json",
        "QCNN", "CNN (1-D)"),
    "hybrid QCNN vs hybrid CNN": (
        lambda t: Path("results/rerun22") / f"planA_{t}" / "hybrid_results.json",
        "H-QCNN", "H-CNN"),
}


def dersimonian_laird(effects: np.ndarray, variances: np.ndarray) -> tuple[float, float, float]:
    w = 1 / variances
    fixed = np.sum(w * effects) / w.sum()
    q = np.sum(w * (effects - fixed) ** 2)
    c = w.sum() - np.sum(w**2) / w.sum()
    tau2 = max(0.0, (q - (len(effects) - 1)) / c) if c > 0 else 0.0
    ws = 1 / (variances + tau2)
    estimate = float(np.sum(ws * effects) / ws.sum())
    se = float(np.sqrt(1 / ws.sum()))
    return estimate, estimate - 1.96 * se, estimate + 1.96 * se


def main() -> int:
    for label, cohorts in (("all four cohorts", ("abide", "adhd", "mdd", "ucla")),
                           ("without UCLA-CNP", ("abide", "adhd", "mdd"))):
        print(label)
        for name, (path, quantum, classical) in PAIRS.items():
            effects, variances = [], []
            for tag in cohorts:
                folds = json.loads(path(tag).read_text())["per_fold"]
                diff = np.array([a["accuracy"] - b["accuracy"]
                                 for a, b in zip(folds[quantum], folds[classical], strict=True)])
                effects.append(diff.mean())
                variances.append(diff.var(ddof=1) / len(diff))
            est, low, high = dersimonian_laird(np.array(effects), np.array(variances))
            print(f"  {name:<30} {est:+.3f}  (95% CI {low:+.3f} to {high:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
