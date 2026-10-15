#!/usr/bin/env python3
"""Sweep every matched quantum/classical comparison and screen the wins.

Reproduces the advantage-sweep table of the paper from the per-fold records in
results/rerun22 and the pooled confusion matrices in results/full_metrics.json
(written by scripts/make_metrics.py).

For each matched pair, cohort and metric (accuracy, F1, AUC, specificity):

- "ahead" counts a positive median per-fold difference for the quantum arm;
- "sig." counts ahead with p < 0.05 on a paired Wilcoxon signed-rank test;
- "clean" keeps a significant win only if neither arm is degenerate on its
  confusion matrix pooled over folds (sensitivity or specificity below 0.30,
  or |MCC| < 0.02) and the comparator is not C-CNN, which fails to train on
  the small cohorts (Section VIII-C of the paper).

Usage:
    python scripts/advantage_sweep.py
    python scripts/advantage_sweep.py --list
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS = Path("results/rerun22")
COHORTS = {"abide": "ABIDE-I", "adhd": "ADHD-200", "mdd": "REST-meta-MDD", "ucla": "UCLA-CNP"}
METRICS = ("accuracy", "f1", "auc", "specificity")

# Each quantum model against the classical model it is matched to.
PAIRS = {
    "qmodels": ("quantum_models_results.json", [
        ("QCNN", "CNN (1-D)"), ("VQC", "MLP"), ("QSVM-Pegasos", "SVM-RBF"),
        ("QSVM-fixed", "SVM-RBF"), ("TQEK", "Trainable-RBF")]),
    "planA": ("hybrid_results.json", [
        ("H-QCNN", "H-CNN"), ("H-QCNN", "H-Linear"), ("H-VQC", "H-MLP")]),
    "planB": ("ensemble_results.json", [
        ("Q-Ensemble", "C-Ensemble"), ("Q-Ensemble", "C-Single")]),
    "planC": ("reupload_results.json", [
        (q, c) for q in ("Q-Plain", "Q-Reup1", "Q-Reup2") for c in ("C-CNN", "C-MLP", "C-Linear")]),
}
FAILS_TO_TRAIN = {"C-CNN"}


def degenerate(m: dict) -> bool:
    mcc = m["mcc"]
    return (not math.isfinite(mcc) or abs(mcc) < 0.02
            or min(m["sensitivity"], m["specificity"]) < 0.30)


def sweep() -> list[dict]:
    pooled = json.loads(Path("results/full_metrics.json").read_text())
    rows = []
    for plan, (fname, pairs) in PAIRS.items():
        for tag, cohort in COHORTS.items():
            path = RESULTS / f"{plan}_{tag}" / fname
            if not path.exists():
                continue
            pf = json.loads(path.read_text())["per_fold"]
            run = pooled[f"{plan}_{tag}"]
            for q, c in pairs:
                deg = degenerate(run[q]) or degenerate(run[c])
                for metric in METRICS:
                    a = np.array([np.nan if f.get(metric) is None else f[metric]
                                  for f in pf[q]], float)
                    b = np.array([np.nan if f.get(metric) is None else f[metric]
                                  for f in pf[c]], float)
                    ok = ~(np.isnan(a) | np.isnan(b))
                    if ok.sum() < 3:      # PegasosQSVC reports no AUC
                        continue
                    diff = a[ok] - b[ok]
                    p = wilcoxon(a[ok], b[ok])[1] if np.any(diff != 0) else 1.0
                    med = float(np.median(diff))
                    sig = med > 0 and p < 0.05
                    rows.append({"cohort": cohort, "quantum": q, "classical": c, "metric": metric,
                                 "median": med, "p": float(p), "ahead": med > 0, "sig": sig,
                                 "clean": sig and not deg and c not in FAILS_TO_TRAIN})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="also list every significant win")
    args = ap.parse_args()

    rows = sweep()
    print(f"{'metric':<12}{'pairs':>6}{'ahead':>7}{'sig.':>6}{'clean':>7}")
    for metric in METRICS + ("total",):
        r = rows if metric == "total" else [x for x in rows if x["metric"] == metric]
        print(f"{metric:<12}{len(r):>6}{sum(x['ahead'] for x in r):>7}"
              f"{sum(x['sig'] for x in r):>6}{sum(x['clean'] for x in r):>7}")
    if args.list:
        print()
        for x in rows:
            if x["sig"]:
                print(f"  {'clean' if x['clean'] else '     '}  {x['cohort']:<14}"
                      f"{x['quantum']:<13}vs {x['classical']:<14}{x['metric']:<12}"
                      f"{x['median']:+.3f}  p={x['p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
