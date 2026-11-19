#!/usr/bin/env python3
"""Effect sizes and confidence intervals for every matched comparison.

Extends the advantage sweep (scripts/advantage_sweep.py, same 284 comparisons
and the same per-fold records in results/rerun22) with three quantities per
comparison, so every difference is reported the same way:

- the median paired difference (quantum minus classical) across folds, with a
  95% percentile bootstrap interval (10,000 resamples of folds, seed 0);
- the matched-pairs rank-biserial correlation, (W+ - W-) / (W+ + W-) over the
  signed ranks that the Wilcoxon test uses (zero differences dropped), which
  runs from -1 (classical ahead on every fold) to +1 (quantum ahead on every fold);
- Benjamini-Hochberg q-values across all comparisons, next to the Bonferroni
  threshold already used in the paper.

Usage:
    python scripts/effect_sizes.py                       # summary
    python scripts/effect_sizes.py --csv results/revision/effect_sizes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from advantage_sweep import COHORTS, FAILS_TO_TRAIN, METRICS, PAIRS, RESULTS, degenerate
from scipy.stats import rankdata, wilcoxon

RESAMPLES = 10_000


def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((plus - minus) / (plus + minus))


def bootstrap_median(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(RESAMPLES, len(diff)))
    medians = np.median(diff[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def rows() -> list[dict]:
    pooled = json.loads(Path("results/full_metrics.json").read_text())
    rng = np.random.default_rng(0)
    out = []
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
                    if ok.sum() < 3:
                        continue
                    diff = a[ok] - b[ok]
                    p = wilcoxon(a[ok], b[ok])[1] if np.any(diff != 0) else 1.0
                    low, high = bootstrap_median(diff, rng)
                    med = float(np.median(diff))
                    sig = med > 0 and p < 0.05
                    out.append({
                        "cohort": cohort, "plan": plan, "quantum": q, "classical": c,
                        "metric": metric, "folds": int(ok.sum()), "median_diff": med,
                        "ci_low": low, "ci_high": high, "rank_biserial": rank_biserial(diff),
                        "p": float(p), "sig": sig,
                        "clean": sig and not deg and c not in FAILS_TO_TRAIN,
                    })
    q = benjamini_hochberg(np.array([r["p"] for r in out]))
    for r, value in zip(out, q, strict=True):
        r["q_bh"] = float(value)
        r["bonferroni_pass"] = r["p"] < 0.05 / len(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="write every comparison to this file")
    args = ap.parse_args()

    table = rows()
    n = len(table)
    q_ahead = [r for r in table if r["ci_low"] > 0]
    c_ahead = [r for r in table if r["ci_high"] < 0]
    print(f"{n} matched comparisons")
    print(f"  95% interval of the median difference excludes zero: {len(q_ahead)} in favour "
          f"of the quantum arm, {len(c_ahead)} in favour of the classical arm")
    print(f"  Benjamini-Hochberg q < 0.05: {sum(r['q_bh'] < 0.05 for r in table)}; "
          f"Bonferroni (p < {0.05 / n:.1e}): {sum(r['bonferroni_pass'] for r in table)}")
    print(f"  smallest q: {min(r['q_bh'] for r in table):.3f}")
    rb = np.array([r["rank_biserial"] for r in table])
    print(f"  rank-biserial r: median {np.median(rb):+.2f}, "
          f"|r| >= 0.5 in {int((np.abs(rb) >= 0.5).sum())} comparisons")
    print("\n  the clean wins:")
    for r in table:
        if r["clean"]:
            print(f"    {r['cohort']:<14}{r['quantum']:<11} vs {r['classical']:<14}"
                  f"{r['metric']:<12}"
                  f"{r['median_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
                  f"r={r['rank_biserial']:+.2f}  p={r['p']:.4f}  q={r['q_bh']:.3f}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
