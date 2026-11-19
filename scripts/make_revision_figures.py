#!/usr/bin/env python3
"""Figures added for the IEEE Access resubmission, in the paper's compact style.

  fig_c8_effdim   test AUC and train-test gap against effective dimension for the
                  fidelity and RBF kernels on all four cohorts (reviewer R2.6),
                  from results/revision/kgen_*/kernel_generalization.json

Usage:
    python scripts/make_revision_figures.py --out <manuscript folder>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COHORT_COLOURS = {"ABIDE I": "#0072B2", "ADHD-200": "#E69F00",
                  "REST-meta-MDD": "#009E73", "UCLA-CNP": "#CC79A7"}
TAGS = {"ABIDE I": "abide", "ADHD-200": "adhd", "REST-meta-MDD": "mdd", "UCLA-CNP": "ucla"}
W = 3.45  # IEEE Access column width, inches

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.pad_inches": 0.02,
})


def fig_c8_effdim(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75))
    for name, tag in TAGS.items():
        table = json.loads(Path(f"results/revision/kgen_{tag}/kernel_generalization.json")
                           .read_text())["table"]
        col = COHORT_COLOURS[name]
        for family, style in (("quantum", "o-"), ("rbf", "s--")):
            rows = sorted((t for t in table if t["family"] == family and t["n_features"] == 8),
                          key=lambda t: t["effective_dimension"])
            x = [t["effective_dimension"] for t in rows]
            axes[0].plot(x, [t["test_auc"] for t in rows], style, color=col, ms=2.2, lw=0.9)
            axes[1].plot(x, [t["gap"] for t in rows], style, color=col, ms=2.2, lw=0.9)
    for ax, label, ylabel in ((axes[0], "(a)", "test AUC"), (axes[1], "(b)", "train$-$test gap")):
        ax.set_xscale("log")
        ax.set_xlabel("effective dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(label, loc="left", fontsize=7.2, pad=2)
    axes[0].axhline(0.5, color="grey", ls=":", lw=0.7)
    handles = [Line2D([], [], color=c, lw=1.4, label=n) for n, c in COHORT_COLOURS.items()]
    handles += [Line2D([], [], color="#333", ls="-", marker="o", ms=2.2, label="quantum fidelity"),
                Line2D([], [], color="#333", ls="--", marker="s", ms=2.2, label="classical RBF")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.17), handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_c8_effdim.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'fig_c8_effdim.pdf'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_c8_effdim(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
