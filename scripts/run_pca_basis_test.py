"""Does a per-subject PCA basis destroy the node features? (reviewer question 1)

FINDING 06 established that per-region node features classify at chance
(AUC 0.459) while the raw correlation vector reaches 0.693, and concluded the
node features carry no class signal. A reviewer proposed an alternative
explanation: the PCA basis is fitted per subject, so component k means
something different for every subject, and any class-discriminative structure
is scrambled rather than absent.

This script separates the two. Under identical leave-site-out folds it compares

  per-subject   PCA fitted on each subject's own connectivity profiles
                (what the graph experiments used)
  fold-wise     one PCA basis fitted on the training sites' profiles and
                applied unchanged to the held-out site (the reviewer's fix)
  ceiling       the flattened correlation vector, no compression

If the reviewer is right, fold-wise lands well above per-subject. If FINDING 06
is right, the two agree and both sit at chance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qagta.graph.connectome import pearson_connectivity  # noqa: E402


def load_profiles(root: Path, pipeline: str, strategy: str, atlas: str,
                  min_timepoints: int, limit: int | None):
    """Correlation profiles, labels and sites for every usable subject."""
    phenotypic = pd.read_csv(root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
    phenotypic = phenotypic[phenotypic["FILE_ID"] != "no_filename"]
    if limit:
        phenotypic = phenotypic.head(limit)
    series_dir = root / "ABIDE_pcp" / pipeline / strategy

    profiles, labels, sites = [], [], []
    for row in phenotypic.itertuples():
        path = series_dir / f"{row.FILE_ID}_{atlas}.1D"
        if not path.exists():
            continue
        try:
            ts = np.loadtxt(path)
        except (ValueError, OSError):
            continue
        if ts.ndim != 2 or ts.shape[0] < min_timepoints:
            continue
        profiles.append(pearson_connectivity(ts).astype(np.float32))
        labels.append(1 if int(row.DX_GROUP) == 1 else 0)
        sites.append(str(row.SITE_ID))
    return profiles, np.array(labels), np.array(sites)


def auc_of(x_train, y_train, x_test, y_test) -> float:
    scaler = StandardScaler().fit(x_train)
    model = SVC(kernel="linear", C=1.0, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    return float(roc_auc_score(y_test, model.decision_function(scaler.transform(x_test))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--min-timepoints", type=int, default=60)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/pca_basis_test.json"))
    args = ap.parse_args()

    print("loading connectivity profiles ...", flush=True)
    profiles, labels, sites = load_profiles(
        args.data_root, args.pipeline, args.strategy, args.atlas,
        args.min_timepoints, args.limit,
    )
    n_roi = profiles[0].shape[0]
    print(f"{len(profiles)} subjects, {len(set(sites))} sites, {n_roi} regions", flush=True)

    # The uncompressed ceiling: upper triangle of each subject's matrix.
    iu = np.triu_indices(n_roi, k=1)
    corr_vec = np.stack([p[iu] for p in profiles])

    # Per-subject compression is basis-independent of the fold, so do it once.
    per_subject = np.stack([
        PCA(n_components=args.components, random_state=0).fit_transform(p).ravel()
        for p in profiles
    ])

    folds = []
    for site in sorted(set(sites.tolist())):
        test = np.where(sites == site)[0]
        train = np.where(sites != site)[0]
        if len(test) < args.min_test_size or len(np.unique(labels[test])) < 2:
            continue
        y_tr, y_te = labels[train], labels[test]

        # Fold-wise: one basis fitted on training-site profiles only.
        basis = PCA(n_components=args.components, random_state=0)
        basis.fit(np.vstack([profiles[i] for i in train]))
        shared = np.stack([basis.transform(p).ravel() for p in profiles])

        row = {
            "site": site,
            "n": int(len(test)),
            "per_subject": auc_of(per_subject[train], y_tr, per_subject[test], y_te),
            "fold_wise": auc_of(shared[train], y_tr, shared[test], y_te),
            "ceiling": auc_of(corr_vec[train], y_tr, corr_vec[test], y_te),
        }
        folds.append(row)
        print(f"  {site:<14} n={row['n']:<4} per-subject {row['per_subject']:.3f}  "
              f"fold-wise {row['fold_wise']:.3f}  ceiling {row['ceiling']:.3f}", flush=True)

    summary = {
        k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))]
        for k in ("per_subject", "fold_wise", "ceiling")
    }
    print("\nmean AUC over folds")
    for k, (m, s) in summary.items():
        print(f"  {k:<12} {m:.3f} +- {s:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"config": vars(args) | {"data_root": str(args.data_root), "out": str(args.out)},
         "n_subjects": len(profiles), "summary": summary, "per_fold": folds},
        indent=2))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
