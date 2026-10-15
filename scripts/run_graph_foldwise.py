"""ABIDE graph study with a fold-wise PCA basis (FINDING 26).

The original study fitted one PCA per subject inside
``compress_connectivity_profiles``, so component k meant something different
for every subject and the node features were not comparable across the cohort.
FINDING 26 measured the cost: an SVM on those features reaches AUC 0.541,
against 0.739 when a single basis is fitted on the training sites and applied
unchanged to the held-out site.

This script repeats the graph experiments with the basis fitted inside each
leave-site-out fold, so nothing from the held-out site touches it. Everything
else -- encoder, topology, k, model, epochs, seed -- is unchanged, so the
comparison against the original run is like for like.

Because the features differ per fold, the quantum encoding cannot be cached
across folds and is recomputed 20 times. That is the whole cost of the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qagta.data.abide import AbideDataset, SubjectRecord  # noqa: E402
from qagta.graph.connectome import pearson_connectivity  # noqa: E402
from qagta.pipelines.connectome_pipeline import encode_cohort  # noqa: E402
from qagta.quantum.fmri_encoder import build_encoder  # noqa: E402
from qagta.training.lso import LSOResult, train_fold  # noqa: E402


def load_raw(root: Path, pipeline: str, strategy: str, atlas: str,
             min_timepoints: int, limit: int | None):
    """Correlation profiles, labels, sites and file ids for usable subjects."""
    phenotypic = pd.read_csv(root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
    phenotypic = phenotypic[phenotypic["FILE_ID"] != "no_filename"]
    if limit:
        phenotypic = phenotypic.head(limit)
    series_dir = root / "ABIDE_pcp" / pipeline / strategy

    profiles, labels, sites, ids, lengths = [], [], [], [], []
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
        ids.append(str(row.FILE_ID))
        lengths.append(int(ts.shape[0]))
    return profiles, np.array(labels), np.array(sites), ids, lengths


def dataset_from_basis(basis, profiles, labels, sites, ids, lengths) -> AbideDataset:
    """Project every subject's profiles onto one shared basis."""
    return AbideDataset(subjects=[
        SubjectRecord(
            file_id=ids[i], site=sites[i], label=int(labels[i]),
            features=basis.transform(profiles[i]).astype(np.float32),
            n_timepoints=lengths[i],
            explained_variance=float(basis.explained_variance_ratio_.sum()),
        )
        for i in range(len(profiles))
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--n-qubits", type=int, default=16)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--k-neighbors", type=int, default=20)
    ap.add_argument("--topology", default="mixed")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--readout", default="mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/graph_foldwise.json"))
    args = ap.parse_args()

    print("loading connectivity profiles ...", flush=True)
    profiles, labels, sites, ids, lengths = load_raw(
        args.data_root, args.pipeline, args.strategy, args.atlas, 60, args.limit)
    print(f"{len(profiles)} subjects, {len(set(sites))} sites, "
          f"{profiles[0].shape[0]} regions", flush=True)

    encoder = build_encoder(args.backend, n_qubits=args.n_qubits)
    results = {name: LSOResult(name=name) for name in
               ("Proposed (quantum + GAT)", "Ablation (quantum + GCN)")}

    started = time.perf_counter()
    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue

        # The whole point: basis fitted on training sites only.
        basis = PCA(n_components=args.n_qubits, random_state=0)
        basis.fit(np.vstack([profiles[i] for i in train_idx]))
        dataset = dataset_from_basis(basis, profiles, labels, sites, ids, lengths)

        cohort = encode_cohort(dataset, encoder, k_neighbors=args.k_neighbors,
                               topology=args.topology, verbose=False)

        for name, model_type in (("Proposed (quantum + GAT)", "gat"),
                                 ("Ablation (quantum + GCN)", "gcn")):
            fold = train_fold(
                cohort, train_idx, test_idx, epochs=args.epochs, lr=args.lr,
                hidden_dim=args.hidden_dim, heads=args.heads, dropout=args.dropout,
                model_type=model_type, readout=args.readout, seed=args.seed,
            )
            results[name].folds.append(fold)
        p = results["Proposed (quantum + GAT)"].folds[-1]
        print(f"  {site:<14} n={p.n_test:<4} F1 {p.f1:.3f}  acc {p.accuracy:.3f}  "
              f"spec {p.specificity:.3f}  MAD {p.mad:.4f}", flush=True)

    print(f"\ntotal {time.perf_counter() - started:.0f}s\n")
    print("LEAVE-SITE-OUT, FOLD-WISE BASIS (mean +- 95% CI)")
    for r in results.values():
        print("  " + r.summary())

    payload = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_subjects": len(profiles),
        "results": {
            name: {
                "mean_ci": {m: r.mean_ci(m) for m in
                            ("f1", "accuracy", "specificity", "sensitivity")},
                "folds": [vars(f) for f in r.folds],
            } for name, r in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
