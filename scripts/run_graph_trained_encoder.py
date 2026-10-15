"""ABIDE graph study with the quantum encoder trained jointly (FINDING 27).

The reported graph runs (``run_abide_study.py``, ``run_graph_foldwise.py``)
evaluate the encoder once, without gradients, at its initial parameters, and
select the top-k support from that single encoding. Only the GAT (or GCN) and
the edge kernel are optimised, so every encoding gain stays at 1.

This script trains the architecture as the manuscript describes it:

* the encoder's gains and variational angles are in the optimiser and receive
  gradients through the Pauli-Z latents, which are both the node features and
  the input to the edge kernel;
* the top-k support is re-selected from the current encoder at the start of
  every epoch, since top-k itself carries no gradient;
* the held-out site's support is selected with the final encoder.

A frozen control runs through the same code with the same folds, seeds and
hyperparameters (encoder fixed at initialisation, support selected once), so
the comparison isolates encoder training. The frozen arm is the configuration
of ``results/graph_foldwise_q8.json`` and should reproduce it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch, Data

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from run_graph_foldwise import load_raw  # noqa: E402

from qagta.graph.connectome import AdaptiveConnectomeEdges, knn_sparsify  # noqa: E402
from qagta.models.connectome_gat import ConnectomeGAT, ConnectomeGCN  # noqa: E402
from qagta.pipelines.connectome_pipeline import _initial_adjacency  # noqa: E402
from qagta.quantum.fmri_encoder import build_encoder  # noqa: E402
from qagta.training.lso import FoldResult, LSOResult, _metrics  # noqa: E402

MODELS = {"gat": "Proposed (quantum + GAT)", "gcn": "Ablation (quantum + GCN)"}
ARMS = ("frozen", "trained")


@torch.no_grad()
def select_support(encoder, X, indices, k, topology) -> dict[int, torch.Tensor]:
    """Top-k support of each subject under the encoder's current parameters."""
    support = {}
    for i in indices:
        z, states = encoder(X[int(i)], return_state=True)
        edge_index, _ = knn_sparsify(_initial_adjacency(z, states, topology), k=k)
        support[int(i)] = edge_index
    return support


def replaced_fraction(before: dict, after: dict, indices, n_nodes: int) -> float:
    """Mean fraction of a subject's edges that differ between two supports."""
    fractions = []
    for i in indices:
        a = before[int(i)][0] * n_nodes + before[int(i)][1]
        b = after[int(i)][0] * n_nodes + after[int(i)][1]
        kept = np.intersect1d(a.numpy(), b.numpy()).size
        fractions.append(1.0 - kept / a.numel())
    return float(np.mean(fractions))


def run_arm(X, labels, train_idx, test_idx, encoder0, model_type, trained, args):
    """Train one model on one fold; the encoder is optimised only if ``trained``."""
    torch.manual_seed(args.seed)
    n_nodes, d = X.shape[1], X.shape[2]
    encoder = copy.deepcopy(encoder0)
    for p in encoder.parameters():
        p.requires_grad_(trained)

    if model_type == "gat":
        model = ConnectomeGAT(latent_dim=d, hidden_dim=args.hidden_dim, heads=args.heads,
                              dropout=args.dropout, readout=args.readout, n_nodes=n_nodes)
    else:
        model = ConnectomeGCN(latent_dim=d, hidden_dim=args.hidden_dim, dropout=args.dropout)
    edge_learner = AdaptiveConnectomeEdges(latent_dim=d)
    params = list(model.parameters()) + list(edge_learner.parameters())
    if trained:
        params += list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)

    counts = torch.bincount(labels[train_idx], minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp_min(1))).to(torch.float32)

    all_idx = np.concatenate([train_idx, test_idx])
    initial = select_support(encoder, X, all_idx, args.k_neighbors, args.topology)
    support = dict(initial)
    if not trained:
        with torch.no_grad():
            cached = {int(i): encoder(X[int(i)]) for i in all_idx}

    def graphs(chunk):
        return Batch.from_data_list([
            Data(x=torch.zeros(n_nodes, 1), edge_index=support[int(i)],
                 y=labels[int(i)].reshape(1)) for i in chunk])

    def latents(chunk):
        if trained:
            return encoder(X[chunk].reshape(-1, d))
        return torch.cat([cached[int(i)] for i in chunk])

    rng = np.random.default_rng(args.seed)
    model.train()
    edge_learner.train()
    for epoch in range(args.epochs):
        if trained and epoch > 0:
            support.update(select_support(encoder, X, train_idx, args.k_neighbors, args.topology))
        order = rng.permutation(train_idx)
        for start in range(0, len(order), args.batch_size):
            chunk = order[start : start + args.batch_size]
            if len(chunk) < 2:
                continue
            batch = graphs(chunk)
            x = latents(chunk)
            optimizer.zero_grad()
            weights = edge_learner(x, batch.edge_index)
            logits = model(x, batch.edge_index, weights, batch.batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_weight)
            loss.backward()
            optimizer.step()

    model.eval()
    edge_learner.eval()
    if trained:
        # train supports are refreshed once more so they match the final encoder
        support.update(select_support(encoder, X, all_idx, args.k_neighbors, args.topology))

    def predict(indices):
        probs = []
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                chunk = indices[start : start + args.batch_size]
                batch = graphs(chunk)
                x = encoder(X[chunk].reshape(-1, d))
                logits = model(x, batch.edge_index, edge_learner(x, batch.edge_index), batch.batch)
                probs.append(logits.softmax(dim=1)[:, 1].numpy())
        return np.concatenate(probs)

    prob_test = predict(test_idx)
    y_true = labels[test_idx].numpy()
    y_pred = (prob_test >= 0.5).astype(int)
    f1, accuracy, specificity, sensitivity = _metrics(y_true, y_pred)
    auc = float(roc_auc_score(y_true, prob_test)) if len(np.unique(y_true)) == 2 else float("nan")
    train_acc = float(((predict(train_idx) >= 0.5).astype(int) == labels[train_idx].numpy()).mean())

    diag = {
        "auc": auc,
        "train_accuracy": train_acc,
        "predicted_positive_rate": float(y_pred.mean()),
        "single_class": bool(y_pred.min() == y_pred.max()),
        "gains": [round(float(g), 4) for g in encoder.encode_scale.detach()],
        "mean_abs_gain_change": float(
            (encoder.encode_scale - encoder0.encode_scale).detach().abs().mean()),
        "mean_abs_angle_change": float(torch.cat([
            (encoder.phi_z - encoder0.phi_z).detach().abs(),
            (encoder.phi_x - encoder0.phi_x).detach().abs()]).mean()),
        "support_replaced_train": replaced_fraction(initial, support, train_idx, n_nodes),
        "support_replaced_test": replaced_fraction(initial, support, test_idx, n_nodes),
    }
    fold = FoldResult(site="", n_test=len(test_idx), f1=f1, accuracy=accuracy,
                      specificity=specificity, sensitivity=sensitivity)
    return fold, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--k-neighbors", type=int, default=20)
    ap.add_argument("--topology", default="mixed")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--readout", default="mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--only-sites", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["gat", "gcn"])
    ap.add_argument("--shard", default=None, help="i/n: run every n-th eligible fold starting at i")
    ap.add_argument("--out", type=Path, default=Path("results/graph_trained_encoder_q8.json"))
    args = ap.parse_args()

    print("loading connectivity profiles ...", flush=True)
    profiles, labels_np, sites, _, _ = load_raw(
        args.data_root, args.pipeline, args.strategy, args.atlas, 60, None)
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    print(f"{len(profiles)} subjects, {len(set(sites))} sites", flush=True)

    results = {(m, a): LSOResult(name=f"{MODELS[m]} [{a}]") for m in args.models for a in ARMS}
    diags = {key: [] for key in results}
    started = time.perf_counter()

    shard, n_shards = (int(v) for v in args.shard.split("/")) if args.shard else (0, 1)
    eligible = -1
    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < args.min_test_size or len(np.unique(labels_np[test_idx])) < 2:
            continue
        eligible += 1
        if args.only_sites and site not in args.only_sites:
            continue
        if eligible % n_shards != shard:
            continue

        fold_start = time.perf_counter()
        basis = PCA(n_components=args.n_qubits, random_state=0)
        basis.fit(np.vstack([profiles[i] for i in train_idx]))
        X = torch.as_tensor(np.stack([basis.transform(p) for p in profiles]), dtype=torch.float32)
        torch.manual_seed(args.seed)
        encoder0 = build_encoder("torch", n_qubits=args.n_qubits)

        for model_type in args.models:
            for arm in ARMS:
                t0 = time.perf_counter()
                fold, diag = run_arm(X, labels, train_idx, test_idx, encoder0,
                                     model_type, arm == "trained", args)
                fold.site = site
                diag["seconds"] = round(time.perf_counter() - t0, 1)
                results[(model_type, arm)].folds.append(fold)
                diags[(model_type, arm)].append({"site": site} | diag)
                print(f"  {site:<10} {model_type} {arm:<7} acc {fold.accuracy:.3f} "
                      f"F1 {fold.f1:.3f} spec {fold.specificity:.3f} AUC {diag['auc']:.3f} "
                      f"train {diag['train_accuracy']:.3f} single-class {diag['single_class']} "
                      f"|dgain| {diag['mean_abs_gain_change']:.3f} "
                      f"support replaced {diag['support_replaced_train']:.2f} "
                      f"({diag['seconds']:.0f}s)", flush=True)
        print(f"  fold {site} done in {time.perf_counter() - fold_start:.0f}s", flush=True)
        save(args, len(profiles), results, diags, time.perf_counter() - started)

    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    for key, r in results.items():
        print("  " + r.summary()
              + f"  single-class folds {sum(d['single_class'] for d in diags[key])}"
              f"/{len(r.folds)}")
    for m in args.models:
        f, t = results[(m, "frozen")], results[(m, "trained")]
        if len(f.folds) >= 5:
            for metric in ("accuracy", "f1"):
                a, b = f._values(metric), t._values(metric)
                diff = b - a
                p = wilcoxon(b, a).pvalue if np.any(diff != 0) else 1.0
                print(f"  {m} trained - frozen {metric}: "
                      f"median {np.median(diff):+.3f}, p = {p:.3f}")
    print(f"saved: {args.out}")


def save(args, n_subjects, results, diags, seconds) -> None:
    payload = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_subjects": n_subjects,
        "seconds": round(seconds),
        "results": {
            f"{MODELS[m]} [{a}]": {
                "mean_ci": {metric: r.mean_ci(metric) for metric in
                            ("f1", "accuracy", "specificity", "sensitivity")},
                "folds": [vars(f) | d for f, d in zip(r.folds, diags[(m, a)], strict=False)],
            } for (m, a), r in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
