"""Subject descriptors for kernel methods.

PCA on flattened correlation vectors is the generic default, but it discards
the network structure a quantum feature map might exploit — principal
components of 19,900 pairwise correlations carry no explicit notion of hubs,
segregation, or modular organisation.

This module adds graph-theoretic descriptors computed from each subject's
connectivity matrix. They are low-dimensional by construction (a few dozen
numbers rather than 19,900), which suits a narrow qubit register, and each
carries an interpretable network meaning.
"""

from __future__ import annotations

import numpy as np

from qagta.graph.connectome import pearson_connectivity


def graph_descriptor(timeseries: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Graph-theoretic summary of one subject's connectome.

    Computes weighted and binarised network measures over the correlation
    matrix, then summarises each nodal measure by its distribution rather than
    keeping it per-node — so the descriptor stays comparable across subjects
    and short enough to encode on a handful of qubits.
    """
    matrix = pearson_connectivity(timeseries)
    np.fill_diagonal(matrix, 0.0)
    weights = np.abs(matrix)
    binary = (weights > threshold).astype(float)

    strength = weights.sum(axis=1)          # weighted degree
    degree = binary.sum(axis=1)             # binary degree

    # Local clustering on the binarised graph.
    triangles = np.diag(binary @ binary @ binary)
    possible = degree * (degree - 1)
    clustering = np.divide(triangles, possible, out=np.zeros_like(triangles),
                           where=possible > 0)

    # Spectral summary: leading eigenvalues of the weighted matrix describe
    # global integration without needing a full graph library.
    try:
        eigenvalues = np.linalg.eigvalsh(weights)[-5:]
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(5)

    def summarise(values: np.ndarray) -> list[float]:
        return [float(values.mean()), float(values.std()),
                float(np.percentile(values, 25)), float(np.percentile(values, 75)),
                float(values.max())]

    features = (
        summarise(strength)
        + summarise(degree)
        + summarise(clustering)
        + [float(weights.mean()), float(weights.std()),
           float((weights > threshold).mean()),          # density
           float(np.percentile(weights, 90))]
        + [float(v) for v in eigenvalues]
    )
    return np.nan_to_num(np.asarray(features, dtype=np.float32))


def mutual_information_connectivity(timeseries: np.ndarray, bins: int = 8) -> np.ndarray:
    """Pairwise mutual information (nats) from a ``(T, n_roi)`` BOLD matrix.

    Each regional series is rank-transformed onto ``bins`` equiprobable levels
    (a copula estimate), so the measure is invariant to monotone transforms and,
    unlike Pearson correlation, registers non-linear dependence. The Miller-Madow
    term removes the first-order bias, which otherwise grows as the scan
    shortens and would differ between sites.
    """
    t, regions = timeseries.shape
    ranks = np.argsort(np.argsort(timeseries, axis=0, kind="stable"), axis=0, kind="stable")
    codes = np.minimum(ranks * bins // t, bins - 1)
    onehot = np.zeros((t, regions * bins), dtype=np.float64)
    onehot[np.arange(t)[:, None], np.arange(regions)[None, :] * bins + codes] = 1.0
    joint = (onehot.T @ onehot / t).reshape(regions, bins, regions, bins)
    marginal = onehot.mean(axis=0).reshape(regions, bins)
    outer = marginal[:, :, None, None] * marginal[None, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log(joint / outer), 0.0)
    mi = terms.sum(axis=(1, 3)) - (bins - 1) ** 2 / (2 * t)
    np.fill_diagonal(mi, 0.0)
    return mi


def build_descriptors(timeseries_list, kind: str = "correlation") -> np.ndarray:
    """Stack per-subject descriptors of the requested kind.

    ``correlation`` returns the flattened upper triangle (high-dimensional,
    the classical default). ``mutual_information`` returns the same upper
    triangle of the rank-based mutual-information matrix, a non-linear
    dependence measure. ``graph`` returns the network summary above.
    ``both`` concatenates correlation and graph.
    """
    rows = []
    for series in timeseries_list:
        if kind == "correlation":
            matrix = pearson_connectivity(series)
            rows.append(matrix[np.triu_indices_from(matrix, k=1)])
        elif kind == "mutual_information":
            matrix = mutual_information_connectivity(series)
            rows.append(matrix[np.triu_indices_from(matrix, k=1)])
        elif kind == "graph":
            rows.append(graph_descriptor(series))
        elif kind == "both":
            matrix = pearson_connectivity(series)
            rows.append(np.concatenate([
                matrix[np.triu_indices_from(matrix, k=1)], graph_descriptor(series)
            ]))
        else:
            raise ValueError(f"unknown descriptor kind {kind!r}")
    return np.asarray(rows, dtype=np.float32)
