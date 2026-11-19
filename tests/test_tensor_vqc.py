"""TTN-VQC and TensorHyper-VQC against PennyLane, gate for gate."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from qagta.quantum.tensor_vqc import (
    TTMLP,
    TTNVQC,
    TensorHyperVQC,
    TensorTrainLinear,
    TTHyperMLP,
    count_parameters,
)

qml = pytest.importorskip("pennylane")


def _ring(n):
    for q in range(n):
        qml.CNOT(wires=[q, (q + 1) % n])


def test_tensor_train_is_a_linear_map():
    torch.manual_seed(0)
    tt = TensorTrainLinear([4, 5, 6], [2, 2, 2], [1, 3, 3, 1])
    x = torch.randn(7, 120)
    dense = tt.dense_weight()
    assert dense.shape == (120, 8)
    assert torch.allclose(tt(x), x @ dense + tt.bias, atol=1e-5)


def test_ttn_vqc_circuit_matches_pennylane():
    n, depth = 4, 3
    model = TTNVQC([3, 4], [2, 2], [1, 2, 1], depth=depth, seed=1)
    with torch.no_grad():
        model.q_params.normal_(0, 1.0)
    angles = torch.rand(5, n) * math.pi
    ours = model.circuit(angles).detach().numpy()

    w = model.q_params.detach().numpy().reshape(3, depth, n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def reference(a):
        for q in range(n):
            qml.RY(a[q], wires=q)
        for k in range(depth):
            _ring(n)
            for q in range(n):
                qml.RX(w[0, k, q], wires=q)
            for q in range(n):
                qml.RY(w[1, k, q], wires=q)
            for q in range(n):
                qml.RZ(w[2, k, q], wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    expected = np.array([reference(a) for a in angles.numpy()], dtype=float)
    assert np.allclose(ours, expected, atol=1e-5)


def test_tensorhyper_circuit_matches_pennylane():
    n, depth = 4, 2
    model = TensorHyperVQC(n, [3, 4], [4, 6], [1, 2, 1], depth=depth, seed=2)
    encoded = torch.rand(3, n) * math.pi / 2
    angles = torch.randn(3, depth, n, 3)
    ours = model.circuit(encoded, angles).detach().numpy()

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def reference(e, a):
        for q in range(n):
            qml.RY(e[q], wires=q)
        for k in range(depth):
            for q in range(n):
                qml.RX(a[k, q, 0], wires=q)
                qml.RY(a[k, q, 1], wires=q)
                qml.RZ(a[k, q, 2], wires=q)
            _ring(n)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    pairs = zip(encoded.numpy(), angles.numpy(), strict=True)
    expected = np.array([reference(e, a) for e, a in pairs], dtype=float)
    assert np.allclose(ours, expected, atol=1e-5)


def test_tensorhyper_uses_generated_angles_per_sample():
    model = TensorHyperVQC(8, [5, 8, 5, 10], [4, 4, 3, 3], [1, 2, 2, 2, 1], seed=0)
    x = torch.randn(4, 8 + 2000)
    out = model(x)
    assert out.shape == (4, 2)
    out.sum().backward()
    assert all(core.grad is not None and core.grad.abs().sum() > 0 for core in model.tt.cores)


def test_classical_twins_match_parameter_counts():
    ttn = TTNVQC([10, 10, 20], [2, 2, 2], [1, 3, 3, 1], depth=6)
    mlp = TTMLP([10, 10, 20], [2, 2, 2], [1, 3, 3, 1], depth=6)
    assert count_parameters(ttn) == count_parameters(mlp)

    hyper = TensorHyperVQC(8, [5, 8, 5, 10], [4, 4, 3, 3], [1, 2, 2, 2, 1], depth=6)
    hmlp = TTHyperMLP(8, [5, 8, 5, 10], [4, 4, 3, 3], [1, 2, 2, 2, 1], depth=6)
    assert count_parameters(hyper) == count_parameters(hmlp)
