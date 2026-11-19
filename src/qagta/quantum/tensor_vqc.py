"""Tensor-train-guided variational quantum classifiers and their classical twins.

TTN-VQC
    Qi, Yang, Chen and Hsieh, npj Quantum Inf. 9:4 (2023). A tensor-train (TT)
    layer compresses the full feature vector to one angle per qubit, which is
    encoded by RY; ``depth`` blocks of [CNOT ring, RX, RY, RZ] follow, then a
    Pauli-Z read-out per qubit and a linear head.

TensorHyper-VQC
    Qi, Yang, Chen and Hsieh, npj Quantum Inf. 12:70 (2026). A TT network acts
    as a hypernetwork: it maps the input to every rotation angle of the circuit,
    added to trainable global angles, so the circuit is a forward evaluator and
    all trainable parameters sit in the classical TT cores. The encoder takes the
    first ``n_qubits`` features as RY angles; ``depth`` blocks of [RX, RY, RZ,
    CNOT ring] follow, then the Pauli-Z read-out and a linear head.

Both are re-implemented in the batched statevector simulator from the published
descriptions and the authors' reference code (github.com/jqi41/TTN-VQC and
github.com/jqi41/TensorHyper), for the same reason as the QCNN: the reference
code evaluates one sample per circuit execution, which does not scale to
thousands of subjects across dozens of folds. ``tests/test_tensor_vqc.py``
checks both circuits gate for gate against PennyLane.

Classical twins keep the TT front-end and the linear head and replace only the
circuit by a classical block with exactly the same number of trainable
parameters (``TTMLP`` for TTN-VQC, ``TTHyperMLP`` for TensorHyper-VQC), so any
difference between the arms is attributable to the circuit itself.
"""

from __future__ import annotations

import math
import string

import torch
from torch import nn

from qagta.quantum.simulator import _apply_rx, _apply_ry, _apply_rz, _cnot_permutation


class TensorTrainLinear(nn.Module):
    """Linear map whose weight matrix is stored as a tensor train.

    The input is reshaped to ``in_modes`` and the output to ``out_modes``; core
    ``k`` has shape ``(ranks[k], in_modes[k], out_modes[k], ranks[k + 1])``.
    """

    def __init__(self, in_modes, out_modes, ranks) -> None:
        super().__init__()
        d = len(in_modes)
        if len(out_modes) != d or len(ranks) != d + 1 or ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("need len(out_modes) == d, len(ranks) == d + 1, end ranks 1")
        self.in_modes, self.out_modes, self.ranks = list(in_modes), list(out_modes), list(ranks)
        self.cores = nn.ParameterList()
        for k in range(d):
            core = torch.empty(ranks[k], in_modes[k], out_modes[k], ranks[k + 1])
            nn.init.xavier_uniform_(core)  # the initialisation of the reference code
            self.cores.append(nn.Parameter(core))
        self.bias = nn.Parameter(torch.zeros(math.prod(out_modes)))

        letters = iter(c for c in string.ascii_letters if c != "z")
        ins = [next(letters) for _ in range(d)]
        outs = [next(letters) for _ in range(d)]
        rs = [next(letters) for _ in range(d + 1)]
        cores = [f"{rs[k]}{ins[k]}{outs[k]}{rs[k + 1]}" for k in range(d)]
        self._equation = f"z{''.join(ins)}," + ",".join(cores) + f"->z{''.join(outs)}"

    @property
    def in_features(self) -> int:
        return math.prod(self.in_modes)

    @property
    def out_features(self) -> int:
        return math.prod(self.out_modes)

    def dense_weight(self) -> torch.Tensor:
        """The full ``(in_features, out_features)`` matrix, for tests only."""
        eye = torch.eye(self.in_features)
        return self.forward(eye) - self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        out = torch.einsum(self._equation, x.reshape(batch, *self.in_modes), *self.cores)
        return out.reshape(batch, -1) + self.bias


class _RingCircuit(nn.Module):
    """Shared batched-simulator machinery: CNOT ring and Pauli-Z read-out."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        # (0,1), (1,2), ..., (n-1,0): the ring of both reference implementations.
        self.ring = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
        for control, target in self.ring:
            self.register_buffer(
                f"_cx_{control}_{target}",
                _cnot_permutation(n_qubits, control, target, torch.device("cpu")),
                persistent=False,
            )
        index = torch.arange(self.dim)
        signs = torch.stack([1.0 - 2.0 * ((index >> (n_qubits - q - 1)) & 1).float()
                             for q in range(n_qubits)])
        self.register_buffer("_z_signs", signs, persistent=False)

    def _zero_state(self, batch: int, device) -> torch.Tensor:
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0
        return state

    def _entangle(self, state: torch.Tensor) -> torch.Tensor:
        for control, target in self.ring:
            state = state.index_select(1, getattr(self, f"_cx_{control}_{target}"))
        return state

    def _rotations(self, state, rx, ry, rz) -> torch.Tensor:
        """RX, RY then RZ on every qubit; each argument is ``(batch, n)`` or ``(n,)``."""
        for q in range(self.n_qubits):
            state = _apply_rx(state, self.n_qubits, q, rx[..., q])
            state = _apply_ry(state, self.n_qubits, q, ry[..., q])
            state = _apply_rz(state, self.n_qubits, q, rz[..., q])
        return state

    def _expectations(self, state: torch.Tensor) -> torch.Tensor:
        probs = state.real**2 + state.imag**2
        return probs @ self._z_signs.T


class TTNVQC(_RingCircuit):
    """TTN-VQC: TT compression to one angle per qubit, then a ring VQC."""

    def __init__(self, in_modes, out_modes, ranks, depth: int = 6, seed: int = 0) -> None:
        torch.manual_seed(seed)
        super().__init__(math.prod(out_modes))
        self.depth = depth
        self.tt = TensorTrainLinear(in_modes, out_modes, ranks)
        self.q_params = nn.Parameter(0.01 * torch.randn(3 * depth * self.n_qubits))
        self.head = nn.Linear(self.n_qubits, 2)

    def circuit(self, angles: torch.Tensor) -> torch.Tensor:
        """Pauli-Z expectations for RY-encoded ``angles`` of shape ``(batch, n)``."""
        state = self._zero_state(angles.shape[0], angles.device)
        for q in range(self.n_qubits):
            state = _apply_ry(state, self.n_qubits, q, angles[:, q])
        w = self.q_params.reshape(3, self.depth, self.n_qubits)
        for k in range(self.depth):
            state = self._entangle(state)
            state = self._rotations(state, w[0, k], w[1, k], w[2, k])
        return self._expectations(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.circuit(self.tt(x) * (math.pi / 2)))


class TTMLP(nn.Module):
    """Classical twin of TTN-VQC: the circuit becomes tanh layers of equal size."""

    def __init__(self, in_modes, out_modes, ranks, depth: int = 6, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        n = math.prod(out_modes)
        budget = 3 * depth * n  # trainable parameters of the replaced circuit
        if budget % (n * n + n):
            raise ValueError(f"cannot match {budget} circuit parameters with {n}x{n} layers")
        self.tt = TensorTrainLinear(in_modes, out_modes, ranks)
        layers = []
        for _ in range(budget // (n * n + n)):
            layers += [nn.Linear(n, n), nn.Tanh()]
        self.block = nn.Sequential(*layers)
        self.head = nn.Linear(n, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(self.tt(x) * (math.pi / 2)))


class TensorHyperVQC(_RingCircuit):
    """TensorHyper-VQC: a TT hypernetwork writes every angle of a ring VQC.

    The input is ``[encoded | features]``: the first ``n_qubits`` columns are the
    RY encoding angles and the rest feed the TT hypernetwork.
    """

    def __init__(self, n_qubits: int, in_modes, out_modes, ranks,
                 depth: int = 6, seed: int = 0) -> None:
        torch.manual_seed(seed)
        super().__init__(n_qubits)
        self.depth = depth
        if math.prod(out_modes) != 3 * depth * n_qubits:
            raise ValueError("TT output must supply 3 * depth * n_qubits angles")
        self.tt = TensorTrainLinear(in_modes, out_modes, ranks)
        self.angles = nn.Parameter(0.1 * torch.randn(depth, n_qubits, 3))
        self.head = nn.Linear(n_qubits, 2)

    def circuit(self, encoded: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Expectations for encoding ``(batch, n)`` and angles ``(batch, depth, n, 3)``."""
        state = self._zero_state(encoded.shape[0], encoded.device)
        for q in range(self.n_qubits):
            state = _apply_ry(state, self.n_qubits, q, encoded[:, q])
        for k in range(self.depth):
            a = angles[:, k]
            state = self._rotations(state, a[..., 0], a[..., 1], a[..., 2])
            state = self._entangle(state)
        return self._expectations(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n_qubits
        encoded, features = x[:, :n], x[:, n:]
        generated = self.tt(features).reshape(-1, self.depth, n, 3)
        return self.head(self.circuit(encoded, self.angles.unsqueeze(0) + generated))


class TTHyperMLP(nn.Module):
    """Classical twin of TensorHyper-VQC.

    The same TT hypernetwork writes, per sample, the weights and biases of two
    ``n x n`` tanh layers applied to the same encoded features: exactly the
    ``3 * depth * n`` numbers the circuit would receive when ``depth = 6`` and
    ``n = 8``.
    """

    def __init__(self, n_features: int, in_modes, out_modes, ranks,
                 depth: int = 6, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        n = n_features
        if math.prod(out_modes) != 3 * depth * n or 3 * depth * n != 2 * (n * n + n):
            raise ValueError("the generated block must hold exactly 2 * (n^2 + n) numbers")
        self.n = n
        self.tt = TensorTrainLinear(in_modes, out_modes, ranks)
        self.globals = nn.Parameter(0.1 * torch.randn(3 * depth * n))
        self.head = nn.Linear(n, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        encoded, features = x[:, :n], x[:, n:]
        p = self.globals.unsqueeze(0) + self.tt(features)
        w1, b1, w2, b2 = torch.split(p, [n * n, n, n * n, n], dim=1)
        h = torch.tanh(torch.bmm(w1.reshape(-1, n, n), encoded.unsqueeze(2)).squeeze(2) + b1)
        h = torch.tanh(torch.bmm(w2.reshape(-1, n, n), h.unsqueeze(2)).squeeze(2) + b2)
        return self.head(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
