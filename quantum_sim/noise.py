from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend
from .statevector import StateVector


class NoiseChannel(ABC):
    name: str
    num_qubits: int

    def __init__(self, name: str, num_qubits: int = 1):
        self.name = name
        self.num_qubits = num_qubits

    @abstractmethod
    def get_kraus_operators(self) -> list[Any]:
        pass

    def apply(self, state: StateVector, qubits: Union[int, list[int]]) -> None:
        if isinstance(qubits, int):
            qubits = [qubits]

        if len(qubits) != self.num_qubits:
            raise ValueError(
                f"Noise channel {self.name} requires {self.num_qubits} qubits, got {len(qubits)}"
            )

        if any(q >= state.num_qubits for q in qubits):
            raise ValueError("Qubit index out of range")

        backend = get_backend()
        xp = backend.xp

        kraus_ops = self.get_kraus_operators()
        original_data = state._data.copy()

        new_data = xp.zeros_like(original_data, dtype=xp.complex128)

        for kraus in kraus_ops:
            kraus = xp.asarray(kraus, dtype=xp.complex128)
            temp_state = StateVector(state.num_qubits, original_data.copy())
            self._apply_kraus(temp_state, kraus, qubits, xp)
            new_data += temp_state._data

        norm = xp.sqrt(xp.sum(xp.abs(new_data) ** 2))
        if norm > 0:
            new_data /= norm

        state._data = new_data

    def _apply_kraus(
        self,
        state: StateVector,
        kraus: Any,
        qubits: list[int],
        xp: Any,
    ) -> None:
        n = self.num_qubits
        kraus_shape = tuple([2] * (2 * n))
        kraus_tensor = kraus.reshape(kraus_shape)

        kraus_in_axes = list(range(n, 2 * n))
        contracted = xp.tensordot(kraus_tensor, state._data, axes=(kraus_in_axes, qubits))

        new_axes = list(range(state.num_qubits))
        remaining = [i for i in range(state.num_qubits) if i not in qubits]
        for i, q in enumerate(qubits):
            new_axes[q] = i
        for i, r in enumerate(remaining):
            new_axes[r] = n + i

        state._data = xp.transpose(contracted, new_axes)

    def __repr__(self) -> str:
        return f"{self.name}()"


class BitFlip(NoiseChannel):
    def __init__(self, p: float):
        super().__init__("BitFlip", num_qubits=1)
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        self.p = p

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        sqrt_p = xp.sqrt(xp.asarray(self.p, dtype=xp.complex128))
        sqrt_1mp = xp.sqrt(xp.asarray(1 - self.p, dtype=xp.complex128))

        E0 = sqrt_1mp * xp.eye(2, dtype=xp.complex128)
        E1 = sqrt_p * xp.array([[0, 1], [1, 0]], dtype=xp.complex128)

        return [E0, E1]

    def __repr__(self) -> str:
        return f"BitFlip(p={self.p})"


class PhaseFlip(NoiseChannel):
    def __init__(self, p: float):
        super().__init__("PhaseFlip", num_qubits=1)
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        self.p = p

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        sqrt_p = xp.sqrt(xp.asarray(self.p, dtype=xp.complex128))
        sqrt_1mp = xp.sqrt(xp.asarray(1 - self.p, dtype=xp.complex128))

        E0 = sqrt_1mp * xp.eye(2, dtype=xp.complex128)
        E1 = sqrt_p * xp.array([[1, 0], [0, -1]], dtype=xp.complex128)

        return [E0, E1]

    def __repr__(self) -> str:
        return f"PhaseFlip(p={self.p})"


class PhaseDamping(NoiseChannel):
    def __init__(self, gamma: float):
        super().__init__("PhaseDamping", num_qubits=1)
        if not 0 <= gamma <= 1:
            raise ValueError("Damping rate gamma must be between 0 and 1")
        self.gamma = gamma

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        sqrt_gamma = xp.sqrt(xp.asarray(self.gamma, dtype=xp.complex128))
        sqrt_1mgamma = xp.sqrt(xp.asarray(1 - self.gamma, dtype=xp.complex128))

        E0 = sqrt_1mgamma * xp.eye(2, dtype=xp.complex128)
        E1 = sqrt_gamma * xp.array([[1, 0], [0, 0]], dtype=xp.complex128)
        E2 = sqrt_gamma * xp.array([[0, 0], [0, 1]], dtype=xp.complex128)

        return [E0, E1, E2]

    def __repr__(self) -> str:
        return f"PhaseDamping(gamma={self.gamma})"


class Depolarizing(NoiseChannel):
    def __init__(self, p: float):
        super().__init__("Depolarizing", num_qubits=1)
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        self.p = p

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        I = xp.eye(2, dtype=xp.complex128)
        X = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
        Y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
        Z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)

        sqrt_1mp = xp.sqrt(xp.asarray(1 - self.p, dtype=xp.complex128))
        sqrt_p3 = xp.sqrt(xp.asarray(self.p / 3, dtype=xp.complex128))

        E0 = sqrt_1mp * I
        E1 = sqrt_p3 * X
        E2 = sqrt_p3 * Y
        E3 = sqrt_p3 * Z

        return [E0, E1, E2, E3]

    def __repr__(self) -> str:
        return f"Depolarizing(p={self.p})"


class AmplitudeDamping(NoiseChannel):
    def __init__(self, gamma: float):
        super().__init__("AmplitudeDamping", num_qubits=1)
        if not 0 <= gamma <= 1:
            raise ValueError("Damping rate gamma must be between 0 and 1")
        self.gamma = gamma

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        sqrt_gamma = xp.sqrt(xp.asarray(self.gamma, dtype=xp.complex128))
        sqrt_1mgamma = xp.sqrt(xp.asarray(1 - self.gamma, dtype=xp.complex128))

        E0 = xp.array([[1, 0], [0, sqrt_1mgamma]], dtype=xp.complex128)
        E1 = xp.array([[0, sqrt_gamma], [0, 0]], dtype=xp.complex128)

        return [E0, E1]

    def __repr__(self) -> str:
        return f"AmplitudeDamping(gamma={self.gamma})"


class TwoQubitDepolarizing(NoiseChannel):
    def __init__(self, p: float):
        super().__init__("TwoQubitDepolarizing", num_qubits=2)
        if not 0 <= p <= 1:
            raise ValueError("Probability p must be between 0 and 1")
        self.p = p

    def get_kraus_operators(self) -> list[Any]:
        backend = get_backend()
        xp = backend.xp

        single_paulis = [
            xp.eye(2, dtype=xp.complex128),
            xp.array([[0, 1], [1, 0]], dtype=xp.complex128),
            xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128),
            xp.array([[1, 0], [0, -1]], dtype=xp.complex128),
        ]

        kraus_ops = []
        sqrt_1mp = xp.sqrt(xp.asarray(1 - self.p, dtype=xp.complex128))
        sqrt_p15 = xp.sqrt(xp.asarray(self.p / 15, dtype=xp.complex128))

        for i, pi in enumerate(single_paulis):
            for j, pj in enumerate(single_paulis):
                if i == 0 and j == 0:
                    op = sqrt_1mp * xp.kron(pi, pj)
                else:
                    op = sqrt_p15 * xp.kron(pi, pj)
                kraus_ops.append(op)

        return kraus_ops

    def __repr__(self) -> str:
        return f"TwoQubitDepolarizing(p={self.p})"


class NoiseModel:
    def __init__(self):
        self._gates_noise: dict[str, list[tuple[NoiseChannel, Optional[list[int]]]]] = {}
        self._qubit_noise: list[list[tuple[NoiseChannel, str]]] = []

    def add_gate_noise(
        self,
        gate_name: str,
        noise_channel: NoiseChannel,
        qubits: Optional[list[int]] = None,
    ) -> "NoiseModel":
        if gate_name not in self._gates_noise:
            self._gates_noise[gate_name] = []
        self._gates_noise[gate_name].append((noise_channel, qubits))
        return self

    def add_qubit_noise(
        self,
        qubit: int,
        noise_channel: NoiseChannel,
        after_gate: str = "all",
    ) -> "NoiseModel":
        while len(self._qubit_noise) <= qubit:
            self._qubit_noise.append([])
        self._qubit_noise[qubit].append((noise_channel, after_gate))
        return self

    def get_noise_for_gate(
        self,
        gate_name: str,
        qubits: list[int],
    ) -> list[tuple[NoiseChannel, list[int]]]:
        noise_list = []

        if gate_name in self._gates_noise:
            for noise_channel, target_qubits in self._gates_noise[gate_name]:
                if target_qubits is None or target_qubits == qubits:
                    noise_list.append((noise_channel, qubits))

        for q in qubits:
            if q < len(self._qubit_noise):
                for noise_channel, after_gate in self._qubit_noise[q]:
                    if after_gate == "all" or after_gate == gate_name:
                        noise_list.append((noise_channel, [q]))

        return noise_list

    def apply_noise_for_gate(
        self,
        state: StateVector,
        gate_name: str,
        qubits: list[int],
    ) -> None:
        for noise_channel, noise_qubits in self.get_noise_for_gate(gate_name, qubits):
            noise_channel.apply(state, noise_qubits)

    def __repr__(self) -> str:
        gate_noise_count = sum(len(v) for v in self._gates_noise.values())
        qubit_noise_count = sum(len(v) for v in self._qubit_noise)
        return f"NoiseModel({gate_noise_count} gate noises, {qubit_noise_count} qubit noises)"
