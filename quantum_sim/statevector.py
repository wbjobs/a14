from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend
from .gates import Gate


class StateVector:
    _data: Any
    _num_qubits: int
    _shape: tuple[int, ...]

    def __init__(self, num_qubits: int, data: Optional[Any] = None):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        if num_qubits > 30:
            raise ValueError(
                "Number of qubits exceeds maximum (30). 2^30 complex numbers require ~16GB memory."
            )

        self._num_qubits = num_qubits
        self._shape = tuple([2] * num_qubits)
        backend = get_backend()
        xp = backend.xp
        is_gpu = backend.is_gpu()

        if data is None:
            if is_gpu and num_qubits > 20:
                size = 2**num_qubits
                data_1d = xp.zeros(size, dtype=xp.complex128)
                data_1d[0] = 1.0
                self._data = xp.reshape(data_1d, self._shape, order='F')
                if hasattr(xp, 'cuda'):
                    xp.cuda.Stream.null.synchronize()
            else:
                self._data = xp.zeros(self._shape, dtype=xp.complex128)
                self._data[tuple([0] * num_qubits)] = 1.0
        else:
            self._data = backend.to_device(data)
            if self._data.shape != self._shape:
                if self._data.size == 2**num_qubits:
                    self._data = xp.reshape(self._data, self._shape, order='F')
                else:
                    self._data = self._data.reshape(self._shape)
            self._data = self._data.astype(xp.complex128)

        if is_gpu and num_qubits > 20 and hasattr(xp, 'cuda'):
            xp.cuda.Stream.null.synchronize()

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def data(self) -> Any:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to_vector(self) -> Any:
        backend = get_backend()
        xp = backend.xp
        return xp.reshape(self._data, -1, order='F')

    def to_numpy(self) -> np.ndarray:
        backend = get_backend()
        xp = backend.xp
        return backend.to_numpy(xp.reshape(self._data, -1, order='F'))

    def conjugate(self) -> "StateVector":
        result = StateVector(self._num_qubits)
        result._data = self._data.conj()
        return result

    def normalize(self) -> None:
        backend = get_backend()
        norm = backend.xp.sqrt(backend.xp.sum(backend.xp.abs(self._data) ** 2))
        if norm > 0:
            self._data /= norm

    def probability(self, qubits: Union[int, list[int]]) -> np.ndarray:
        if isinstance(qubits, int):
            qubits = [qubits]

        backend = get_backend()
        xp = backend.xp

        all_qubits = list(range(self._num_qubits))
        other_qubits = [q for q in all_qubits if q not in qubits]

        abs_sq = xp.abs(self._data) ** 2

        if other_qubits:
            for q in sorted(other_qubits, reverse=True):
                abs_sq = xp.sum(abs_sq, axis=q)

        remaining_axes = list(range(len(qubits)))
        desired_order = [qubits.index(q) for q in sorted(qubits)]
        if remaining_axes != desired_order:
            abs_sq = xp.transpose(abs_sq, desired_order)

        return backend.to_numpy(xp.reshape(abs_sq, -1, order='F'))

    def measure(self, qubits: Union[int, list[int]]) -> tuple[Any, np.ndarray]:
        if isinstance(qubits, int):
            qubits = [qubits]

        probs = self.probability(qubits)
        result = np.random.choice(len(probs), p=probs)

        backend = get_backend()
        xp = backend.xp

        bitstring = format(result, f"0{len(qubits)}b")

        mask = xp.ones_like(self._data, dtype=bool)
        for i, q in enumerate(qubits):
            bit = int(bitstring[i])
            sl = [slice(None)] * self._num_qubits
            sl[q] = 1 - bit
            mask[tuple(sl)] = False

        new_data = xp.where(mask, self._data, 0)
        norm = xp.sqrt(xp.sum(xp.abs(new_data) ** 2))
        if norm > 0:
            new_data /= norm

        new_state = StateVector(self._num_qubits, new_data)
        return new_state, result

    def apply_gate(self, gate: Gate, qubits: Union[int, list[int]], params: Optional[dict] = None) -> None:
        if isinstance(qubits, int):
            qubits = [qubits]

        if len(qubits) != gate.num_qubits:
            raise ValueError(f"Gate {gate.name} requires {gate.num_qubits} qubits, got {len(qubits)}")

        if any(q >= self._num_qubits for q in qubits):
            raise ValueError("Qubit index out of range")

        backend = get_backend()
        xp = backend.xp
        is_gpu = backend.is_gpu()

        gate_matrix = gate.matrix(params)
        n = gate.num_qubits

        if self._num_qubits > 20 and is_gpu:
            self._apply_gate_large(gate_matrix, qubits, n, xp, backend, is_gpu)
            return

        if n == 1:
            q = qubits[0]
            gate_matrix = gate_matrix.reshape(2, 2)
            self._data = xp.tensordot(gate_matrix, self._data, axes=([1], [q]))
            if q != 0:
                self._data = xp.moveaxis(self._data, 0, q)
        else:
            gate_shape = tuple([2] * (2 * n))
            gate_tensor = gate_matrix.reshape(gate_shape)

            gate_in_axes = list(range(n, 2 * n))
            gate_out_axes = list(range(n))

            contracted = xp.tensordot(gate_tensor, self._data, axes=(gate_in_axes, qubits))

            new_axes = list(range(self._num_qubits))
            remaining = [i for i in range(self._num_qubits) if i not in qubits]
            for i, q in enumerate(qubits):
                new_axes[q] = i
            for i, r in enumerate(remaining):
                new_axes[r] = n + i

            self._data = xp.transpose(contracted, new_axes)

    def _apply_gate_large(
        self,
        gate_matrix: Any,
        qubits: list[int],
        n: int,
        xp: Any,
        backend: Any,
        is_gpu: bool,
    ) -> None:
        gate_shape = tuple([2] * (2 * n))
        gate_tensor = gate_matrix.reshape(gate_shape)

        original_shape = self._data.shape
        qubits_sorted = sorted(qubits)
        qubits_map = {q: i for i, q in enumerate(qubits)}

        other_qubits = [i for i in range(self._num_qubits) if i not in qubits]

        if n == 1:
            q = qubits[0]
            gate_matrix_2d = gate_matrix.reshape(2, 2)
            new_data = xp.tensordot(gate_matrix_2d, self._data, axes=([1], [q]))
            if q != 0:
                new_data = xp.moveaxis(new_data, 0, q)
            self._data = new_data
            return

        perm_front = list(qubits) + other_qubits
        data_permuted = xp.transpose(self._data, perm_front)

        gate_in_axes = list(range(n, 2 * n))
        contracted = xp.tensordot(gate_tensor, data_permuted, axes=(gate_in_axes, list(range(n))))

        perm_back = [0] * self._num_qubits
        for i, q in enumerate(qubits):
            perm_back[q] = i
        for i, r in enumerate(other_qubits):
            perm_back[r] = n + i

        inv_perm = [0] * len(perm_back)
        for i, p in enumerate(perm_back):
            inv_perm[p] = i

        try:
            self._data = xp.transpose(contracted, inv_perm)
        except Exception:
            self._data = xp.ascontiguousarray(xp.transpose(contracted, inv_perm))

        if is_gpu and hasattr(xp, 'cuda'):
            xp.cuda.Stream.null.synchronize()

    def expectation_value(self, observable: "PauliOp") -> complex:
        backend = get_backend()
        xp = backend.xp

        result = 0.0 + 0.0j
        for term in observable.terms:
            coeff = term[0]
            pauli_string = term[1]

            temp_state = StateVector(self._num_qubits, self._data.copy())

            for qubit, pauli in enumerate(pauli_string):
                if pauli == "I":
                    continue
                elif pauli == "X":
                    from .gates import X

                    temp_state.apply_gate(X, qubit)
                elif pauli == "Y":
                    from .gates import Y

                    temp_state.apply_gate(Y, qubit)
                elif pauli == "Z":
                    from .gates import Z

                    temp_state.apply_gate(Z, qubit)

            bra = xp.reshape(self._data.conj(), -1, order='F')
            ket = xp.reshape(temp_state._data, -1, order='F')
            result += coeff * xp.sum(bra * ket)

        return float(xp.real(result)) if abs(xp.imag(result)) < 1e-10 else complex(result)

    def copy(self) -> "StateVector":
        new_sv = StateVector(self._num_qubits)
        new_sv._data = self._data.copy()
        return new_sv

    def __repr__(self) -> str:
        backend = get_backend()
        xp = backend.xp
        vec = backend.to_numpy(xp.reshape(self._data, -1, order='F'))
        return f"StateVector({self._num_qubits} qubits, {len(vec)} elements)"

    def __str__(self) -> str:
        backend = get_backend()
        xp = backend.xp
        vec = backend.to_numpy(xp.reshape(self._data, -1, order='F'))
        lines = []
        for i, amp in enumerate(vec):
            if abs(amp) > 1e-10:
                ket = format(i, f"0{self._num_qubits}b")
                lines.append(f"  {amp.real:+.6f}{amp.imag:+.6f}j |{ket}⟩")
        if not lines:
            return f"StateVector({self._num_qubits} qubits, zero vector)"
        return f"StateVector({self._num_qubits} qubits):\n" + "\n".join(lines[:20]) + ("\n  ..." if len(lines) > 20 else "")

    def __matmul__(self, other: "StateVector") -> "StateVector":
        if not isinstance(other, StateVector):
            raise TypeError("Can only tensor product with another StateVector")

        backend = get_backend()
        xp = backend.xp

        new_num_qubits = self._num_qubits + other._num_qubits
        new_data = xp.tensordot(self._data, other._data, axes=0)
        return StateVector(new_num_qubits, new_data)


def tensor_product(states: list[StateVector]) -> StateVector:
    if not states:
        raise ValueError("Need at least one state for tensor product")

    result = states[0]
    for state in states[1:]:
        result = result @ state
    return result
