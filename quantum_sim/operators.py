from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend


class PauliOp:
    def __init__(self, terms: Optional[list[tuple[complex, str]]] = None):
        if terms is None:
            self.terms: list[tuple[complex, str]] = []
        else:
            self.terms = []
            for coeff, pauli_string in terms:
                self._add_term(coeff, pauli_string)

    def _add_term(self, coeff: complex, pauli_string: str) -> None:
        for c in pauli_string:
            if c not in "IXYZ":
                raise ValueError(f"Invalid Pauli character: {c}")

        for i, (existing_coeff, existing_string) in enumerate(self.terms):
            if existing_string == pauli_string:
                self.terms[i] = (existing_coeff + coeff, pauli_string)
                return

        if abs(coeff) > 1e-15:
            self.terms.append((complex(coeff), pauli_string))

    @property
    def num_qubits(self) -> int:
        if not self.terms:
            return 0
        return len(self.terms[0][1])

    def __add__(self, other: Union["PauliOp", complex, float]) -> "PauliOp":
        result = PauliOp()
        for coeff, ps in self.terms:
            result._add_term(coeff, ps)

        if isinstance(other, PauliOp):
            for coeff, ps in other.terms:
                result._add_term(coeff, ps)
        else:
            n = self.num_qubits if self.num_qubits > 0 else 1
            result._add_term(complex(other), "I" * n)

        return result

    def __radd__(self, other: Union[complex, float]) -> "PauliOp":
        return self.__add__(other)

    def __sub__(self, other: Union["PauliOp", complex, float]) -> "PauliOp":
        return self.__add__(-other)

    def __rsub__(self, other: Union[complex, float]) -> "PauliOp":
        return (-self).__add__(other)

    def __neg__(self) -> "PauliOp":
        result = PauliOp()
        for coeff, ps in self.terms:
            result._add_term(-coeff, ps)
        return result

    def __mul__(self, other: Union["PauliOp", complex, float]) -> "PauliOp":
        if isinstance(other, (complex, float, int)):
            result = PauliOp()
            for coeff, ps in self.terms:
                result._add_term(coeff * complex(other), ps)
            return result

        pauli_mult = {
            ("I", "I"): (1, "I"),
            ("I", "X"): (1, "X"),
            ("I", "Y"): (1, "Y"),
            ("I", "Z"): (1, "Z"),
            ("X", "I"): (1, "X"),
            ("X", "X"): (1, "I"),
            ("X", "Y"): (1j, "Z"),
            ("X", "Z"): (-1j, "Y"),
            ("Y", "I"): (1, "Y"),
            ("Y", "X"): (-1j, "Z"),
            ("Y", "Y"): (1, "I"),
            ("Y", "Z"): (1j, "X"),
            ("Z", "I"): (1, "Z"),
            ("Z", "X"): (1j, "Y"),
            ("Z", "Y"): (-1j, "X"),
            ("Z", "Z"): (1, "I"),
        }

        result = PauliOp()
        for coeff1, ps1 in self.terms:
            for coeff2, ps2 in other.terms:
                if len(ps1) != len(ps2):
                    raise ValueError("Pauli strings must have the same length for multiplication")

                total_coeff = coeff1 * coeff2
                result_string = []
                for p1, p2 in zip(ps1, ps2):
                    phase, op = pauli_mult[(p1, p2)]
                    total_coeff *= phase
                    result_string.append(op)

                result._add_term(total_coeff, "".join(result_string))

        return result

    def __rmul__(self, other: Union[complex, float]) -> "PauliOp":
        return self.__mul__(other)

    def to_matrix(self) -> np.ndarray:
        backend = get_backend()
        xp = backend.xp

        if not self.terms:
            return xp.array([[0]], dtype=xp.complex128)

        n = self.num_qubits
        dim = 2**n
        matrix = xp.zeros((dim, dim), dtype=xp.complex128)

        pauli_matrices = {
            "I": xp.array([[1, 0], [0, 1]], dtype=xp.complex128),
            "X": xp.array([[0, 1], [1, 0]], dtype=xp.complex128),
            "Y": xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128),
            "Z": xp.array([[1, 0], [0, -1]], dtype=xp.complex128),
        }

        for coeff, pauli_string in self.terms:
            term_matrix = None
            for p in pauli_string:
                p_mat = pauli_matrices[p]
                if term_matrix is None:
                    term_matrix = p_mat
                else:
                    term_matrix = xp.kron(term_matrix, p_mat)
            matrix += coeff * term_matrix

        return matrix

    def simplify(self) -> "PauliOp":
        result = PauliOp()
        for coeff, ps in self.terms:
            if abs(coeff) > 1e-15:
                result._add_term(coeff, ps)
        return result

    def __repr__(self) -> str:
        if not self.terms:
            return "PauliOp(0)"

        terms_str = []
        for coeff, ps in self.terms:
            if abs(coeff.imag) < 1e-10:
                coeff_str = f"{coeff.real:+.6f}"
            else:
                coeff_str = f"{coeff:+.6f}"
            terms_str.append(f"{coeff_str} * {ps}")

        return "PauliOp(" + " ".join(terms_str) + ")"

    def __str__(self) -> str:
        return self.__repr__()


class Hamiltonian(PauliOp):
    def __init__(self, terms: Optional[list[tuple[float, str]]] = None):
        super().__init__()
        if terms is not None:
            for coeff, ps in terms:
                self._add_term(float(coeff), ps)

    def _add_term(self, coeff: complex, pauli_string: str) -> None:
        if abs(coeff.imag) > 1e-15:
            raise ValueError("Hamiltonian must be Hermitian: coefficients must be real")
        super()._add_term(complex(coeff), pauli_string)

    @property
    def is_hermitian(self) -> bool:
        return True

    def ground_state_energy(self, method: str = "exact") -> float:
        if method == "exact":
            matrix = self.to_matrix()
            backend = get_backend()
            eigenvalues = backend.linalg.eigvalsh(matrix)
            return float(backend.to_numpy(eigenvalues[0]))
        else:
            raise ValueError(f"Unknown method: {method}")

    def diagonalize(self) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.to_matrix()
        backend = get_backend()
        eigenvalues, eigenvectors = backend.linalg.eigh(matrix)
        return backend.to_numpy(eigenvalues), backend.to_numpy(eigenvectors)


def pauli_x(qubit: int, num_qubits: int) -> PauliOp:
    ps = "I" * qubit + "X" + "I" * (num_qubits - qubit - 1)
    return PauliOp([(1.0, ps)])


def pauli_y(qubit: int, num_qubits: int) -> PauliOp:
    ps = "I" * qubit + "Y" + "I" * (num_qubits - qubit - 1)
    return PauliOp([(1.0, ps)])


def pauli_z(qubit: int, num_qubits: int) -> PauliOp:
    ps = "I" * qubit + "Z" + "I" * (num_qubits - qubit - 1)
    return PauliOp([(1.0, ps)])


def identity(num_qubits: int) -> PauliOp:
    return PauliOp([(1.0, "I" * num_qubits)])


def from_openfermion(operator) -> PauliOp:
    result = PauliOp()
    for term, coeff in operator.terms.items():
        pauli_string = []
        for qubit, pauli_char in term:
            if pauli_char == "X":
                pauli_string.append(("X", qubit))
            elif pauli_char == "Y":
                pauli_string.append(("Y", qubit))
            elif pauli_char == "Z":
                pauli_string.append(("Z", qubit))

        n = max([q for _, q in pauli_string]) + 1 if pauli_string else 1
        ps = ["I"] * n
        for p_char, q in pauli_string:
            ps[q] = p_char

        result._add_term(complex(coeff), "".join(ps))

    return result
