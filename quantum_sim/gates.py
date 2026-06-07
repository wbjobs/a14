from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend


class Gate(ABC):
    name: str
    num_qubits: int
    is_parameterized: bool = False

    def __init__(self, name: str, num_qubits: int):
        self.name = name
        self.num_qubits = num_qubits

    @abstractmethod
    def matrix(self, params: Optional[dict] = None) -> Any:
        pass

    def derivative_matrix(self, param_name: str, params: Optional[dict] = None) -> Any:
        raise NotImplementedError(f"Gate {self.name} is not parameterized")

    def __repr__(self) -> str:
        return f"{self.name}()"


class FixedGate(Gate):
    _matrix: Any

    def __init__(self, name: str, matrix: Any, num_qubits: int = 1):
        super().__init__(name, num_qubits)
        backend = get_backend()
        self._matrix = backend.to_device(matrix.astype(np.complex128))

    def matrix(self, params: Optional[dict] = None) -> Any:
        return self._matrix

    def __repr__(self) -> str:
        return f"{self.name}"


class ParameterizedGate(Gate):
    is_parameterized: bool = True
    param_name: str

    def __init__(self, name: str, param_name: str, num_qubits: int = 1):
        super().__init__(name, num_qubits)
        self.param_name = param_name

    @abstractmethod
    def matrix(self, params: Optional[dict] = None) -> Any:
        pass

    @abstractmethod
    def derivative_matrix(self, param_name: str, params: Optional[dict] = None) -> Any:
        pass


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

_CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_CZ = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_SWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)

_TOFFOLI = np.eye(8, dtype=np.complex128)
_TOFFOLI[6, 6] = 0
_TOFFOLI[6, 7] = 1
_TOFFOLI[7, 7] = 0
_TOFFOLI[7, 6] = 1
_TOFFOLI = _TOFFOLI.reshape(2, 2, 2, 2, 2, 2)


H = FixedGate("H", _H)
X = FixedGate("X", _X)
Y = FixedGate("Y", _Y)
Z = FixedGate("Z", _Z)
I = FixedGate("I", _I)
S = FixedGate("S", _S)
T = FixedGate("T", _T)
CNOT = FixedGate("CNOT", _CNOT, num_qubits=2)
CZ = FixedGate("CZ", _CZ, num_qubits=2)
SWAP = FixedGate("SWAP", _SWAP, num_qubits=2)
TOFFOLI = FixedGate("TOFFOLI", _TOFFOLI, num_qubits=3)


class RX(ParameterizedGate):
    def __init__(self):
        super().__init__("RX", "theta")

    def matrix(self, params: Optional[dict] = None) -> Any:
        if params is None or "theta" not in params:
            raise ValueError("RX gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        c = xp.cos(theta / 2)
        s = xp.sin(theta / 2)
        mat = xp.array([[c, -1j * s], [-1j * s, c]], dtype=xp.complex128)
        return mat

    def derivative_matrix(self, param_name: str, params: Optional[dict] = None) -> Any:
        if param_name != "theta":
            raise ValueError(f"Unknown parameter: {param_name}")
        if params is None or "theta" not in params:
            raise ValueError("RX gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        c = xp.cos(theta / 2)
        s = xp.sin(theta / 2)
        dmat = -0.5 * xp.array([[s, 1j * c], [1j * c, s]], dtype=xp.complex128)
        return dmat


class RY(ParameterizedGate):
    def __init__(self):
        super().__init__("RY", "theta")

    def matrix(self, params: Optional[dict] = None) -> Any:
        if params is None or "theta" not in params:
            raise ValueError("RY gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        c = xp.cos(theta / 2)
        s = xp.sin(theta / 2)
        mat = xp.array([[c, -s], [s, c]], dtype=xp.complex128)
        return mat

    def derivative_matrix(self, param_name: str, params: Optional[dict] = None) -> Any:
        if param_name != "theta":
            raise ValueError(f"Unknown parameter: {param_name}")
        if params is None or "theta" not in params:
            raise ValueError("RY gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        c = xp.cos(theta / 2)
        s = xp.sin(theta / 2)
        dmat = -0.5 * xp.array([[s, c], [-c, s]], dtype=xp.complex128)
        return dmat


class RZ(ParameterizedGate):
    def __init__(self):
        super().__init__("RZ", "theta")

    def matrix(self, params: Optional[dict] = None) -> Any:
        if params is None or "theta" not in params:
            raise ValueError("RZ gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        phase = xp.exp(-1j * theta / 2)
        mat = xp.array([[phase, 0], [0, xp.conj(phase)]], dtype=xp.complex128)
        return mat

    def derivative_matrix(self, param_name: str, params: Optional[dict] = None) -> Any:
        if param_name != "theta":
            raise ValueError(f"Unknown parameter: {param_name}")
        if params is None or "theta" not in params:
            raise ValueError("RZ gate requires 'theta' parameter")
        theta = params["theta"]
        backend = get_backend()
        xp = backend.xp
        phase = xp.exp(-1j * theta / 2)
        dmat = -0.5j * xp.array([[phase, 0], [0, -xp.conj(phase)]], dtype=xp.complex128)
        return dmat


RX = RX()
RY = RY()
RZ = RZ()


def get_gate(gate_name: str) -> Gate:
    gate_map = {
        "H": H,
        "X": X,
        "Y": Y,
        "Z": Z,
        "I": I,
        "S": S,
        "T": T,
        "CNOT": CNOT,
        "CZ": CZ,
        "SWAP": SWAP,
        "TOFFOLI": TOFFOLI,
        "RX": RX,
        "RY": RY,
        "RZ": RZ,
    }
    if gate_name not in gate_map:
        raise ValueError(f"Unknown gate: {gate_name}")
    return gate_map[gate_name]
