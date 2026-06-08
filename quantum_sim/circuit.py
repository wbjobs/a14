from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend
from .gates import Gate, H, X, Y, Z, I, S, T, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI
from .statevector import StateVector
from .autograd import Parameter, parameter_shift_gradient, numerical_gradient
from .operators import PauliOp, Hamiltonian
from .noise import NoiseModel, NoiseChannel


class CircuitInstruction:
    def __init__(
        self,
        gate: Gate,
        qubits: Union[int, list[int]],
        params: Optional[dict] = None,
        param_names: Optional[list[str]] = None,
    ):
        self.gate = gate
        self.qubits = [qubits] if isinstance(qubits, int) else qubits
        self.params = params or {}
        self.param_names = param_names or []

    def __repr__(self) -> str:
        param_str = f", params={self.params}" if self.params else ""
        return f"CircuitInstruction(gate={self.gate}, qubits={self.qubits}{param_str})"


class QuantumCircuit:
    def __init__(self, num_qubits: int):
        if num_qubits < 1 or num_qubits > 30:
            raise ValueError("Number of qubits must be between 1 and 30")
        self._num_qubits = num_qubits
        self._instructions: list[CircuitInstruction] = []
        self._parameters: dict[str, Parameter] = {}
        self._parameter_counter: int = 0

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def instructions(self) -> list[CircuitInstruction]:
        return self._instructions

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self._parameters

    def _add_parameter(self, value: Union[float, Parameter], name: Optional[str] = None) -> tuple[str, Parameter]:
        if isinstance(value, Parameter):
            param = value
            if param.name and name is None:
                name = param.name
        else:
            if name is None:
                name = f"theta_{self._parameter_counter}"
                self._parameter_counter += 1
            param = Parameter(float(value), name=name)

        if name in self._parameters:
            existing_param = self._parameters[name]
            if isinstance(value, Parameter) and param is not existing_param:
                raise ValueError(f"Parameter '{name}' already exists with a different value")
            param = existing_param
        else:
            self._parameters[name] = param

        return name, param

    def h(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(H, qubit))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(X, qubit))
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(Y, qubit))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(Z, qubit))
        return self

    def i(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(I, qubit))
        return self

    def s(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(S, qubit))
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(T, qubit))
        return self

    def rx(self, theta: Union[float, Parameter], qubit: int) -> "QuantumCircuit":
        name, param = self._add_parameter(theta)
        self._instructions.append(
            CircuitInstruction(RX, qubit, params={"theta": param}, param_names=["theta"])
        )
        return self

    def ry(self, theta: Union[float, Parameter], qubit: int) -> "QuantumCircuit":
        name, param = self._add_parameter(theta)
        self._instructions.append(
            CircuitInstruction(RY, qubit, params={"theta": param}, param_names=["theta"])
        )
        return self

    def rz(self, theta: Union[float, Parameter], qubit: int) -> "QuantumCircuit":
        name, param = self._add_parameter(theta)
        self._instructions.append(
            CircuitInstruction(RZ, qubit, params={"theta": param}, param_names=["theta"])
        )
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(CNOT, [control, target]))
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(CZ, [control, target]))
        return self

    def swap(self, qubit1: int, qubit2: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(SWAP, [qubit1, qubit2]))
        return self

    def toffoli(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        self._instructions.append(CircuitInstruction(TOFFOLI, [control1, control2, target]))
        return self

    def append(self, gate: Gate, qubits: Union[int, list[int]], params: Optional[dict] = None) -> "QuantumCircuit":
        if params:
            resolved_params = {}
            param_names = []
            for pname, pvalue in params.items():
                name, param = self._add_parameter(pvalue, name=pname if isinstance(pvalue, (int, float)) else None)
                resolved_params[pname] = param
                param_names.append(pname)
            self._instructions.append(
                CircuitInstruction(gate, qubits, params=resolved_params, param_names=param_names)
            )
        else:
            self._instructions.append(CircuitInstruction(gate, qubits))
        return self

    def _resolve_params(self, param_values: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resolved = {}
        for name, param in self._parameters.items():
            if param_values and name in param_values:
                val = param_values[name]
                if hasattr(val, 'value'):
                    resolved[name] = float(val.value)
                else:
                    resolved[name] = float(val)
            else:
                resolved[name] = float(param.value)
        return resolved

    def run(
        self,
        param_values: Optional[dict[str, Any]] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> StateVector:
        resolved_params = self._resolve_params(param_values)
        state = StateVector(self._num_qubits)

        for instr in self._instructions:
            gate_params = {}
            for pname in instr.param_names:
                param_obj = instr.params[pname]
                if hasattr(param_obj, 'name') and param_obj.name in resolved_params:
                    gate_params[pname] = resolved_params[param_obj.name]
                elif isinstance(param_obj, Parameter):
                    gate_params[pname] = float(param_obj.value)
                else:
                    gate_params[pname] = param_obj

            state.apply_gate(instr.gate, instr.qubits, gate_params if gate_params else None)

            if noise_model is not None:
                noise_model.apply_noise_for_gate(state, instr.gate.name, instr.qubits)

        return state

    def expectation_value(
        self,
        observable: Union[PauliOp, Hamiltonian],
        param_values: Optional[dict[str, Any]] = None,
        noise_model: Optional[NoiseModel] = None,
        shots: Optional[int] = None,
    ) -> float:
        if shots is not None and shots > 0:
            total = 0.0
            for _ in range(shots):
                state = self.run(param_values, noise_model)
                total += state.expectation_value(observable)
            return total / shots
        state = self.run(param_values, noise_model)
        return state.expectation_value(observable)

    def _get_parameters_list(self, parameters: Optional[Union[str, Parameter, list[Union[str, Parameter]]]] = None) -> list[Parameter]:
        if parameters is None:
            return list(self._parameters.values())

        if isinstance(parameters, str):
            if parameters not in self._parameters:
                raise ValueError(f"Parameter '{parameters}' not found")
            return [self._parameters[parameters]]

        if isinstance(parameters, Parameter):
            return [parameters]

        if isinstance(parameters, list):
            result = []
            for p in parameters:
                if isinstance(p, str):
                    if p not in self._parameters:
                        raise ValueError(f"Parameter '{p}' not found")
                    result.append(self._parameters[p])
                elif isinstance(p, Parameter):
                    result.append(p)
                else:
                    raise ValueError(f"Invalid parameter type: {type(p)}")
            return result

        raise ValueError(f"Invalid parameters type: {type(parameters)}")

    def gradient(
        self,
        parameters: Optional[Union[str, Parameter, list[Union[str, Parameter]]]] = None,
        observable: Optional[Union[PauliOp, Hamiltonian]] = None,
        method: str = "parameter_shift",
        noise_model: Optional[NoiseModel] = None,
    ) -> Union[float, list[float]]:
        params_list = self._get_parameters_list(parameters)

        if not params_list:
            return []

        if observable is None:
            raise ValueError("Observable is required for gradient computation")

        def cost_fn():
            return self.expectation_value(observable, noise_model=noise_model)

        if method == "parameter_shift":
            grads = parameter_shift_gradient(cost_fn, params_list)
        elif method == "numerical":
            grads = numerical_gradient(cost_fn, params_list)
        else:
            raise ValueError(f"Unknown gradient method: {method}. Use 'parameter_shift' or 'numerical'.")

        single_param = parameters is not None and (
            isinstance(parameters, str) or isinstance(parameters, Parameter)
        )
        return grads[0] if single_param else grads

    def measure(
        self,
        qubits: Union[int, list[int]],
        param_values: Optional[dict[str, Any]] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> int:
        state = self.run(param_values, noise_model)
        _, result = state.measure(qubits)
        return result

    def sample(
        self,
        shots: int = 1024,
        param_values: Optional[dict[str, Any]] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> dict[str, int]:
        counts = {}
        for _ in range(shots):
            state = self.run(param_values, noise_model)
            all_qubits = list(range(self._num_qubits))
            _, result = state.measure(all_qubits)
            bitstring = format(result, f"0{self._num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._instructions)

    def __repr__(self) -> str:
        return f"QuantumCircuit({self._num_qubits} qubits, {len(self._instructions)} gates)"

    def __str__(self) -> str:
        lines = [f"QuantumCircuit({self._num_qubits} qubits):"]
        for instr in self._instructions:
            gate_name = instr.gate.name
            qubits_str = ",".join(map(str, instr.qubits))
            param_str = ""
            if instr.param_names:
                param_vals = []
                for pname in instr.param_names:
                    param = instr.params[pname]
                    val = float(param.value) if isinstance(param, Parameter) else param
                    param_vals.append(f"{pname}={val:.4f}")
                param_str = f"({', '.join(param_vals)})"
            lines.append(f"  {gate_name}{param_str} q[{qubits_str}]")
        return "\n".join(lines)

    def copy(self) -> "QuantumCircuit":
        new_circuit = QuantumCircuit(self._num_qubits)
        new_circuit._instructions = self._instructions.copy()
        new_circuit._parameters = self._parameters.copy()
        new_circuit._parameter_counter = self._parameter_counter
        return new_circuit

    def reset(self) -> None:
        self._instructions = []
        self._parameters = {}
        self._parameter_counter = 0
