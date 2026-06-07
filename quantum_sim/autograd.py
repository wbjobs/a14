from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend


class Parameter:
    def __init__(self, value: float | np.ndarray | Any, name: Optional[str] = None):
        backend = get_backend()
        self._value = backend.to_device(value) if not hasattr(value, '__array_function__') else value
        self._name = name
        self._grad: Optional[Any] = None
        self._requires_grad: bool = True

    @property
    def value(self) -> Any:
        return self._value

    @property
    def grad(self) -> Optional[Any]:
        return self._grad

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    def zero_grad(self) -> None:
        self._grad = None

    def backward(self, grad: Any = None) -> None:
        backend = get_backend()
        if grad is None:
            grad = backend.xp.ones_like(self._value)
        self._grad = grad

    def __repr__(self) -> str:
        return f"Parameter(value={self._value}, name={self._name}, requires_grad={self._requires_grad})"

    def __float__(self) -> float:
        backend = get_backend()
        val = backend.to_numpy(self._value)
        return float(val)

    def __add__(self, other: Union["Parameter", float, Any]) -> "Parameter":
        if isinstance(other, Parameter):
            return Parameter(self._value + other._value, name=f"({self._name}+{other._name})")
        return Parameter(self._value + other, name=f"({self._name}+{other})")

    def __radd__(self, other: Union[float, Any]) -> "Parameter":
        return Parameter(other + self._value, name=f"({other}+{self._name})")

    def __mul__(self, other: Union["Parameter", float, Any]) -> "Parameter":
        if isinstance(other, Parameter):
            return Parameter(self._value * other._value, name=f"({self._name}*{other._name})")
        return Parameter(self._value * other, name=f"({self._name}*{other})")

    def __rmul__(self, other: Union[float, Any]) -> "Parameter":
        return Parameter(other * self._value, name=f"({other}*{self._name})")


class ComputationNode:
    def __init__(self, operation: str, inputs: list[Any], output: Any):
        self.operation = operation
        self.inputs = inputs
        self.output = output
        self.grad_fn = None

    def backward(self, grad_output: Any) -> list[Any]:
        if self.grad_fn is None:
            raise RuntimeError(f"No gradient function for operation: {self.operation}")
        return self.grad_fn(grad_output)


class ComputationGraph:
    def __init__(self):
        self.nodes: list[ComputationNode] = []
        self.tape: list[ComputationNode] = []

    def add_node(self, node: ComputationNode) -> None:
        self.nodes.append(node)
        self.tape.append(node)

    def clear(self) -> None:
        self.nodes = []
        self.tape = []

    def backward(self, parameters: list[Parameter]) -> None:
        for param in parameters:
            param.zero_grad()

        grad_output = None
        for node in reversed(self.tape):
            grads = node.backward(grad_output)
            for i, inp in enumerate(node.inputs):
                if isinstance(inp, Parameter) and inp.requires_grad:
                    if inp._grad is None:
                        inp._grad = grads[i]
                    else:
                        inp._grad += grads[i]
            grad_output = grads[0] if len(grads) == 1 else grads

    def __enter__(self) -> "ComputationGraph":
        self.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


_global_graph: Optional[ComputationGraph] = None


def get_graph() -> ComputationGraph:
    global _global_graph
    if _global_graph is None:
        _global_graph = ComputationGraph()
    return _global_graph


def parameter_shift_gradient(
    func,
    params: list[Parameter],
    shift: float = np.pi / 2,
) -> list[float]:
    gradients = []
    backend = get_backend()
    xp = backend.xp

    denom = 2 * np.sin(shift)
    if abs(denom) < 1e-10:
        denom = 1e-10 if denom >= 0 else -1e-10

    for i, param in enumerate(params):
        original_value = param._value.copy() if hasattr(param._value, 'copy') else param._value

        param._value = original_value + shift
        plus_val = func()

        param._value = original_value - shift
        minus_val = func()

        param._value = original_value

        plus_val = xp.asarray(plus_val)
        minus_val = xp.asarray(minus_val)

        plus_val = xp.where(xp.isnan(plus_val), 0.0, plus_val)
        minus_val = xp.where(xp.isnan(minus_val), 0.0, minus_val)
        plus_val = xp.where(xp.isinf(plus_val), 1e30, plus_val)
        minus_val = xp.where(xp.isinf(minus_val), 1e30, minus_val)

        grad = (plus_val - minus_val) / denom
        grad = xp.where(xp.isnan(grad), 0.0, grad)
        grad = xp.where(xp.isinf(grad), 0.0, grad)

        gradients.append(float(grad))

    return gradients


def adjoint_differentiation(
    circuit_forward,
    circuit_backward,
    observable,
    params: list[Parameter],
) -> tuple[float, list[float]]:
    backend = get_backend()
    xp = backend.xp

    state = circuit_forward(params)
    expectation = state.expectation_value(observable)

    gradients = []
    for i, param in enumerate(params):
        if not param.requires_grad:
            gradients.append(0.0)
            continue

        grad = 0.0
        gate_deriv = circuit_backward(params, i)

        for term in gate_deriv:
            coeff = term[0]
            apply_gate_fn = term[1]

            temp_state = state.copy()
            apply_gate_fn(temp_state)

            bra = xp.reshape(temp_state.data.conj(), -1, order='F')

            obs_state = state.copy()
            for t in observable.terms:
                t_coeff = t[0]
                pauli_string = t[1]
                for qubit, pauli in enumerate(pauli_string):
                    if pauli == "I":
                        continue
                    elif pauli == "X":
                        from .gates import X
                        obs_state.apply_gate(X, qubit)
                    elif pauli == "Y":
                        from .gates import Y
                        obs_state.apply_gate(Y, qubit)
                    elif pauli == "Z":
                        from .gates import Z
                        obs_state.apply_gate(Z, qubit)

                ket = xp.reshape(obs_state.data, -1, order='F')
                grad += coeff * t_coeff * xp.real(xp.sum(bra * ket))

        gradients.append(float(grad))

    return float(expectation), gradients


def numerical_gradient(
    func,
    params: list[Parameter],
    eps: float = 1e-7,
) -> list[float]:
    gradients = []
    backend = get_backend()
    xp = backend.xp

    denom = 2 * eps
    if abs(denom) < 1e-15:
        denom = 1e-15 if denom >= 0 else -1e-15

    for i, param in enumerate(params):
        original_value = param._value.copy() if hasattr(param._value, 'copy') else param._value

        param._value = original_value + eps
        plus_val = func()

        param._value = original_value - eps
        minus_val = func()

        param._value = original_value

        plus_val = xp.asarray(plus_val)
        minus_val = xp.asarray(minus_val)

        plus_val = xp.where(xp.isnan(plus_val), 0.0, plus_val)
        minus_val = xp.where(xp.isnan(minus_val), 0.0, minus_val)
        plus_val = xp.where(xp.isinf(plus_val), 1e30, plus_val)
        minus_val = xp.where(xp.isinf(minus_val), 1e30, minus_val)

        grad = (plus_val - minus_val) / denom
        grad = xp.where(xp.isnan(grad), 0.0, grad)
        grad = xp.where(xp.isinf(grad), 0.0, grad)

        gradients.append(float(grad))

    return gradients
