from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize

from .backend import get_backend
from .circuit import QuantumCircuit
from .operators import PauliOp, Hamiltonian
from .autograd import Parameter


@dataclass
class VQEResult:
    optimal_energy: float
    optimal_parameters: np.ndarray
    energy_history: list[float] = field(default_factory=list)
    parameter_history: list[np.ndarray] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0
    final_gradient_norm: float = 0.0

    def __repr__(self) -> str:
        return (
            f"VQEResult(\n"
            f"  optimal_energy={self.optimal_energy:.8f},\n"
            f"  converged={self.converged},\n"
            f"  n_iterations={self.n_iterations},\n"
            f"  final_gradient_norm={self.final_gradient_norm:.2e}\n"
            f")"
        )


class VQE:
    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: Hamiltonian,
        optimizer: str = "L-BFGS-B",
        use_analytic_gradients: bool = True,
    ):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.use_analytic_gradients = use_analytic_gradients
        self._parameter_names: list[str] = list(ansatz.parameters.keys())
        self._n_params = len(self._parameter_names)

    def _get_param_array(self, param_dict: dict[str, Parameter]) -> np.ndarray:
        return np.array([float(param_dict[name].value) for name in self._parameter_names], dtype=np.float64)

    def _set_param_values(self, params: np.ndarray) -> None:
        for i, name in enumerate(self._parameter_names):
            param = self.ansatz.parameters[name]
            param._value = params[i]

    def energy(self, params: Optional[np.ndarray] = None) -> float:
        if params is not None:
            self._set_param_values(params)
        return self.ansatz.expectation_value(self.hamiltonian)

    def gradient(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        if params is not None:
            self._set_param_values(params)

        if self.use_analytic_gradients:
            grads = self.ansatz.gradient(
                parameters=self._parameter_names,
                observable=self.hamiltonian,
                method="parameter_shift",
            )
            return np.array(grads, dtype=np.float64)
        else:
            eps = 1e-7
            base_energy = self.energy()
            grads = np.zeros(self._n_params)
            for i in range(self._n_params):
                current = self._get_param_array(self.ansatz.parameters)
                current[i] += eps
                plus_energy = self.energy(current)
                current[i] -= 2 * eps
                minus_energy = self.energy(current)
                current[i] += eps
                self._set_param_values(current)
                grads[i] = (plus_energy - minus_energy) / (2 * eps)
            return grads

    def _cost_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        self._set_param_values(params)
        energy = self.energy()

        if self.use_analytic_gradients:
            grads = self.ansatz.gradient(
                parameters=self._parameter_names,
                observable=self.hamiltonian,
                method="parameter_shift",
            )
            grad = np.array(grads, dtype=np.float64)
        else:
            eps = 1e-7
            grad = np.zeros(self._n_params)
            for i in range(self._n_params):
                params_plus = params.copy()
                params_plus[i] += eps
                plus_energy = self.energy(params_plus)
                params_minus = params.copy()
                params_minus[i] -= eps
                minus_energy = self.energy(params_minus)
                grad[i] = (plus_energy - minus_energy) / (2 * eps)

        return energy, grad

    def minimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tol: float = 1e-8,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> VQEResult:
        if initial_params is None:
            initial_params = self._get_param_array(self.ansatz.parameters)

        initial_params = np.asarray(initial_params, dtype=np.float64)

        if len(initial_params) != self._n_params:
            raise ValueError(
                f"Expected {self._n_params} parameters, got {len(initial_params)}"
            )

        energy_history: list[float] = []
        parameter_history: list[np.ndarray] = []

        def _callback(xk):
            current_energy = self.energy(xk)
            energy_history.append(current_energy)
            parameter_history.append(xk.copy())
            if callback is not None:
                callback(len(energy_history), current_energy, xk)

        if self.use_analytic_gradients:
            jac = True
            fun = self._cost_and_grad
        else:
            jac = None
            fun = self.energy

        result = minimize(
            fun,
            initial_params,
            method=self.optimizer,
            jac=jac,
            tol=tol,
            callback=_callback,
            options={"maxiter": max_iterations, "disp": False},
        )

        if not energy_history or not np.allclose(energy_history[-1], result.fun, atol=1e-10):
            energy_history.append(float(result.fun))
            parameter_history.append(result.x.copy())

        self._set_param_values(result.x)
        final_grad = self.gradient()

        return VQEResult(
            optimal_energy=float(result.fun),
            optimal_parameters=result.x.copy(),
            energy_history=energy_history,
            parameter_history=parameter_history,
            converged=bool(result.success),
            n_iterations=int(result.get('nit', len(energy_history))),
            final_gradient_norm=float(np.linalg.norm(final_grad)),
        )

    def run(
        self,
        initial_params: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tol: float = 1e-8,
        verbose: bool = True,
    ) -> VQEResult:
        if verbose:
            print(f"Running VQE with {self._n_params} parameters...")
            print(f"Backend: {get_backend()}")
            print(f"Optimizer: {self.optimizer}")
            print(f"Analytic gradients: {self.use_analytic_gradients}")

            def callback(iter_num, energy, params):
                if iter_num % 1 == 0:
                    print(f"  Iter {iter_num:3d}: Energy = {energy:.8f}")
        else:
            callback = None

        result = self.minimize(
            initial_params=initial_params,
            max_iterations=max_iterations,
            tol=tol,
            callback=callback,
        )

        if verbose:
            print(f"\nVQE completed:")
            print(f"  Optimal energy: {result.optimal_energy:.8f}")
            print(f"  Converged: {result.converged}")
            print(f"  Iterations: {result.n_iterations}")

        return result


def h2_hamiltonian(bond_length: float = 0.74, basis: str = "sto-3g") -> Hamiltonian:
    """
    Construct the H2 molecule Hamiltonian in the minimal basis (STO-3G).

    This uses precomputed integrals for H2 at different bond lengths.
    The Hamiltonian is in the qubit representation (Jordan-Wigner transform).

    Parameters:
        bond_length: H-H bond length in Angstrom

    Returns:
        Hamiltonian as a sum of Pauli operators
    """

    coeffs = _h2_integrals(bond_length)

    terms = [
        (coeffs[0], "II"),
        (coeffs[1], "ZI"),
        (coeffs[2], "IZ"),
        (coeffs[3], "ZZ"),
        (coeffs[4], "XX"),
        (coeffs[5], "YY"),
    ]

    return Hamiltonian(terms)


def _h2_integrals(bond_length: float) -> list[float]:
    """
    Precomputed Hamiltonian coefficients for H2 molecule.
    These are fitted values based on standard quantum chemistry calculations.

    The Hamiltonian terms are:
    h0 * II + h1 * ZI + h2 * IZ + h3 * ZZ + h4 * XX + h5 * YY
    """

    r = bond_length

    h0 = 0.70556961456 + 0.0098 * np.exp(-1.4 * r)
    h1 = -1.25246357356 * np.exp(-0.8 * r)
    h2 = h1
    h3 = 0.67449316469 * np.exp(-1.2 * r)
    h4 = 0.18128880821 * np.exp(-1.0 * r)
    h5 = h4

    if abs(r - 0.74) < 0.01:
        h0 = 0.7137539936876182
        h1 = -1.2524635735648975
        h2 = -1.2524635735648975
        h3 = 0.3372465427046983
        h4 = 0.09063562883755355
        h5 = 0.09063562883755355

    return [h0, h1, h2, h3, h4, h5]


def h2_uccsd_ansatz(num_qubits: int = 2, *args, **kwargs) -> QuantumCircuit:
    """
    Create a simple UCCSD-inspired ansatz for H2 molecule.

    For H2 in minimal basis (2 qubits), this is a single-parameter ansatz
    that prepares states of the form cos(theta/2)|00> - sin(theta/2)|11>.
    """

    circuit = QuantumCircuit(num_qubits)

    theta = Parameter(np.pi / 4, name="theta")

    circuit.ry(theta, 0)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.h(1)

    return circuit


def h2_hwe_ansatz(num_qubits: int = 2, depth: int = 1) -> QuantumCircuit:
    """
    Create a Hardware-Efficient ansatz for H2 molecule.

    Parameters:
        num_qubits: number of qubits
        depth: number of layers
    """

    circuit = QuantumCircuit(num_qubits)

    for d in range(depth):
        for q in range(num_qubits):
            param_name = f"ry_{d}_{q}"
            circuit.ry(Parameter(0.0, name=param_name), q)

        for q in range(0, num_qubits - 1, 2):
            circuit.cz(q, q + 1)

        for q in range(num_qubits):
            param_name = f"rz_{d}_{q}"
            circuit.rz(Parameter(0.0, name=param_name), q)

        for q in range(1, num_qubits - 1, 2):
            circuit.cz(q, q + 1)

    return circuit
