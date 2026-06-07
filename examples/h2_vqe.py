#!/usr/bin/env python3
"""
VQE Example: Solving H2 Molecule Ground State Energy

This script demonstrates the use of the quantum simulator to run
Variational Quantum Eigensolver (VQE) to find the ground state energy
of the hydrogen molecule (H2).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_sim import (
    QuantumCircuit,
    Parameter,
    Hamiltonian,
    VQE,
    h2_hamiltonian,
    h2_uccsd_ansatz,
    h2_hwe_ansatz,
    get_backend,
    set_backend,
)


def main():
    print("=" * 70)
    print("VQE for H2 Molecule Ground State Energy")
    print("=" * 70)
    print()

    backend = get_backend()
    print(f"Using backend: {backend}")
    print()

    bond_length = 0.74
    print(f"1. Constructing H2 Hamiltonian at bond length {bond_length} Angstrom...")
    h2_h = h2_hamiltonian(bond_length=bond_length)
    print(f"   Hamiltonian: {h2_h}")
    print()

    print("2. Computing exact ground state energy (classical diagonalization)...")
    exact_energy = h2_h.ground_state_energy()
    print(f"   Exact ground state energy: {exact_energy:.8f} Hartree")
    print()

    print("3. Building UCCSD-inspired ansatz...")
    ansatz = h2_uccsd_ansatz(num_qubits=2)
    print(f"   {ansatz}")
    print()

    print("4. Running VQE optimization...")
    vqe = VQE(
        ansatz=ansatz,
        hamiltonian=h2_h,
        optimizer="L-BFGS-B",
        use_analytic_gradients=True,
    )

    initial_params = np.array([0.01])
    result = vqe.run(
        initial_params=initial_params,
        max_iterations=50,
        tol=1e-10,
        verbose=True,
    )
    print()

    print("5. Results summary:")
    print(f"   Exact energy:    {exact_energy:.8f} Hartree")
    print(f"   VQE energy:      {result.optimal_energy:.8f} Hartree")
    print(f"   Error:           {abs(result.optimal_energy - exact_energy):.2e} Hartree")
    print(f"   Optimal params:  {result.optimal_parameters}")
    print(f"   Converged:       {result.converged}")
    print(f"   Iterations:      {result.n_iterations}")
    print()

    print("6. Comparing with hardware-efficient ansatz...")
    hwe_ansatz = h2_hwe_ansatz(num_qubits=2, depth=1)
    print(f"   HWE ansatz: {hwe_ansatz}")

    vqe_hwe = VQE(
        ansatz=hwe_ansatz,
        hamiltonian=h2_h,
        optimizer="L-BFGS-B",
        use_analytic_gradients=True,
    )

    n_params_hwe = len(hwe_ansatz.parameters)
    initial_params_hwe = np.random.randn(n_params_hwe) * 0.1
    result_hwe = vqe_hwe.run(
        initial_params=initial_params_hwe,
        max_iterations=50,
        tol=1e-10,
        verbose=True,
    )

    print()
    print("7. Potential energy surface scan...")
    bond_lengths = np.linspace(0.5, 2.0, 16)
    energies = []
    exact_energies = []

    for r in bond_lengths:
        h = h2_hamiltonian(bond_length=r)
        e_exact = h.ground_state_energy()

        ansatz_scan = h2_uccsd_ansatz(num_qubits=2)
        vqe_scan = VQE(
            ansatz=ansatz_scan,
            hamiltonian=h,
            optimizer="L-BFGS-B",
            use_analytic_gradients=True,
        )
        result_scan = vqe_scan.run(
            initial_params=np.array([0.01]),
            max_iterations=30,
            tol=1e-8,
            verbose=False,
        )

        energies.append(result_scan.optimal_energy)
        exact_energies.append(e_exact)
        print(f"   r = {r:.2f} Angstrom: VQE = {result_scan.optimal_energy:.6f}, "
              f"Exact = {e_exact:.6f}, Error = {abs(result_scan.optimal_energy - e_exact):.2e}")

    print()
    print("8. Gradient verification...")
    test_circuit = QuantumCircuit(2)
    test_param = Parameter(0.5, name="test_theta")
    test_circuit.rx(test_param, 0)
    test_circuit.cnot(0, 1)
    test_circuit.ry(Parameter(0.3, name="test_phi"), 1)

    test_obs = Hamiltonian([
        (1.0, "ZZ"),
        (0.5, "XX"),
    ])

    analytic_grads = test_circuit.gradient(
        parameters=["test_theta", "test_phi"],
        observable=test_obs,
        method="parameter_shift",
    )

    numerical_grads = test_circuit.gradient(
        parameters=["test_theta", "test_phi"],
        observable=test_obs,
        method="numerical",
    )

    print(f"   Analytic gradients:  {analytic_grads}")
    print(f"   Numerical gradients: {numerical_grads}")
    print(f"   Difference:          {[abs(a - n) for a, n in zip(analytic_grads, numerical_grads)]}")
    print()

    print("=" * 70)
    print("VQE Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
