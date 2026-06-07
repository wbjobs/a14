#!/usr/bin/env python3
"""
API Demo: Basic usage of the quantum simulator

This script demonstrates the core API as specified in the requirements:
- circuit = QuantumCircuit(20)
- circuit.rx(0.5, 0)
- grad = circuit.gradient(parameter)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_sim import (
    QuantumCircuit,
    Parameter,
    Hamiltonian,
    get_backend,
)


def main():
    print("=" * 60)
    print("Quantum Circuit Simulator - API Demo")
    print("=" * 60)
    print()

    backend = get_backend()
    print(f"Backend: {backend}")
    print()

    print("1. Creating a 20-qubit quantum circuit...")
    circuit = QuantumCircuit(20)
    print(f"   {circuit}")
    print()

    print("2. Adding parameterized gates...")
    theta = Parameter(0.5, name="theta")
    circuit.rx(theta, 0)
    circuit.ry(0.3, 1)
    circuit.rz(Parameter(0.7, name="phi"), 2)

    circuit.h(3)
    circuit.cnot(3, 4)
    circuit.cnot(4, 5)

    circuit.toffoli(0, 1, 6)

    print(f"   Circuit after adding gates:")
    print(circuit)
    print()

    print("3. Running the circuit...")
    state = circuit.run()
    print(f"   State vector: {state}")
    print(f"   Number of amplitudes: {2**20:,}")
    print()

    print("4. Computing expectation value...")
    obs = Hamiltonian([
        (1.0, "ZIIIIIIIIIIIIIIIIIII"),
        (0.5, "IZIIIIIIIIIIIIIIIIII"),
        (0.3, "IIZIIIIIIIIIIIIIIIII"),
    ])
    exp_val = circuit.expectation_value(obs)
    print(f"   <O> = {exp_val:.8f}")
    print()

    print("5. Computing gradients...")
    grad_theta = circuit.gradient(parameters="theta", observable=obs)
    grad_phi = circuit.gradient(parameters="phi", observable=obs)
    print(f"   d<O>/d(theta) = {grad_theta:.8f}")
    print(f"   d<O>/d(phi)   = {grad_phi:.8f}")
    print()

    print("6. All parameters gradient...")
    all_params = ["theta", "phi"]
    grads = circuit.gradient(parameters=all_params, observable=obs)
    for name, g in zip(all_params, grads):
        print(f"   d<O>/d({name}) = {g:.8f}")
    print()

    print("7. Sampling from the circuit...")
    counts = circuit.sample(shots=1000)
    top_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   Top 5 results from 1000 shots:")
    for bitstring, count in top_results:
        print(f"     |{bitstring[:8]}...⟩: {count} times")
    print()

    print("8. Measuring specific qubits...")
    result = circuit.measure([0, 1, 2])
    print(f"   Measurement result (qubits 0,1,2): {result:03b}")
    print()

    print("=" * 60)
    print("API Demo Complete!")
    print("=" * 60)
    print()
    print("Key features demonstrated:")
    print("  - Up to 30 qubit state vector simulation")
    print("  - Parameterized gates (Rx, Ry, Rz)")
    print("  - Automatic differentiation via parameter-shift rule")
    print("  - Expectation value calculation")
    print("  - Sampling and measurement")
    print("  - GPU acceleration support (via CuPy/CUDA)")


if __name__ == "__main__":
    main()
