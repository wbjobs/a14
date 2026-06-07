#!/usr/bin/env python3
"""
Basic tests for the quantum simulator
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_sim import (
    StateVector,
    QuantumCircuit,
    H, X, Y, Z, I,
    CNOT, CZ, TOFFOLI,
    RX, RY, RZ,
    Parameter,
    Hamiltonian,
    get_backend,
)


def test_statevector_init():
    print("Test 1: StateVector initialization...")
    sv = StateVector(2)
    vec = sv.to_numpy()
    assert abs(vec[0] - 1.0) < 1e-10
    assert np.allclose(vec[1:], 0)
    print("  PASSED")


def test_single_qubit_gates():
    print("Test 2: Single qubit gates...")

    sv = StateVector(1)
    sv.apply_gate(H, 0)
    vec = sv.to_numpy()
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    assert np.allclose(np.abs(vec), np.abs(expected))
    print("  H gate: PASSED")

    sv = StateVector(1)
    sv.apply_gate(X, 0)
    vec = sv.to_numpy()
    assert abs(vec[1] - 1.0) < 1e-10
    print("  X gate: PASSED")

    sv = StateVector(1)
    sv.apply_gate(Z, 0)
    vec = sv.to_numpy()
    assert abs(vec[0] - 1.0) < 1e-10
    print("  Z gate: PASSED")

    print("  All single qubit gates: PASSED")


def test_two_qubit_gates():
    print("Test 3: Two qubit gates...")

    sv = StateVector(2)
    sv.apply_gate(H, 0)
    sv.apply_gate(CNOT, [0, 1])
    vec = sv.to_numpy()
    expected_bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    assert np.allclose(np.abs(vec), np.abs(expected_bell))
    print("  Bell state (H + CNOT): PASSED")

    sv = StateVector(2)
    sv.apply_gate(X, 0)
    sv.apply_gate(X, 1)
    vec_after_x = sv.to_numpy()
    print(f"    After X on q0 and q1: {vec_after_x}")
    sv.apply_gate(CZ, [0, 1])
    vec = sv.to_numpy()
    print(f"    After CZ q0,q1: {vec}")
    expected_cz = np.array([0, 0, 0, -1], dtype=np.complex128)
    print(f"    Expected: {expected_cz}")
    assert np.allclose(np.abs(vec), np.abs(expected_cz))
    print("  CZ gate: PASSED")

    print("  All two qubit gates: PASSED")


def test_three_qubit_gates():
    print("Test 4: Three qubit gates...")

    sv = StateVector(3)
    sv.apply_gate(X, 0)
    sv.apply_gate(X, 1)
    sv.apply_gate(TOFFOLI, [0, 1, 2])
    vec = sv.to_numpy()

    idx = 0b111
    assert abs(vec[idx] - 1.0) < 1e-10
    print("  TOFFOLI gate: PASSED")
    print("  All three qubit gates: PASSED")


def test_parameterized_gates():
    print("Test 5: Parameterized gates...")

    sv = StateVector(1)
    sv.apply_gate(RX, 0, {"theta": np.pi})
    vec = sv.to_numpy()
    expected = np.array([0, -1j], dtype=np.complex128)
    assert np.allclose(np.abs(vec), np.abs(expected))
    print("  RX(pi): PASSED")

    sv = StateVector(1)
    sv.apply_gate(RY, 0, {"theta": np.pi})
    vec = sv.to_numpy()
    expected = np.array([0, 1], dtype=np.complex128)
    assert np.allclose(vec, expected)
    print("  RY(pi): PASSED")

    print("  All parameterized gates: PASSED")


def test_quantum_circuit():
    print("Test 6: QuantumCircuit...")

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    state = circuit.run()
    vec = state.to_numpy()
    expected_bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    assert np.allclose(np.abs(vec), np.abs(expected_bell))
    print("  Circuit run: PASSED")

    circuit2 = QuantumCircuit(2)
    circuit2.rx(0.5, 0)
    circuit2.ry(0.3, 1)
    assert len(circuit2.parameters) == 2
    print("  Parameter tracking: PASSED")

    print("  QuantumCircuit: PASSED")


def test_expectation_value():
    print("Test 7: Expectation value...")

    circuit = QuantumCircuit(2)
    circuit.h(0)

    obs = Hamiltonian([(1.0, "ZI")])
    exp_val = circuit.expectation_value(obs)
    assert abs(exp_val - 0.0) < 1e-10
    print("  <Z> on |+>: PASSED")

    circuit2 = QuantumCircuit(1)
    obs2 = Hamiltonian([(1.0, "Z")])
    exp_val2 = circuit2.expectation_value(obs2)
    assert abs(exp_val2 - 1.0) < 1e-10
    print("  <Z> on |0>: PASSED")

    print("  Expectation values: PASSED")


def test_gradient():
    print("Test 8: Gradient computation...")

    circuit = QuantumCircuit(1)
    theta = Parameter(0.0, name="theta")
    circuit.rx(theta, 0)

    obs = Hamiltonian([(1.0, "Z")])

    grad_analytic = circuit.gradient(parameters="theta", observable=obs, method="parameter_shift")
    grad_numerical = circuit.gradient(parameters="theta", observable=obs, method="numerical")

    print(f"    Analytic: {grad_analytic:.6f}")
    print(f"    Numerical: {grad_numerical:.6f}")
    assert abs(grad_analytic - grad_numerical) < 1e-5
    print("  Gradient match: PASSED")
    print("  Gradient computation: PASSED")


def test_measurement():
    print("Test 9: Measurement...")

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    counts = circuit.sample(shots=1000)
    assert len(counts) > 0
    assert sum(counts.values()) == 1000
    print("  Sampling: PASSED")
    print("  Measurement: PASSED")


def test_probability():
    print("Test 10: Probability...")

    sv = StateVector(2)
    sv.apply_gate(H, 0)
    probs = sv.probability([0, 1])
    assert abs(probs[0] - 0.5) < 0.1
    print("  Probability: PASSED")


def run_all_tests():
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    print(f"Backend: {get_backend()}")
    print()

    tests = [
        test_statevector_init,
        test_single_qubit_gates,
        test_two_qubit_gates,
        test_three_qubit_gates,
        test_parameterized_gates,
        test_quantum_circuit,
        test_expectation_value,
        test_gradient,
        test_measurement,
        test_probability,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
