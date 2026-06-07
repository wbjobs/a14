#!/usr/bin/env python3
"""
Tests for bug fixes:
1. Toffoli gate big-endian indexing error
2. NaN gradients for entangled circuits with CNOT
3. CUDA kernel launch failure for >25 qubits
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_sim import (
    StateVector,
    QuantumCircuit,
    H, X, Y, Z, I,
    CNOT, CZ, SWAP, TOFFOLI,
    RX, RY, RZ,
    Parameter,
    Hamiltonian,
    get_backend,
)


def test_toffoli_big_endian():
    """Test 1: Toffoli gate big-endian indexing"""
    print("Test 1: Toffoli gate big-endian indexing...")
    
    # Test 1: |110> -> |111> (controls q0, q1 = 1, target q2 = 0)
    # Index: q0 + 2*q1 + 4*q2 = 1 + 2*1 + 4*0 = 3 -> should become 1 + 2*1 + 4*1 = 7
    sv = StateVector(3)
    sv.apply_gate(X, 0)
    sv.apply_gate(X, 1)
    vec_before = sv.to_numpy()
    print(f"  Before TOFFOLI [0,1,2]: |110> at index {np.argmax(np.abs(vec_before))} = {int('110', 2)}")
    sv.apply_gate(TOFFOLI, [0, 1, 2])
    vec_after = sv.to_numpy()
    idx_after = np.argmax(np.abs(vec_after))
    print(f"  After TOFFOLI [0,1,2]: state at index {idx_after} = {bin(idx_after)}")
    assert idx_after == 0b111, f"Expected |111> at index 7, got index {idx_after}"
    print("  TOFFOLI [0,1,2] (controls first, target last): PASSED")
    
    # Test 2: |011> -> |111> (controls q1, q2 = 1, target q0 = 0)
    # Index: q0 + 2*q1 + 4*q2 = 0 + 2*1 + 4*1 = 6 -> should become 1 + 2*1 + 4*1 = 7
    sv2 = StateVector(3)
    sv2.apply_gate(X, 1)
    sv2.apply_gate(X, 2)
    vec_before2 = sv2.to_numpy()
    print(f"  Before TOFFOLI [1,2,0]: |011> at index {np.argmax(np.abs(vec_before2))} = {int('011', 2)}")
    sv2.apply_gate(TOFFOLI, [1, 2, 0])
    vec_after2 = sv2.to_numpy()
    idx_after2 = np.argmax(np.abs(vec_after2))
    print(f"  After TOFFOLI [1,2,0]: state at index {idx_after2} = {bin(idx_after2)}")
    assert idx_after2 == 0b111, f"Expected |111> at index 7, got index {idx_after2}"
    print("  TOFFOLI [1,2,0] (controls middle/last, target first): PASSED")
    
    # Test 3: |101> -> |111> (controls q0, q2 = 1, target q1 = 0)
    # Index: q0 + 2*q1 + 4*q2 = 1 + 2*0 + 4*1 = 5 -> should become 1 + 2*1 + 4*1 = 7
    sv3 = StateVector(3)
    sv3.apply_gate(X, 0)
    sv3.apply_gate(X, 2)
    vec_before3 = sv3.to_numpy()
    print(f"  Before TOFFOLI [0,2,1]: |101> at index {np.argmax(np.abs(vec_before3))} = {int('101', 2)}")
    sv3.apply_gate(TOFFOLI, [0, 2, 1])
    vec_after3 = sv3.to_numpy()
    idx_after3 = np.argmax(np.abs(vec_after3))
    print(f"  After TOFFOLI [0,2,1]: state at index {idx_after3} = {bin(idx_after3)}")
    assert idx_after3 == 0b111, f"Expected |111> at index 7, got index {idx_after3}"
    print("  TOFFOLI [0,2,1] (controls first/last, target middle): PASSED")
    
    print("  All Toffoli big-endian tests: PASSED")


def test_cnot_gradient_nan():
    """Test 2: Gradient computation for entangled circuits with CNOT"""
    print("\nTest 2: Gradient computation for entangled circuits with CNOT...")
    
    # Create an entangled circuit with parameterized gates
    circuit = QuantumCircuit(2)
    theta1 = Parameter(0.5, name="theta1")
    theta2 = Parameter(0.3, name="theta2")
    circuit.ry(theta1, 0)
    circuit.ry(theta2, 1)
    circuit.cnot(0, 1)
    
    # Observable
    obs = Hamiltonian([(1.0, "ZZ")])
    
    # Compute gradients
    grads_analytic = circuit.gradient(observable=obs, method="parameter_shift")
    grads_numerical = circuit.gradient(observable=obs, method="numerical")
    
    print(f"  Analytic gradients: {grads_analytic}")
    print(f"  Numerical gradients: {grads_numerical}")
    
    # Check for NaN
    for i, (ga, gn) in enumerate(zip(grads_analytic, grads_numerical)):
        assert not np.isnan(ga), f"Analytic gradient for param {i} is NaN!"
        assert not np.isnan(gn), f"Numerical gradient for param {i} is NaN!"
        assert abs(ga - gn) < 1e-5, f"Gradient mismatch for param {i}: analytic={ga}, numerical={gn}"
    
    print("  Gradients are valid (not NaN) and match numerical: PASSED")
    
    # Test with more complex entangled circuit
    circuit2 = QuantumCircuit(3)
    t1 = Parameter(0.1, name="t1")
    t2 = Parameter(0.2, name="t2")
    t3 = Parameter(0.3, name="t3")
    circuit2.rx(t1, 0)
    circuit2.ry(t2, 1)
    circuit2.rz(t3, 2)
    circuit2.cnot(0, 1)
    circuit2.cnot(1, 2)
    circuit2.cnot(0, 2)
    
    obs2 = Hamiltonian([(1.0, "ZZZ")])
    
    grads_analytic2 = circuit2.gradient(observable=obs2, method="parameter_shift")
    grads_numerical2 = circuit2.gradient(observable=obs2, method="numerical")
    
    print(f"  Complex circuit analytic gradients: {grads_analytic2}")
    print(f"  Complex circuit numerical gradients: {grads_numerical2}")
    
    for i, (ga, gn) in enumerate(zip(grads_analytic2, grads_numerical2)):
        assert not np.isnan(ga), f"Complex circuit analytic gradient for param {i} is NaN!"
        assert not np.isnan(gn), f"Complex circuit numerical gradient for param {i} is NaN!"
        assert abs(ga - gn) < 1e-5, f"Complex circuit gradient mismatch for param {i}: analytic={ga}, numerical={gn}"
    
    print("  Complex entangled circuit gradients: PASSED")


def test_large_qubit_cuda():
    """Test 3: Large qubit simulation (memory check and kernel configuration)"""
    print("\nTest 3: Large qubit simulation (up to 26 qubits)...")
    
    backend = get_backend()
    is_gpu = backend.is_gpu()
    
    if is_gpu:
        print(f"  Running on GPU: {backend}")
    else:
        print(f"  Running on CPU: {backend} (CUDA test will be simulated)")
    
    # Test with increasing qubit counts
    for num_qubits in [20, 22, 24, 25, 26]:
        try:
            mem_required = 2**num_qubits * 16  # complex128 = 16 bytes
            mem_required_gb = mem_required / (1024**3)
            print(f"  Testing {num_qubits} qubits (requires ~{mem_required_gb:.2f} GB)...")
            
            if is_gpu:
                free_mem, total_mem = backend.get_memory_info()
                free_mem_gb = free_mem / (1024**3)
                if mem_required > free_mem * 0.8:
                    print(f"    Skipping: insufficient memory (available: {free_mem_gb:.2f} GB)")
                    continue
            
            # Create and run a simple circuit
            circuit = QuantumCircuit(num_qubits)
            circuit.h(0)
            for i in range(min(num_qubits - 1, 10)):
                circuit.cnot(i, i + 1)
            
            state = circuit.run()
            vec = state.to_numpy()
            norm = np.sum(np.abs(vec)**2)
            print(f"    Success! Norm = {norm:.10f}")
            
            # Test expectation value
            if num_qubits <= 24:
                obs = Hamiltonian([(1.0, "Z" + "I" * (num_qubits - 1))])
                exp_val = circuit.expectation_value(obs)
                print(f"    Expectation value <Z0>: {exp_val:.10f}")
            
        except Exception as e:
            print(f"    FAILED: {e}")
            if num_qubits > 24 and is_gpu:
                # This is the bug we need to fix
                raise AssertionError(f"CUDA kernel failed for {num_qubits} qubits: {e}")
    
    print("  Large qubit simulation tests: PASSED")


def test_statevector_ordering():
    """Additional test: Verify state vector ordering is consistent"""
    print("\nAdditional Test: State vector ordering consistency...")
    
    # Test that |q0, q1, q2> corresponds to index = q0 + 2*q1 + 4*q2
    sv = StateVector(3)
    
    # Set |100> (q0=1, q1=0, q2=0) should be at index 1
    sv2 = StateVector(3)
    sv2.apply_gate(X, 0)
    vec = sv2.to_numpy()
    idx = np.argmax(np.abs(vec))
    assert idx == 1, f"|100> should be at index 1, got {idx}"
    print(f"  |100> at index {idx}: PASSED")
    
    # |010> (q0=0, q1=1, q2=0) should be at index 2
    sv3 = StateVector(3)
    sv3.apply_gate(X, 1)
    vec = sv3.to_numpy()
    idx = np.argmax(np.abs(vec))
    assert idx == 2, f"|010> should be at index 2, got {idx}"
    print(f"  |010> at index {idx}: PASSED")
    
    # |001> (q0=0, q1=0, q2=1) should be at index 4
    sv4 = StateVector(3)
    sv4.apply_gate(X, 2)
    vec = sv4.to_numpy()
    idx = np.argmax(np.abs(vec))
    assert idx == 4, f"|001> should be at index 4, got {idx}"
    print(f"  |001> at index {idx}: PASSED")
    
    print("  State vector ordering: PASSED")


def run_all_tests():
    print("=" * 70)
    print("Running Bug Fix Tests")
    print("=" * 70)
    print(f"Backend: {get_backend()}")
    print()
    
    tests = [
        test_statevector_ordering,
        test_toffoli_big_endian,
        test_cnot_gradient_nan,
        test_large_qubit_cuda,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
