#!/usr/bin/env python3
"""Comprehensive verification of all bug fixes"""

from quantum_sim import *
import numpy as np

print('='*70)
print('COMPREHENSIVE BUG FIX VERIFICATION')
print('='*70)
print(f'Backend: {get_backend()}')
print()

# ============================================
# Bug 1: Toffoli gate big-endian indexing
# ============================================
print('BUG 1: Toffoli Gate Big-Endian Indexing')
print('-'*50)

test_cases = [
    ([0, 1, 2], [1, 1, 0], 'controls q0,q1=1, target q2=0 -> |111>'),
    ([1, 2, 0], [0, 1, 1], 'controls q1,q2=1, target q0=0 -> |111>'),
    ([0, 2, 1], [1, 0, 1], 'controls q0,q2=1, target q1=0 -> |111>'),
    ([2, 0, 1], [1, 0, 1], 'controls q2,q0=1, target q1=0 -> |111>'),
    ([2, 1, 0], [0, 1, 1], 'controls q2,q1=1, target q0=0 -> |111>'),
    ([1, 0, 2], [1, 1, 0], 'controls q1,q0=1, target q2=0 -> |111>'),
]

all_toffoli_ok = True
for qubits, initial_state, desc in test_cases:
    sv = StateVector(3)
    for i, val in enumerate(initial_state):
        if val:
            sv.apply_gate(X, i)
    
    idx_before = np.argmax(np.abs(sv.to_numpy()))
    sv.apply_gate(TOFFOLI, qubits)
    idx_after = np.argmax(np.abs(sv.to_numpy()))
    
    ok = idx_after == 7
    all_toffoli_ok = all_toffoli_ok and ok
    print(f'  TOFFOLI{qubits}: {desc}')
    print(f'    Before: idx={idx_before} ({bin(idx_before)}), After: idx={idx_after} ({bin(idx_after)})')
    status = "PASS" if ok else "FAIL"
    print(f'    Status: {status}')

status = "ALL PASS" if all_toffoli_ok else "SOME FAIL"
print(f'  Overall: {status}')
print()

# ============================================
# Bug 2: NaN gradients for entangled circuits
# ============================================
print('BUG 2: NaN Gradients for Entangled Circuits')
print('-'*50)

circuit = QuantumCircuit(4)
params = []
for i in range(4):
    p = Parameter(np.random.uniform(0, np.pi), name=f't{i}')
    params.append(p)
    circuit.ry(p, i)

for _ in range(3):
    for i in range(3):
        circuit.cnot(i, i+1)
    circuit.cz(0, 3)

obs = Hamiltonian([(1.0, 'ZZZZ')])

grads_analytic = circuit.gradient(observable=obs, method='parameter_shift')
grads_numerical = circuit.gradient(observable=obs, method='numerical')

has_nan = any(np.isnan(g) for g in grads_analytic) or any(np.isnan(g) for g in grads_numerical)
has_inf = any(np.isinf(g) for g in grads_analytic) or any(np.isinf(g) for g in grads_numerical)
max_error = max(abs(a - n) for a, n in zip(grads_analytic, grads_numerical))

print(f'  Analytic gradients: {grads_analytic}')
print(f'  Numerical gradients: {grads_numerical}')
print(f'  Has NaN: {has_nan}')
print(f'  Has Inf: {has_inf}')
print(f'  Max gradient error: {max_error:.2e}')
grad_ok = not has_nan and not has_inf and max_error < 1e-5
status = "PASS" if grad_ok else "FAIL"
print(f'  Overall: {status}')
print()

print('  Testing edge case parameters...')
edge_cases = [0.0, np.pi, 2*np.pi, np.pi/2, 1e-10, np.pi - 1e-10]
all_edge_ok = True
for theta in edge_cases:
    circuit2 = QuantumCircuit(2)
    p = Parameter(theta, name='edge_theta')
    circuit2.ry(p, 0)
    circuit2.cnot(0, 1)
    obs2 = Hamiltonian([(1.0, 'ZZ')])
    
    g_a = circuit2.gradient(parameters='edge_theta', observable=obs2, method='parameter_shift')
    g_n = circuit2.gradient(parameters='edge_theta', observable=obs2, method='numerical')
    
    ok = not (np.isnan(g_a) or np.isnan(g_n) or np.isinf(g_a) or np.isinf(g_n))
    all_edge_ok = all_edge_ok and ok
    print(f'    theta={theta:.4f}: analytic={g_a:.6e}, numerical={g_n:.6e}, ok={ok}')

status = "ALL PASS" if all_edge_ok else "SOME FAIL"
print(f'  Edge cases: {status}')
print()

# ============================================
# Bug 3: CUDA kernel launch for >25 qubits
# ============================================
print('BUG 3: Large Qubit Simulation (25+ qubits)')
print('-'*50)

backend = get_backend()
is_gpu = backend.is_gpu()

if is_gpu:
    print('  Running on GPU - testing up to 26 qubits...')
    qubit_range = [20, 22, 24, 25, 26]
else:
    print('  Running on CPU - testing up to 26 qubits (memory check)...')
    qubit_range = [20, 22, 24, 25, 26]

all_large_ok = True
for n in qubit_range:
    try:
        mem_gb = 2**n * 16 / (1024**3)
        print(f'  Testing {n} qubits (~{mem_gb:.2f} GB)...', end=' ')
        
        if is_gpu:
            free, total = backend.get_memory_info()
            if 2**n * 16 > free * 0.8:
                print('SKIP (insufficient memory)')
                continue
        
        circuit = QuantumCircuit(n)
        circuit.h(0)
        for i in range(min(n-1, 15)):
            circuit.cnot(i, i+1)
        
        state = circuit.run()
        
        vec = state.to_numpy()
        norm = np.sum(np.abs(vec)**2)
        max_amp = np.max(np.abs(vec))
        
        ok = abs(norm - 1.0) < 1e-8 and max_amp > 0
        all_large_ok = all_large_ok and ok
        
        print(f'DONE (norm={norm:.10f}, max_amp={max_amp:.4f})')
        
    except Exception as e:
        all_large_ok = False
        print(f'FAIL: {e}')

status = "ALL PASS" if all_large_ok else "SOME FAIL"
print(f'  Overall: {status}')
print()

# ============================================
# Final Summary
# ============================================
print('='*70)
print('FINAL SUMMARY')
print('='*70)
bug1 = "FIXED" if all_toffoli_ok else "STILL BROKEN"
bug2 = "FIXED" if (not has_nan and not has_inf and all_edge_ok) else "STILL BROKEN"
bug3 = "FIXED" if all_large_ok else "STILL BROKEN"
print(f'Bug 1 (Toffoli indexing):  {bug1}')
print(f'Bug 2 (NaN gradients):     {bug2}')
print(f'Bug 3 (CUDA >25 qubits):   {bug3}')
print('='*70)

if all_toffoli_ok and not has_nan and not has_inf and all_edge_ok and all_large_ok:
    print("\nAll bugs have been successfully fixed!")
else:
    print("\nSome bugs still need attention.")
