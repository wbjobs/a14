import numpy as np
import sys

from quantum_sim import (
    QuantumCircuit,
    StateVector,
    Parameter,
    get_backend,
    PauliOp,
    NoiseChannel,
    NoiseModel,
    BitFlip,
    PhaseFlip,
    PhaseDamping,
    Depolarizing,
    AmplitudeDamping,
    TwoQubitDepolarizing,
    Transpiler,
    transpile,
    SingleQubitFusion,
    DoubleCXCancel,
    IIDElimination,
    DistributedContext,
    DistributedStateVector,
    DistributedQuantumCircuit,
    JITCompiledCircuit,
    JITCompiler,
    jit,
    run_jit,
    H, X, Y, Z, I, S, T, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI,
)

print('=' * 70)
print('COMPREHENSIVE TEST SUITE FOR ALL NEW FEATURES')
print('=' * 70)
print()

passed = 0
failed = 0

def run_test(name, test_fn):
    global passed, failed
    try:
        test_fn()
        print(f'✓ {name}')
        passed += 1
        return True
    except Exception as e:
        print(f'✗ {name}: {e}')
        failed += 1
        import traceback
        traceback.print_exc()
        return False

print('--- NOISE MODEL TESTS ---')
print()

def test_bit_flip_channel():
    noise = BitFlip(p=0.1)
    assert noise.p == 0.1
    assert noise.num_qubits == 1
    kraus_ops = noise.get_kraus_operators()
    assert len(kraus_ops) == 2
    state = StateVector(1)
    noise.apply(state, 0)
    probs = state.probability(0)
    assert np.isclose(np.sum(probs), 1.0)

def test_phase_damping_channel():
    noise = PhaseDamping(gamma=0.2)
    assert noise.gamma == 0.2
    kraus_ops = noise.get_kraus_operators()
    assert len(kraus_ops) == 3

def test_depolarizing_channel():
    noise = Depolarizing(p=0.3)
    assert noise.p == 0.3
    kraus_ops = noise.get_kraus_operators()
    assert len(kraus_ops) == 4

def test_noise_model_with_circuit():
    noise_model = NoiseModel()
    noise_model.add_gate_noise(CNOT, BitFlip(0.05))
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    state = qc.run(noise_model=noise_model)
    probs = state.probability()
    assert np.isclose(np.sum(probs), 1.0)

def test_sample_with_noise():
    noise_model = NoiseModel()
    noise_model.add_qubit_noise(0, BitFlip(0.5))
    qc = QuantumCircuit(1)
    samples = qc.sample(shots=1000, noise_model=noise_model)
    assert isinstance(samples, dict)
    total_shots = sum(samples.values())
    assert total_shots == 1000

run_test('BitFlip channel', test_bit_flip_channel)
run_test('PhaseDamping channel', test_phase_damping_channel)
run_test('Depolarizing channel', test_depolarizing_channel)
run_test('NoiseModel with circuit', test_noise_model_with_circuit)
run_test('Sampling with noise', test_sample_with_noise)

print()
print('--- TRANSPILER TESTS ---')
print()

def test_iid_elimination():
    qc = QuantumCircuit(3)
    qc.i(0)
    qc.h(1)
    qc.i(2)
    qc.cnot(0, 1)
    transpiled = transpile(qc, optimization_level=1)
    i_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'I')
    assert i_count == 0

def test_double_cx_cancel():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(0, 1)
    qc.h(1)
    transpiled = transpile(qc, optimization_level=2)
    cx_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'CNOT')
    assert cx_count == 0

def test_single_qubit_fusion():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.x(0)
    qc.z(0)
    qc.h(1)
    transpiled = transpile(qc, optimization_level=3)
    original_single = sum(1 for instr in qc.instructions if instr.gate.num_qubits == 1)
    transpiled_single = sum(1 for instr in transpiled.instructions if instr.gate.num_qubits == 1)
    assert transpiled_single < original_single

def test_transpiler_optimization_levels():
    for level in range(4):
        qc = QuantumCircuit(2)
        qc.i(0)
        qc.h(0)
        qc.x(0)
        qc.cnot(0, 1)
        qc.cnot(0, 1)
        qc.h(1)
        qc.i(1)
        transpiled = transpile(qc, optimization_level=level)
        original_result = qc.run().to_numpy()
        transpiled_result = transpiled.run().to_numpy()
        fidelity = np.abs(np.vdot(original_result, transpiled_result))**2
        assert fidelity > 0.9999

def test_parameterized_fusion_skip():
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 0)
    qc.cnot(0, 1)
    transpiled = transpile(qc, optimization_level=3)
    rx_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'RX')
    ry_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'RY')
    assert rx_count == 1
    assert ry_count == 1
    original_result = qc.run({'theta': 0.5, 'phi': 1.0}).to_numpy()
    transpiled_result = transpiled.run({'theta': 0.5, 'phi': 1.0}).to_numpy()
    fidelity = np.abs(np.vdot(original_result, transpiled_result))**2
    assert fidelity > 0.9999

run_test('Identity elimination', test_iid_elimination)
run_test('Double CNOT cancellation', test_double_cx_cancel)
run_test('Single qubit fusion', test_single_qubit_fusion)
run_test('Optimization levels 0-3', test_transpiler_optimization_levels)
run_test('Parameterized fusion skip', test_parameterized_fusion_skip)

print()
print('--- DISTRIBUTED TESTS ---')
print()

def test_distributed_context_no_mpi():
    ctx = DistributedContext.get(use_mpi=False)
    assert ctx.rank == 0
    assert ctx.size == 1
    assert not ctx.is_distributed

def test_distributed_state_vector():
    ctx = DistributedContext.get(use_mpi=False)
    dsv = DistributedStateVector(4, ctx=ctx)
    assert dsv.num_qubits == 4
    assert dsv.local_size == 16
    assert dsv.total_size == 16
    assert np.isclose(dsv.norm(), 1.0)

def test_distributed_single_qubit_gate():
    ctx = DistributedContext.get(use_mpi=False)
    dsv = DistributedStateVector(2, ctx=ctx)
    backend = get_backend()
    xp = backend.xp
    dsv.apply_single_qubit_gate(xp.asarray(H.matrix(), dtype=xp.complex128), 0)
    dsv.apply_single_qubit_gate(xp.asarray(X.matrix(), dtype=xp.complex128), 1)
    full_state = dsv.to_local_numpy()
    expected = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    fidelity = np.abs(np.vdot(full_state, expected))**2
    assert fidelity > 0.9999

def test_distributed_two_qubit_gate():
    ctx = DistributedContext.get(use_mpi=False)
    dsv = DistributedStateVector(2, ctx=ctx)
    backend = get_backend()
    xp = backend.xp
    dsv.apply_single_qubit_gate(xp.asarray(H.matrix(), dtype=xp.complex128), 0)
    dsv.apply_two_qubit_gate(xp.asarray(CNOT.matrix(), dtype=xp.complex128), [0, 1])
    full_state = dsv.to_local_numpy()
    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    fidelity = np.abs(np.vdot(full_state, expected))**2
    assert fidelity > 0.9999

def test_distributed_quantum_circuit():
    dc = DistributedQuantumCircuit(3, use_mpi=False)
    dc.h(0)
    dc.cnot(0, 1)
    dc.cnot(1, 2)
    dist_state = dc.run()
    full_state = dist_state.allgather_global()
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    regular_state = qc.run().to_numpy()
    fidelity = np.abs(np.vdot(full_state, regular_state))**2
    assert fidelity > 0.9999

def test_distributed_various_gates():
    ctx = DistributedContext.get(use_mpi=False)
    dsv = DistributedStateVector(2, ctx=ctx)
    backend = get_backend()
    xp = backend.xp
    dsv.apply_single_qubit_gate(xp.asarray(H.matrix(), dtype=xp.complex128), 0)
    dsv.apply_single_qubit_gate(xp.asarray(H.matrix(), dtype=xp.complex128), 1)
    dsv.apply_two_qubit_gate(xp.asarray(CZ.matrix(), dtype=xp.complex128), [0, 1])
    full_state = dsv.to_local_numpy()
    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=np.complex128)
    fidelity = np.abs(np.vdot(full_state, expected))**2
    assert fidelity > 0.9999

run_test('DistributedContext (no MPI)', test_distributed_context_no_mpi)
run_test('DistributedStateVector', test_distributed_state_vector)
run_test('Distributed single qubit gate', test_distributed_single_qubit_gate)
run_test('Distributed two qubit gate', test_distributed_two_qubit_gate)
run_test('DistributedQuantumCircuit', test_distributed_quantum_circuit)
run_test('Distributed CZ gate', test_distributed_various_gates)

print()
print('--- JIT COMPILATION TESTS ---')
print()

def test_jit_compiled_circuit_creation():
    JITCompiler.clear_cache()
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 1)
    qc.h(0)
    qc.x(1)
    compiled = JITCompiledCircuit(qc)
    assert compiled._num_qubits == 2
    assert len(compiled._param_order) == 2
    assert 'theta' in compiled._param_order
    assert 'phi' in compiled._param_order

def test_jit_correctness():
    JITCompiler.clear_cache()
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 1)
    qc.h(0)
    qc.x(1)
    params = {'theta': 0.5, 'phi': 1.2}
    
    jit_result = run_jit(qc, params, use_cache=False)
    regular_result = qc.run(params).to_numpy()
    jit_flat = np.asarray(jit_result).flatten()
    regular_flat = np.asarray(regular_result).flatten()
    fidelity = np.abs(np.vdot(jit_flat, regular_flat))**2
    assert fidelity > 0.9999

def test_jit_cache():
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 1)
    qc.h(0)
    qc.x(1)
    JITCompiler.clear_cache()
    compiled1 = JITCompiler.compile(qc, use_cache=True)
    compiled2 = JITCompiler.compile(qc, use_cache=True)
    assert compiled1 is compiled2

def test_jit_multi_parameter():
    JITCompiler.clear_cache()
    qc = QuantumCircuit(3)
    qc.rx(Parameter(0.0, 't1'), 0)
    qc.rz(Parameter(0.0, 't2'), 1)
    qc.ry(Parameter(0.0, 't3'), 2)
    qc.h(0)
    qc.s(1)
    qc.t(2)
    params = {'t1': 0.1, 't2': 0.2, 't3': 0.3}
    jit_result = run_jit(qc, params, use_cache=False)
    regular_result = qc.run(params).to_numpy()
    jit_flat = np.asarray(jit_result).flatten()
    regular_flat = np.asarray(regular_result).flatten()
    fidelity = np.abs(np.vdot(jit_flat, regular_flat))**2
    assert fidelity > 0.9999

def test_jit_gradient():
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 1)
    qc.h(0)
    qc.z(1)
    obs = PauliOp([(1.0, 'ZZ')])
    params = {'theta': 0.5, 'phi': 1.2}
    from quantum_sim.jit import JITGradient
    jit_grad = JITGradient(qc, obs)
    gradients = jit_grad.gradient(params)
    assert len(gradients) == 2
    assert all(isinstance(g, float) for g in gradients)

def test_jit_cuda_code_generation():
    qc = QuantumCircuit(2)
    qc.rx(Parameter(0.0, 'theta'), 0)
    qc.ry(Parameter(0.0, 'phi'), 1)
    qc.h(0)
    qc.x(1)
    compiled = JITCompiledCircuit(qc)
    code, kernel_name = compiled._generate_cuda_code()
    assert 'quantum_circuit_' in kernel_name
    extern_str = 'extern "C" __global__'
    assert extern_str in code
    assert 'cos' in code
    assert 'sin' in code
    assert 'param_idx' in code

run_test('JITCompiledCircuit creation', test_jit_compiled_circuit_creation)
run_test('JIT correctness (Fidelity=1.0)', test_jit_correctness)
run_test('JIT caching', test_jit_cache)
run_test('JIT multi-parameter', test_jit_multi_parameter)
run_test('JIT gradient calculation', test_jit_gradient)
run_test('JIT CUDA code generation', test_jit_cuda_code_generation)

print()
print('=' * 70)
print(f'TEST SUMMARY: {passed} passed, {failed} failed')
print('=' * 70)

if failed > 0:
    sys.exit(1)
else:
    print('\nAll tests passed successfully!')
    sys.exit(0)
