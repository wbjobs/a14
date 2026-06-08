from __future__ import annotations

import numpy as np
import pytest

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


class TestNoiseModel:
    def test_bit_flip_channel(self):
        noise = BitFlip(p=0.1)
        assert noise.p == 0.1
        assert noise.num_qubits == 1
        
        kraus_ops = noise.kraus_operators()
        assert len(kraus_ops) == 2
        
        state = StateVector(1)
        noise.apply(state, 0)
        
        probs = state.probability(0)
        assert np.isclose(np.sum(probs), 1.0)

    def test_phase_damping_channel(self):
        noise = PhaseDamping(gamma=0.2)
        assert noise.gamma == 0.2
        
        kraus_ops = noise.kraus_operators()
        assert len(kraus_ops) == 2

    def test_depolarizing_channel(self):
        noise = Depolarizing(p=0.3)
        assert noise.p == 0.3
        
        kraus_ops = noise.kraus_operators()
        assert len(kraus_ops) == 4

    def test_noise_model_with_circuit(self):
        noise_model = NoiseModel()
        noise_model.add_gate_noise(CNOT, BitFlip(0.05))
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        
        state = qc.run(noise_model=noise_model)
        probs = state.probability()
        assert np.isclose(np.sum(probs), 1.0)

    def test_sample_with_noise(self):
        noise_model = NoiseModel()
        noise_model.add_qubit_noise(0, BitFlip(0.5))
        
        qc = QuantumCircuit(1)
        
        samples = qc.sample(shots=1000, noise_model=noise_model)
        assert len(samples) == 1000

    def test_gradient_with_noise(self):
        noise_model = NoiseModel()
        noise_model.add_gate_noise(RX, Depolarizing(0.01))
        
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.cnot(0, 1)
        
        obs = PauliOp([(1.0, 'ZZ')])
        grads = qc.gradient('theta', obs, noise_model=noise_model, shots=100)
        assert len(grads) == 1


class TestTranspiler:
    def test_iid_elimination(self):
        qc = QuantumCircuit(3)
        qc.i(0)
        qc.h(1)
        qc.i(2)
        qc.cnot(0, 1)
        
        transpiled = transpile(qc, optimization_level=1)
        i_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'I')
        assert i_count == 0

    def test_double_cx_cancel(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        qc.cnot(0, 1)
        qc.h(1)
        
        transpiled = transpile(qc, optimization_level=2)
        cx_count = sum(1 for instr in transpiled.instructions if instr.gate.name == 'CNOT')
        assert cx_count == 0

    def test_single_qubit_fusion(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(0)
        qc.z(0)
        qc.h(1)
        
        transpiled = transpile(qc, optimization_level=3)
        
        original_single = sum(1 for instr in qc.instructions if instr.gate.num_qubits == 1)
        transpiled_single = sum(1 for instr in transpiled.instructions if instr.gate.num_qubits == 1)
        assert transpiled_single < original_single

    def test_transpiler_optimization_levels(self):
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

    def test_parameterized_fusion_skip(self):
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


class TestDistributed:
    def test_distributed_context_no_mpi(self):
        ctx = DistributedContext.get(use_mpi=False)
        assert ctx.rank == 0
        assert ctx.size == 1
        assert not ctx.is_distributed

    def test_distributed_state_vector(self):
        ctx = DistributedContext.get(use_mpi=False)
        dsv = DistributedStateVector(4, ctx=ctx)
        
        assert dsv.num_qubits == 4
        assert dsv.local_size == 16
        assert dsv.total_size == 16
        assert np.isclose(dsv.norm(), 1.0)

    def test_distributed_single_qubit_gate(self):
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

    def test_distributed_two_qubit_gate(self):
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

    def test_distributed_quantum_circuit(self):
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

    def test_distributed_various_gates(self):
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


class TestJIT:
    def test_jit_compiled_circuit_creation(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.ry(Parameter(0.0, 'phi'), 1)
        qc.cnot(0, 1)
        
        compiled = JITCompiledCircuit(qc)
        assert compiled._num_qubits == 2
        assert len(compiled._param_order) == 2
        assert 'theta' in compiled._param_order
        assert 'phi' in compiled._param_order

    def test_jit_correctness(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.ry(Parameter(0.0, 'phi'), 1)
        qc.cnot(0, 1)
        
        params = {'theta': 0.5, 'phi': 1.2}
        
        jit_result = run_jit(qc, params, use_cache=False)
        regular_result = qc.run(params).to_numpy()
        
        jit_flat = np.asarray(jit_result).flatten()
        regular_flat = np.asarray(regular_result).flatten()
        
        fidelity = np.abs(np.vdot(jit_flat, regular_flat))**2
        assert fidelity > 0.9999

    def test_jit_cache(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.ry(Parameter(0.0, 'phi'), 1)
        qc.cnot(0, 1)
        
        JITCompiler.clear_cache()
        
        compiled1 = JITCompiler.compile(qc, use_cache=True)
        compiled2 = JITCompiler.compile(qc, use_cache=True)
        
        assert compiled1 is compiled2

    def test_jit_multi_parameter(self):
        qc = QuantumCircuit(3)
        qc.rx(Parameter(0.0, 't1'), 0)
        qc.rz(Parameter(0.0, 't2'), 1)
        qc.ry(Parameter(0.0, 't3'), 2)
        qc.cnot(0, 1)
        qc.cnot(1, 2)
        
        params = {'t1': 0.1, 't2': 0.2, 't3': 0.3}
        
        jit_result = run_jit(qc, params, use_cache=False)
        regular_result = qc.run(params).to_numpy()
        
        jit_flat = np.asarray(jit_result).flatten()
        regular_flat = np.asarray(regular_result).flatten()
        
        fidelity = np.abs(np.vdot(jit_flat, regular_flat))**2
        assert fidelity > 0.9999

    def test_jit_gradient(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.ry(Parameter(0.0, 'phi'), 1)
        qc.cnot(0, 1)
        
        obs = PauliOp([(1.0, 'ZZ')])
        params = {'theta': 0.5, 'phi': 1.2}
        
        from quantum_sim.jit import JITGradient
        jit_grad = JITGradient(qc, obs)
        gradients = jit_grad.gradient(params)
        
        assert len(gradients) == 2
        assert all(isinstance(g, float) for g in gradients)

    def test_jit_cuda_code_generation(self):
        qc = QuantumCircuit(2)
        qc.rx(Parameter(0.0, 'theta'), 0)
        qc.ry(Parameter(0.0, 'phi'), 1)
        qc.cnot(0, 1)
        
        compiled = JITCompiledCircuit(qc)
        code, kernel_name = compiled._generate_cuda_code()
        
        assert 'quantum_circuit_' in kernel_name
        assert 'extern "C" __global__' in code
        assert 'RX' in code or 'RY' in code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
