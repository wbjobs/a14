from __future__ import annotations

from typing import Any, Optional, Union

import hashlib
import numpy as np

from .backend import get_backend
from .circuit import QuantumCircuit, CircuitInstruction
from .autograd import Parameter


class JITCompiledCircuit:
    def __init__(
        self,
        circuit: QuantumCircuit,
        use_constant_folding: bool = True,
    ):
        self._circuit = circuit
        self._num_qubits = circuit.num_qubits
        self._use_constant_folding = use_constant_folding
        self._compiled_kernel = None
        self._kernel_name = None
        self._param_order: list[str] = []
        self._constant_gates: list[tuple[Any, list[int]]] = []
        self._parameterized_gates: list[tuple[Any, list[int], str, str]] = []
        self._is_compiled = False
        self._backend = get_backend()
        self._xp = self._backend.xp
        self._cache_key: Optional[str] = None

        self._analyze_circuit()

    def _analyze_circuit(self) -> None:
        for instr in self._circuit.instructions:
            gate = instr.gate
            qubits = instr.qubits

            if gate.is_parameterized and instr.param_names:
                for pname in instr.param_names:
                    param_obj = instr.params[pname]
                    if isinstance(param_obj, Parameter):
                        if param_obj.name not in self._param_order:
                            self._param_order.append(param_obj.name)
                        self._parameterized_gates.append(
                            (gate, qubits, pname, param_obj.name)
                        )
            else:
                self._constant_gates.append((gate, qubits))

    def _generate_cache_key(self) -> str:
        key_parts = [
            f"num_qubits={self._num_qubits}",
            f"constant_folding={self._use_constant_folding}",
        ]

        for gate, qubits in self._constant_gates:
            key_parts.append(f"const:{gate.name}:{qubits}")

        for gate, qubits, pname, param_name in self._parameterized_gates:
            key_parts.append(f"param:{gate.name}:{qubits}:{pname}:{param_name}")

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _generate_cuda_code(self) -> tuple[str, str]:
        kernel_name = f"quantum_circuit_{self._generate_cache_key()[:8]}"
        self._kernel_name = kernel_name

        code = f"""
#include <cupy/complex.cuh>
typedef complex<double> cdouble;

extern "C" __global__
void {kernel_name}(
    const cdouble* __restrict__ input,
    cdouble* __restrict__ output,
    const double* __restrict__ params,
    int num_qubits,
    int batch_size)
{{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_size = 1ULL << num_qubits;

    if (idx >= total_size) return;

    cdouble val = input[idx];
    int param_idx = 0;
"""

        for gate, qubits, pname, param_name in self._parameterized_gates:
            if gate.name == "RX":
                code += self._gen_rx_code(qubits)
            elif gate.name == "RY":
                code += self._gen_ry_code(qubits)
            elif gate.name == "RZ":
                code += self._gen_rz_code(qubits)

        for gate, qubits in self._constant_gates:
            code += self._gen_constant_gate_code(gate, qubits)

        code += f"""
    output[idx] = val;
}}
"""
        return code, kernel_name

    def _gen_rx_code(self, qubits: list[int]) -> str:
        q = qubits[0]
        return f"""
    {{
        double theta = params[param_idx++];
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        unsigned long long mask = 1ULL << {q};
        unsigned long long other_idx = idx ^ mask;

        if (idx & mask) {{
            cdouble v0 = val;
            cdouble v1 = input[other_idx];
            val = cdouble(0, -s) * v1 + cdouble(c, 0) * v0;
        }} else {{
            cdouble v0 = val;
            cdouble v1 = input[other_idx];
            val = cdouble(c, 0) * v0 + cdouble(0, -s) * v1;
        }}
    }}
"""

    def _gen_ry_code(self, qubits: list[int]) -> str:
        q = qubits[0]
        return f"""
    {{
        double theta = params[param_idx++];
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        unsigned long long mask = 1ULL << {q};
        unsigned long long other_idx = idx ^ mask;

        if (idx & mask) {{
            cdouble v0 = val;
            cdouble v1 = input[other_idx];
            val = cdouble(s, 0) * v1 + cdouble(c, 0) * v0;
        }} else {{
            cdouble v0 = val;
            cdouble v1 = input[other_idx];
            val = cdouble(c, 0) * v0 + cdouble(-s, 0) * v1;
        }}
    }}
"""

    def _gen_rz_code(self, qubits: list[int]) -> str:
        q = qubits[0]
        return f"""
    {{
        double theta = params[param_idx++];
        double c = cos(theta / 2.0);
        double s = sin(theta / 2.0);
        unsigned long long mask = 1ULL << {q};

        if (idx & mask) {{
            val *= cdouble(c, -s);
        }} else {{
            val *= cdouble(c, s);
        }}
    }}
"""

    def _gen_constant_gate_code(self, gate: Any, qubits: list[int]) -> str:
        if gate.num_qubits == 1:
            q = qubits[0]
            matrix = gate.matrix()
            return f"""
    {{
        unsigned long long mask = 1ULL << {q};
        unsigned long long other_idx = idx ^ mask;
        cdouble g00 = cdouble({matrix[0,0].real}, {matrix[0,0].imag});
        cdouble g01 = cdouble({matrix[0,1].real}, {matrix[0,1].imag});
        cdouble g10 = cdouble({matrix[1,0].real}, {matrix[1,0].imag});
        cdouble g11 = cdouble({matrix[1,1].real}, {matrix[1,1].imag});

        if (idx & mask) {{
            cdouble v0 = input[other_idx];
            cdouble v1 = val;
            val = g10 * v0 + g11 * v1;
        }} else {{
            cdouble v0 = val;
            cdouble v1 = input[other_idx];
            val = g00 * v0 + g01 * v1;
        }}
    }}
"""
        return ""

    def compile(self) -> "JITCompiledCircuit":
        if self._is_compiled:
            return self

        if not self._backend.is_gpu():
            self._is_compiled = True
            return self

        try:
            code, kernel_name = self._generate_cuda_code()
            from cupy import RawModule

            module = RawModule(code=code, options=("-O3", "--use_fast_math"))
            self._compiled_kernel = module.get_function(kernel_name)
            self._is_compiled = True

        except Exception as e:
            print(f"JIT compilation failed, falling back to regular execution: {e}")
            self._is_compiled = True

        return self

    def run(
        self,
        param_values: Optional[dict[str, Any]] = None,
        input_state: Optional[Any] = None,
    ) -> Any:
        if not self._is_compiled:
            self.compile()

        if not self._backend.is_gpu() or self._compiled_kernel is None:
            result = self._circuit.run(param_values)
            return self._xp.reshape(result._data, -1, order='F')

        xp = self._xp

        if input_state is None:
            total_size = 2**self._num_qubits
            input_data = xp.zeros(total_size, dtype=xp.complex128)
            input_data[0] = 1.0
        else:
            input_data = xp.asarray(input_state, dtype=xp.complex128)
            if input_data.ndim > 1:
                input_data = xp.reshape(input_data, -1, order='F')

        param_array = xp.zeros(len(self._param_order), dtype=xp.float64)
        resolved = self._circuit._resolve_params(param_values)
        for i, pname in enumerate(self._param_order):
            param_array[i] = resolved.get(pname, 0.0)

        output_data = xp.zeros_like(input_data, dtype=xp.complex128)

        total_size = 2**self._num_qubits
        block_size = 256
        grid_size = (total_size + block_size - 1) // block_size

        self._compiled_kernel(
            (grid_size,),
            (block_size,),
            (input_data, output_data, param_array, np.int32(self._num_qubits), np.int32(1)),
        )

        self._backend.synchronize()

        return output_data

    def __call__(
        self,
        param_values: Optional[dict[str, Any]] = None,
        input_state: Optional[Any] = None,
    ) -> Any:
        return self.run(param_values, input_state)


class JITCompiler:
    _cache: dict[str, JITCompiledCircuit] = {}

    @classmethod
    def compile(
        cls,
        circuit: QuantumCircuit,
        use_cache: bool = True,
        **kwargs,
    ) -> JITCompiledCircuit:
        cache_key = cls._get_cache_key(circuit, **kwargs)
        
        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key]

        compiled = JITCompiledCircuit(circuit, **kwargs)
        compiled.compile()

        if use_cache:
            cls._cache[cache_key] = compiled

        return compiled

    @staticmethod
    def _get_cache_key(circuit: QuantumCircuit, **kwargs) -> str:
        key_parts = [f"num_qubits={circuit.num_qubits}"]
        for instr in circuit.instructions:
            key_parts.append(
                f"{instr.gate.name}:{instr.qubits}:{instr.param_names}"
            )
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()


def jit(
    circuit: QuantumCircuit,
    **kwargs,
) -> JITCompiledCircuit:
    return JITCompiler.compile(circuit, **kwargs)


def run_jit(
    circuit: QuantumCircuit,
    param_values: Optional[dict[str, Any]] = None,
    **kwargs,
) -> Any:
    compiled = jit(circuit, **kwargs)
    return compiled.run(param_values)


class JITGradient:
    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: Any,
    ):
        self._circuit = circuit
        self._observable = observable
        self._forward = JITCompiledCircuit(circuit)
        self._backward = JITCompiledCircuit(circuit)
        self._backend = get_backend()
        self._xp = self._backend.xp

    def compile(self) -> "JITGradient":
        self._forward.compile()
        self._backward.compile()
        return self

    def gradient(
        self,
        param_values: Optional[dict[str, Any]] = None,
        shift: float = np.pi / 2,
    ) -> list[float]:
        params_list = list(self._circuit.parameters.values())

        gradients = []
        resolved = self._circuit._resolve_params(param_values)

        for i, param in enumerate(params_list):
            param_name = param.name

            params_plus = dict(resolved)
            params_plus[param_name] = resolved[param_name] + shift
            plus_state = self._forward.run(params_plus)

            params_minus = dict(resolved)
            params_minus[param_name] = resolved[param_name] - shift
            minus_state = self._backward.run(params_minus)

            backend = get_backend()
            xp = backend.xp

            obs_matrix = xp.asarray(self._observable.to_matrix(), dtype=xp.complex128)

            plus_flat = xp.asarray(plus_state).flatten()
            minus_flat = xp.asarray(minus_state).flatten()

            plus_exp = float(xp.real(xp.dot(xp.conj(plus_flat), obs_matrix @ plus_flat)))
            minus_exp = float(xp.real(xp.dot(xp.conj(minus_flat), obs_matrix @ minus_flat)))

            denom = 2 * np.sin(shift)
            if abs(denom) < 1e-10:
                denom = 1e-10
            grad = (plus_exp - minus_exp) / denom

            gradients.append(grad)

        return gradients
