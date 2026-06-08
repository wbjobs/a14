from quantum_sim import *
import numpy as np

qc = QuantumCircuit(2)
qc.rx(Parameter(0.0, 'theta'), 0)
qc.ry(Parameter(0.0, 'phi'), 1)
qc.cnot(0, 1)

params = {'theta': 0.5, 'phi': 1.2}

compiled = jit(qc, use_cache=False)
result_compiled = compiled.run(params)
print(f'compiled.run result: {np.asarray(result_compiled).flatten()}')

# Test _generate_cuda_code
code, kernel_name = compiled._generate_cuda_code()
print(f'Kernel name: {kernel_name}')
print(f'Has RX: {"RX" in code}')
print(f'Has RY: {"RY" in code}')
extern_str = 'extern "C" __global__'
print(f'Has extern: {extern_str in code}')
print()
print('Code:')
print(code)
