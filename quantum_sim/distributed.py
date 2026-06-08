from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend, Backend


class DistributedContext:
    _instance: Optional["DistributedContext"] = None

    def __init__(self, use_mpi: bool = True):
        self._use_mpi = use_mpi
        self._mpi_available = False
        self._comm = None
        self._rank = 0
        self._size = 1
        self._local_gpu_id = 0

        if use_mpi:
            try:
                from mpi4py import MPI

                self._comm = MPI.COMM_WORLD
                self._rank = self._comm.Get_rank()
                self._size = self._comm.Get_size()
                self._mpi_available = True

                if self._size > 1:
                    backend = get_backend()
                    if backend.is_gpu():
                        num_gpus = backend.get_num_gpus()
                        if num_gpus > 0:
                            self._local_gpu_id = self._rank % num_gpus
                            backend.set_device(self._local_gpu_id)

            except ImportError:
                self._use_mpi = False
                self._mpi_available = False

    @classmethod
    def get(cls, use_mpi: bool = True) -> "DistributedContext":
        if cls._instance is None:
            cls._instance = cls(use_mpi)
        return cls._instance

    @property
    def use_mpi(self) -> bool:
        return self._use_mpi and self._mpi_available

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    @property
    def local_gpu_id(self) -> int:
        return self._local_gpu_id

    @property
    def is_distributed(self) -> bool:
        return self._use_mpi and self._mpi_available and self._size > 1

    def barrier(self) -> None:
        if self.is_distributed:
            self._comm.Barrier()

    def bcast(self, data: Any, root: int = 0) -> Any:
        if not self.is_distributed:
            return data
        return self._comm.bcast(data, root=root)

    def allreduce_sum(self, data: Any) -> Any:
        if not self.is_distributed:
            return data

        if hasattr(data, '__len__'):
            data_array = np.asarray(data, dtype=np.float64)
            result = np.zeros_like(data_array)
            from mpi4py import MPI
            self._comm.Allreduce(data_array, result, op=MPI.SUM)
            return result
        else:
            data_val = float(data)
            from mpi4py import MPI
            result = self._comm.allreduce(data_val, op=MPI.SUM)
            return result

    def allgather(self, data: Any) -> list[Any]:
        if not self.is_distributed:
            return [data]
        return self._comm.allgather(data)

    def finalize(self) -> None:
        if self.is_distributed:
            self._comm.Barrier()


class DistributedStateVector:
    def __init__(
        self,
        num_qubits: int,
        ctx: Optional[DistributedContext] = None,
    ):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self._num_qubits = num_qubits
        self._ctx = ctx or DistributedContext.get()

        total_size = 2**num_qubits
        self._total_size = total_size

        if self._ctx.is_distributed:
            size = self._ctx.size
            rank = self._ctx.rank

            if total_size < size:
                raise ValueError(
                    f"State vector size ({total_size}) is smaller than number of processes ({size})"
                )

            self._local_size = total_size // size
            self._global_start = rank * self._local_size
            self._global_end = self._global_start + self._local_size

            if rank == size - 1:
                self._local_size += total_size % size
                self._global_end = total_size
        else:
            self._local_size = total_size
            self._global_start = 0
            self._global_end = total_size

        self._shape = tuple([2] * num_qubits)
        backend = get_backend()
        xp = backend.xp

        self._data = xp.zeros(self._local_size, dtype=xp.complex128)
        if self._global_start <= 0 < self._global_end:
            self._data[0 - self._global_start] = 1.0

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def local_size(self) -> int:
        return self._local_size

    @property
    def total_size(self) -> int:
        return self._total_size

    @property
    def global_start(self) -> int:
        return self._global_start

    @property
    def global_end(self) -> int:
        return self._global_end

    @property
    def is_distributed(self) -> bool:
        return self._ctx.is_distributed

    def local_to_global(self, local_idx: int) -> int:
        return self._global_start + local_idx

    def global_to_local(self, global_idx: int) -> Optional[int]:
        if self._global_start <= global_idx < self._global_end:
            return global_idx - self._global_start
        return None

    def apply_single_qubit_gate(
        self,
        gate_matrix: Any,
        qubit: int,
    ) -> None:
        backend = get_backend()
        xp = backend.xp

        if not self._ctx.is_distributed:
            data_reshaped = xp.reshape(self._data, self._shape, order='F')
            result = xp.tensordot(gate_matrix, data_reshaped, axes=([1], [qubit]))
            if qubit != 0:
                result = xp.moveaxis(result, 0, qubit)
            self._data = xp.reshape(result, -1, order='F')
            return

        n = self._num_qubits
        stride = 2**qubit
        block_size = 2 * stride

        data = self._data
        new_data = xp.zeros_like(data, dtype=xp.complex128)

        g00, g01 = gate_matrix[0, 0], gate_matrix[0, 1]
        g10, g11 = gate_matrix[1, 0], gate_matrix[1, 1]

        for local_i in range(self._local_size):
            global_i = self.local_to_global(local_i)
            bit = (global_i >> qubit) & 1
            other_global = global_i ^ (1 << qubit)
            other_local = self.global_to_local(other_global)

            if other_local is not None:
                if bit == 0:
                    new_data[local_i] = g00 * data[local_i] + g01 * data[other_local]
                else:
                    new_data[local_i] = g10 * data[other_local] + g11 * data[local_i]
            else:
                other_rank = other_global // self._local_size
                if other_rank < self._ctx.size:
                    from mpi4py import MPI

                    if bit == 0:
                        recv_buf = xp.zeros(1, dtype=xp.complex128)
                        send_buf = data[local_i].copy()
                        self._ctx._comm.Sendrecv(
                            send_buf, dest=other_rank, sendtag=0,
                            recvbuf=recv_buf, source=other_rank, recvtag=0
                        )
                        new_data[local_i] = g00 * data[local_i] + g01 * recv_buf[0]
                    else:
                        recv_buf = xp.zeros(1, dtype=xp.complex128)
                        send_buf = data[local_i].copy()
                        self._ctx._comm.Sendrecv(
                            send_buf, dest=other_rank, sendtag=0,
                            recvbuf=recv_buf, source=other_rank, recvtag=0
                        )
                        new_data[local_i] = g10 * recv_buf[0] + g11 * data[local_i]
                else:
                    if bit == 0:
                        new_data[local_i] = g00 * data[local_i]
                    else:
                        new_data[local_i] = g11 * data[local_i]

        self._data = new_data

    def apply_two_qubit_gate(
        self,
        gate_matrix: Any,
        qubits: list[int],
    ) -> None:
        backend = get_backend()
        xp = backend.xp

        if not self._ctx.is_distributed:
            gate_shape = (2, 2, 2, 2)
            gate_tensor = gate_matrix.reshape(gate_shape)
            gate_in_axes = [2, 3]
            data_reshaped = xp.reshape(self._data, self._shape, order='F')
            contracted = xp.tensordot(gate_tensor, data_reshaped, axes=(gate_in_axes, qubits))

            new_axes = list(range(self._num_qubits))
            remaining = [i for i in range(self._num_qubits) if i not in qubits]
            for i, q in enumerate(qubits):
                new_axes[q] = i
            for i, r in enumerate(remaining):
                new_axes[r] = 2 + i

            result = xp.transpose(contracted, new_axes)
            self._data = xp.reshape(result, -1, order='F')
            return

        q0, q1 = qubits
        n = self._num_qubits

        new_data = xp.zeros_like(self._data, dtype=xp.complex128)

        for local_i in range(self._local_size):
            global_i = self.local_to_global(local_i)
            b0 = (global_i >> q0) & 1
            b1 = (global_i >> q1) & 1

            row_idx = b0 * 2 + b1
            acc = 0.0 + 0.0j

            for col_b0 in [0, 1]:
                for col_b1 in [0, 1]:
                    col_idx = col_b0 * 2 + col_b1
                    g = gate_matrix[row_idx, col_idx]
                    if abs(g) < 1e-15:
                        continue

                    other_global = global_i
                    if b0 != col_b0:
                        other_global ^= (1 << q0)
                    if b1 != col_b1:
                        other_global ^= (1 << q1)

                    other_local = self.global_to_local(other_global)
                    if other_local is not None:
                        acc += g * self._data[other_local]
                    else:
                        from mpi4py import MPI

                        other_rank = other_global // self._local_size
                        if other_rank < self._ctx.size:
                            recv_buf = xp.zeros(1, dtype=xp.complex128)
                            send_buf = xp.zeros(1, dtype=xp.complex128)
                            tag = (q0 * 100 + q1) % 32768

                            self._ctx._comm.Sendrecv(
                                send_buf, dest=other_rank, sendtag=tag,
                                recvbuf=recv_buf, source=other_rank, recvtag=tag
                            )
                            acc += g * recv_buf[0]

            new_data[local_i] = acc

        self._data = new_data

    def norm(self) -> float:
        backend = get_backend()
        xp = backend.xp

        local_norm_sq = float(xp.sum(xp.abs(self._data) ** 2))
        total_norm_sq = self._ctx.allreduce_sum(local_norm_sq)
        return float(np.sqrt(total_norm_sq))

    def normalize(self) -> None:
        norm = self.norm()
        if norm > 0:
            backend = get_backend()
            xp = backend.xp
            self._data /= norm

    def expectation_value(self, observable: Any) -> float:
        backend = get_backend()
        xp = backend.xp

        obs_matrix = observable.to_matrix()
        data_conj = xp.conj(self._data)

        if not self._ctx.is_distributed:
            full_data = self._data
            result = float(xp.real(xp.dot(data_conj, obs_matrix @ full_data)))
            return result

        raise NotImplementedError(
            "Distributed expectation value requires full state access"
        )

    def to_local_numpy(self) -> np.ndarray:
        backend = get_backend()
        return backend.to_numpy(self._data)

    def gather_global(self, root: int = 0) -> Optional[np.ndarray]:
        if not self._ctx.is_distributed:
            return self.to_local_numpy()

        local_data = self.to_local_numpy()
        all_local = self._ctx.allgather(local_data)

        if self._ctx.rank == root:
            return np.concatenate(all_local)
        return None

    def allgather_global(self) -> np.ndarray:
        if not self._ctx.is_distributed:
            return self.to_local_numpy()

        local_data = self.to_local_numpy()
        all_local = self._ctx.allgather(local_data)
        return np.concatenate(all_local)


class DistributedQuantumCircuit:
    def __init__(self, num_qubits: int, use_mpi: bool = True):
        self._num_qubits = num_qubits
        self._ctx = DistributedContext.get(use_mpi=use_mpi)
        self._instructions = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def ctx(self) -> DistributedContext:
        return self._ctx

    def h(self, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import H
        self._instructions.append((H, [qubit], {}))
        return self

    def x(self, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import X
        self._instructions.append((X, [qubit], {}))
        return self

    def y(self, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import Y
        self._instructions.append((Y, [qubit], {}))
        return self

    def z(self, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import Z
        self._instructions.append((Z, [qubit], {}))
        return self

    def rx(self, theta: float, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import RX
        self._instructions.append((RX, [qubit], {"theta": theta}))
        return self

    def ry(self, theta: float, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import RY
        self._instructions.append((RY, [qubit], {"theta": theta}))
        return self

    def rz(self, theta: float, qubit: int) -> "DistributedQuantumCircuit":
        from .gates import RZ
        self._instructions.append((RZ, [qubit], {"theta": theta}))
        return self

    def cnot(self, control: int, target: int) -> "DistributedQuantumCircuit":
        from .gates import CNOT
        self._instructions.append((CNOT, [control, target], {}))
        return self

    def cz(self, control: int, target: int) -> "DistributedQuantumCircuit":
        from .gates import CZ
        self._instructions.append((CZ, [control, target], {}))
        return self

    def run(self) -> DistributedStateVector:
        state = DistributedStateVector(self._num_qubits, ctx=self._ctx)
        backend = get_backend()
        xp = backend.xp

        for gate, qubits, params in self._instructions:
            gate_matrix = gate.matrix(params if params else None)
            gate_matrix = xp.asarray(gate_matrix, dtype=xp.complex128)

            if gate.num_qubits == 1:
                state.apply_single_qubit_gate(gate_matrix, qubits[0])
            elif gate.num_qubits == 2:
                state.apply_two_qubit_gate(gate_matrix, qubits)
            else:
                raise NotImplementedError(
                    f"Distributed gate application not implemented for {gate.num_qubits} qubits"
                )

        self._ctx.barrier()
        return state

    def __repr__(self) -> str:
        return (
            f"DistributedQuantumCircuit({self._num_qubits} qubits, "
            f"{len(self._instructions)} gates, rank={self._ctx.rank}, size={self._ctx.size})"
        )


def run_distributed_example(num_qubits: int = 5) -> None:
    ctx = DistributedContext.get(use_mpi=True)

    if ctx.rank == 0:
        print(f"Running distributed example with {num_qubits} qubits")
        print(f"MPI size: {ctx.size}, MPI rank: {ctx.rank}")

    circuit = DistributedQuantumCircuit(num_qubits, use_mpi=True)

    circuit.h(0)
    for i in range(min(num_qubits - 1, 10)):
        circuit.cnot(i, i + 1)

    state = circuit.run()

    norm = state.norm()
    if ctx.rank == 0:
        print(f"State norm: {norm:.10f}")
        print(f"Local size per rank: {state.local_size}")
        print(f"Total size: {state.total_size}")

    ctx.barrier()
    full_state = state.allgather_global()

    if ctx.rank == 0:
        print(f"Gathered state norm: {np.sum(np.abs(full_state)**2):.10f}")
        print("Distributed example completed successfully!")
