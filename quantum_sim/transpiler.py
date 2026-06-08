from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from .backend import get_backend
from .circuit import QuantumCircuit, CircuitInstruction
from .gates import Gate, H, X, Y, Z, I, CNOT, CZ, SWAP
from .autograd import Parameter


class OptimizationPass(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass

    def __repr__(self) -> str:
        return f"{self.name}()"


class SingleQubitFusion(OptimizationPass):
    def __init__(self):
        super().__init__("SingleQubitFusion")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        backend = get_backend()
        xp = backend.xp

        new_circuit = QuantumCircuit(circuit.num_qubits)
        num_qubits = circuit.num_qubits

        qubit_gates: list[list[CircuitInstruction]] = [[] for _ in range(num_qubits)]
        blocked_qubits: set[int] = set()

        def flush_qubit(qubit: int) -> None:
            if not qubit_gates[qubit]:
                return

            if len(qubit_gates[qubit]) == 1:
                instr = qubit_gates[qubit][0]
                new_circuit._instructions.append(instr)
                qubit_gates[qubit] = []
                return

            has_params = False
            for instr in qubit_gates[qubit]:
                if instr.param_names:
                    has_params = True
                    break

            if has_params:
                for instr in qubit_gates[qubit]:
                    new_circuit._instructions.append(instr)
                qubit_gates[qubit] = []
                return

            combined_matrix = xp.eye(2, dtype=xp.complex128)
            for instr in qubit_gates[qubit]:
                gate_matrix = instr.gate.matrix()
                combined_matrix = gate_matrix @ combined_matrix

            qubit_gates[qubit] = []

            class _FusedGate(Gate):
                def __init__(self, matrix):
                    super().__init__("FusedSingleQubit", num_qubits=1)
                    self._matrix = matrix

                def matrix(self, params=None):
                    return self._matrix

            fused_gate = _FusedGate(combined_matrix)
            new_instr = CircuitInstruction(fused_gate, [qubit])
            new_circuit._instructions.append(new_instr)

        for instr in circuit.instructions:
            if instr.gate.num_qubits == 1:
                qubit = instr.qubits[0]
                if qubit in blocked_qubits:
                    flush_qubit(qubit)
                    blocked_qubits.discard(qubit)
                qubit_gates[qubit].append(instr)
            else:
                for q in instr.qubits:
                    flush_qubit(q)
                    blocked_qubits.discard(q)
                new_circuit._instructions.append(instr)
                for q in instr.qubits:
                    blocked_qubits.add(q)

        for qubit in range(num_qubits):
            flush_qubit(qubit)

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class CXCancel(OptimizationPass):
    def __init__(self):
        super().__init__("CXCancel")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)

        last_cx: dict[tuple[int, int], int] = {}

        for instr in circuit.instructions:
            if instr.gate.name == "CNOT" and instr.gate.num_qubits == 2:
                cnot_key = (instr.qubits[0], instr.qubits[1])

                if cnot_key in last_cx and last_cx[cnot_key] == len(new_circuit._instructions) - 1:
                    new_circuit._instructions.pop()
                    del last_cx[cnot_key]
                    continue

                last_cx.clear()
                last_cx[cnot_key] = len(new_circuit._instructions)
                new_circuit._instructions.append(instr)
            else:
                last_cx.clear()
                new_circuit._instructions.append(instr)

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class DoubleCXCancel(OptimizationPass):
    def __init__(self):
        super().__init__("DoubleCXCancel")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)
        instructions = circuit.instructions
        n = len(instructions)
        i = 0

        while i < n:
            if i + 1 < n:
                instr1 = instructions[i]
                instr2 = instructions[i + 1]

                if (instr1.gate.name == "CNOT" and
                    instr2.gate.name == "CNOT" and
                    instr1.qubits == instr2.qubits and
                    not instr1.param_names and
                    not instr2.param_names):
                    i += 2
                    continue

            new_circuit._instructions.append(instructions[i])
            i += 1

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class IIDElimination(OptimizationPass):
    def __init__(self):
        super().__init__("IIDElimination")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        backend = get_backend()
        xp = backend.xp

        new_circuit = QuantumCircuit(circuit.num_qubits)

        for instr in circuit.instructions:
            if instr.gate.name == "I":
                continue

            if instr.gate.num_qubits == 1 and not instr.param_names:
                gate_matrix = instr.gate.matrix()
                if xp.allclose(gate_matrix, xp.eye(2, dtype=xp.complex128)):
                    continue

            new_circuit._instructions.append(instr)

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class CZCancel(OptimizationPass):
    def __init__(self):
        super().__init__("CZCancel")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)
        instructions = circuit.instructions
        n = len(instructions)
        i = 0

        while i < n:
            if i + 1 < n:
                instr1 = instructions[i]
                instr2 = instructions[i + 1]

                if (instr1.gate.name == "CZ" and
                    instr2.gate.name == "CZ" and
                    set(instr1.qubits) == set(instr2.qubits) and
                    not instr1.param_names and
                    not instr2.param_names):
                    i += 2
                    continue

            new_circuit._instructions.append(instructions[i])
            i += 1

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class SwapCancel(OptimizationPass):
    def __init__(self):
        super().__init__("SwapCancel")

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        new_circuit = QuantumCircuit(circuit.num_qubits)
        instructions = circuit.instructions
        n = len(instructions)
        i = 0

        while i < n:
            if i + 1 < n:
                instr1 = instructions[i]
                instr2 = instructions[i + 1]

                if (instr1.gate.name == "SWAP" and
                    instr2.gate.name == "SWAP" and
                    set(instr1.qubits) == set(instr2.qubits) and
                    not instr1.param_names and
                    not instr2.param_names):
                    i += 2
                    continue

            new_circuit._instructions.append(instructions[i])
            i += 1

        new_circuit._parameters = circuit._parameters.copy()
        return new_circuit


class Transpiler:
    def __init__(self, passes: Optional[list[OptimizationPass]] = None):
        self._passes = passes or [
            IIDElimination(),
            DoubleCXCancel(),
            CZCancel(),
            SwapCancel(),
            SingleQubitFusion(),
        ]

    def add_pass(self, optimization_pass: OptimizationPass) -> "Transpiler":
        self._passes.append(optimization_pass)
        return self

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        optimized = circuit
        for opt_pass in self._passes:
            optimized = opt_pass.run(optimized)
        return optimized

    def optimize(
        self,
        circuit: QuantumCircuit,
        level: int = 1,
    ) -> QuantumCircuit:
        if level == 0:
            return circuit

        passes: list[OptimizationPass] = []

        if level >= 1:
            passes.extend([
                IIDElimination(),
                DoubleCXCancel(),
                CZCancel(),
                SwapCancel(),
            ])

        if level >= 2:
            passes.append(SingleQubitFusion())

        if level >= 3:
            passes.extend([
                DoubleCXCancel(),
                SingleQubitFusion(),
            ])

        optimized = circuit
        for opt_pass in passes:
            optimized = opt_pass.run(optimized)

        return optimized

    def __repr__(self) -> str:
        return f"Transpiler(passes={self._passes})"


def transpile(
    circuit: QuantumCircuit,
    optimization_level: int = 1,
) -> QuantumCircuit:
    transpiler = Transpiler()
    return transpiler.optimize(circuit, level=optimization_level)


def fuse_single_qubit_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    return SingleQubitFusion().run(circuit)


def cancel_double_cx(circuit: QuantumCircuit) -> QuantumCircuit:
    return DoubleCXCancel().run(circuit)
