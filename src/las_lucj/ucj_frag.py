import ffsim
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Barrier
from qiskit.converters import circuit_to_dag, dag_to_circuit

gate_map = {
    ffsim.UCJOpSpinBalanced: ffsim.qiskit.UCJOpSpinBalancedJW,
    ffsim.UCJOpSpinUnbalanced: ffsim.qiskit.UCJOpSpinUnbalancedJW,
}


def ucj_frag_circuit_naive(
    small_ucj_ops: list[ffsim.UCJOpSpinBalanced | ffsim.UCJOpSpinUnbalanced],
    nelecs: list[tuple[int, int]],
    big_ucj_op: ffsim.UCJOpSpinBalanced | ffsim.UCJOpSpinUnbalanced,
) -> QuantumCircuit:
    norb = sum(op.norb for op in small_ucj_ops)
    assert norb == big_ucj_op.norb
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    current_orb = 0
    for op, nelec in zip(small_ucj_ops, nelecs):
        active_qubits = (
            qubits[current_orb : current_orb + op.norb]
            + qubits[norb + current_orb : norb + current_orb + op.norb]
        )
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(op.norb, nelec), active_qubits)
        circuit.append(gate_map[type(op)](op), active_qubits)
        current_orb += op.norb
    circuit.append(gate_map[type(big_ucj_op)](big_ucj_op), qubits)
    return circuit


def ucj_frag_circuit_opt(
    small_ucj_ops: list[ffsim.UCJOpSpinBalanced | ffsim.UCJOpSpinUnbalanced],
    nelecs: list[tuple[int, int]],
    big_ucj_op: ffsim.UCJOpSpinBalanced | ffsim.UCJOpSpinUnbalanced,
) -> QuantumCircuit:
    norb = sum(op.norb for op in small_ucj_ops)
    assert norb == big_ucj_op.norb
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    current_orb = 0
    for op, nelec in zip(small_ucj_ops, nelecs):
        active_qubits = (
            qubits[current_orb : current_orb + op.norb]
            + qubits[norb + current_orb : norb + current_orb + op.norb]
        )
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(op.norb, nelec), active_qubits)
        circuit.append(gate_map[type(op)](op), active_qubits)
        current_orb += op.norb
    circuit.barrier()
    circuit.append(gate_map[type(big_ucj_op)](big_ucj_op), qubits)
    circuit = ffsim.qiskit.PRE_INIT.run(circuit)
    dag = circuit_to_dag(circuit)
    barrier_node = dag.op_nodes(op=Barrier)[0]
    predecessors = list(dag.op_predecessors(barrier_node))
    successors = list(dag.op_successors(barrier_node))
    assert len(predecessors) == len(small_ucj_ops)
    assert len(successors) == 1
    big_node = list(dag.op_successors(barrier_node))[0]
    orb_rot_a = big_node.op.orbital_rotation_a
    orb_rot_b = big_node.op.orbital_rotation_b
    for small_node in predecessors:
        qubit_indices = [dag.find_bit(qubit).index for qubit in small_node.qargs]
        small_orb_rot_a = small_node.op.orbital_rotation_a
        small_orb_rot_b = small_node.op.orbital_rotation_b
        norb_small, _ = small_orb_rot_a.shape
        assert len(qubit_indices) == 2 * norb_small
        orbs = qubit_indices[:norb_small]
        assert qubit_indices[norb_small:] == [orb + norb for orb in orbs]
        orb_rot_a = ffsim.linalg.apply_matrix_to_slices(
            orb_rot_a, small_orb_rot_a.T, [np.s_[:, i] for i in orbs]
        )
        orb_rot_b = ffsim.linalg.apply_matrix_to_slices(
            orb_rot_b, small_orb_rot_b.T, [np.s_[:, i] for i in orbs]
        )
    dag.replace_block_with_op(
        predecessors + [barrier_node] + successors,
        ffsim.qiskit.OrbitalRotationJW(norb, (orb_rot_a, orb_rot_b)),
        {q: i for i, q in enumerate(big_node.qargs)},
    )
    return dag_to_circuit(dag)
