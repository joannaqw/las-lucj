import ffsim
from qiskit.circuit import QuantumCircuit, QuantumRegister

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
    # TODO perhaps decompose
    return circuit
