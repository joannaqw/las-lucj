import numpy as np

from pyscf import cc, gto

import ffsim

from qiskit import QuantumCircuit, QuantumRegister

import scipy


def LUCJ_load(norb, nelec_a, nelec_b, hcore, eri, las_ci):
    """Create and optimize a LUCJ circuit for one LAS fragment."""

    # Convert fragment integrals to the fragment HF MO basis.
    mol = gto.M()
    mol.nelectron = nelec_a + nelec_b
    mol.spin = abs(nelec_a - nelec_b)
    mol.nao = norb
    mol.symmetry = False

    mf_as = mol.RHF()
    mf_as.get_hcore = lambda *args: hcore
    mf_as.get_ovlp = lambda *args: np.eye(norb)
    mf_as._eri = eri
    mf_as.max_cycle = 200
    mf_as.kernel()

    C = mf_as.mo_coeff

    h1e_MO = np.einsum(
        "pi,pr,rj->ij",
        C,
        hcore,
        C,
        optimize=True,
    )

    h2e_MO = np.einsum(
        "pi,rj,prqs,qk,sl->ijkl",
        C,
        C,
        eri,
        C,
        C,
        optimize=True,
    )

    mol_MO = gto.M()
    mol_MO.symmetry = False
    mol_MO.nelectron = nelec_a + nelec_b
    mol_MO.spin = abs(nelec_a - nelec_b)
    mol_MO.nao = norb

    mf_as_MO = mol_MO.RHF()
    mf_as_MO.get_hcore = lambda *args: h1e_MO
    mf_as_MO.get_ovlp = lambda *args: np.eye(norb)
    mf_as_MO._eri = h2e_MO
    mf_as_MO.max_cycle = 200
    mf_as_MO.kernel()

    h0e_MO = mf_as_MO.mol.energy_nuc()

    # Select balanced or unbalanced CCSD and UCJ classes.
    spin_balanced = nelec_a == nelec_b

    if spin_balanced:
        mc = cc.RCCSD(mf_as_MO)
        ucj_class = ffsim.UCJOpSpinBalanced
        ucj_gate_class = ffsim.qiskit.UCJOpSpinBalancedJW
    else:
        mc = cc.UCCSD(mf_as_MO)
        ucj_class = ffsim.UCJOpSpinUnbalanced
        ucj_gate_class = ffsim.qiskit.UCJOpSpinUnbalancedJW

    mc.max_cycle = 200
    mc.kernel()

    t1 = mc.t1
    t2 = mc.t2

    H = ffsim.MolecularHamiltonian(
        h1e_MO,
        h2e_MO,
        h0e_MO,
    )

    n_reps = 1

    operator = ucj_class.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=n_reps,
    )

    nelec = (nelec_a, nelec_b)
    print("nelec", nelec)

    reference_state = ffsim.hartree_fock_state(
        norb,
        nelec,
    )

    hamiltonian = ffsim.linear_operator(
        H,
        norb=norb,
        nelec=nelec,
    )

    num_orbitals = norb

    alpha_alpha_indices = [
        (p, p + 1)
        for p in range(num_orbitals - 1)
    ]

    alpha_beta_indices = [
        (p, p)
        for p in range(0, num_orbitals, 4)
    ]

    if spin_balanced:
        # UCJOpSpinBalanced expects:
        # (same-spin, opposite-spin)
        interaction_pairs = (
            alpha_alpha_indices,
            alpha_beta_indices,
        )
    else:
        beta_beta_indices = [
            (p, p + 1)
            for p in range(num_orbitals - 1)
        ]

        # UCJOpSpinUnbalanced expects:
        # (alpha-alpha, alpha-beta, beta-beta)
        interaction_pairs = (
            alpha_alpha_indices,
            alpha_beta_indices,
            beta_beta_indices,
        )

    def fun(x):
        current_operator = ucj_class.from_parameters(
            x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )

        final_state = ffsim.apply_unitary(
            reference_state,
            current_operator,
            norb=norb,
            nelec=nelec,
        )

        fidelity = (
            np.abs(
                np.vdot(
                    final_state,
                    las_ci.reshape(-1),
                )
            )
            ** 2
        )

        print("current fid", fidelity)

        return 1.0 - fidelity

    result = scipy.optimize.minimize(
        fun,
        x0=operator.to_parameters(
            interaction_pairs=interaction_pairs
        ),
        method="L-BFGS-B",
    )

    ucj_op = ucj_class.from_parameters(
        result.x,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )

    qubits = QuantumRegister(
        2 * int(norb),
        name="q",
    )

    print("norb", norb)
    print("qubits", qubits)

    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            int(norb),
            nelec,
        ),
        qubits,
    )

    # Balanced operators require UCJOpSpinBalancedJW;
    # unbalanced operators require UCJOpSpinUnbalancedJW.
    circuit.append(
        ucj_gate_class(ucj_op),
        qubits,
    )

    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager,
    )

    backend = GenericBackendV2(2 * norb)
    level = 3

    pass_manager = generate_preset_pass_manager(
        optimization_level=level,
        backend=backend,
    )

    pass_manager.pre_init = ffsim.qiskit.PRE_INIT

    transpiled = pass_manager.run(circuit)

    print(
        "Optimization level",
        level,
        transpiled.count_ops(),
        transpiled.depth(),
    )

    print(
        "two qubit gate depth",
        transpiled.depth(
            lambda instruction:
            instruction.operation.num_qubits == 2
        ),
    )

    return transpiled, ucj_op
