import numpy as np   
from pyscf import gto, scf, mcscf, ao2mo, tools, fci, cc
import ffsim
from qiskit_aer import AerSimulator
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
from qiskit.compiler import transpile
import scipy
from scipy import optimize
def LUCJ_load(norb, nelec_a, nelec_b, hcore, eri,las_ci):
    '''LUCJ circuit for loading las'''
    
    '''convert las h1, h2 to mo basis'''
    mol = gto.M()                                                                           
    mol.nelectron = nelec_a + nelec_b
    mol.spin = np.abs(nelec_a-nelec_b)
    mol.nao = norb
    #mol.symmetry = False
    mf_as = mol.RHF()
    mf_as.get_hcore = lambda *args: hcore
    mf_as.get_ovlp = lambda *args: np.eye(norb)
    mf_as._eri = eri 
    mf_as.kernel()
    C = mf_as.mo_coeff
    h1e_MO = np.einsum('pi,pr,rj->ij',C,hcore,C,optimize=True)
    h2e_MO = np.einsum('pi,rj,prqs,qk,sl->ijkl',C,C,eri,C,C,optimize=True)
    
    mol_MO = gto.M()
    mol_MO.nelectron = nelec_a + nelec_b
    mol_MO.spin = np.abs(nelec_a-nelec_b)
    mol_MO.nao = norb
    #mol.symmetry = False
    mf_as_MO= mol_MO.RHF()
    mf_as_MO.get_hcore = lambda *args: h1e_MO
    mf_as_MO.get_ovlp = lambda *args: np.eye(norb)
    mf_as_MO._eri = h2e_MO
    mf_as_MO.kernel() 
    h0e_MO = mf_as_MO.mol.energy_nuc()
    mc  = cc.CCSD(mf_as_MO)
    mc.kernel()
    t1 = mc.t1
    t2 = mc.t2
    #print('t1',t1)
    #print('t2',t2)
   
 
    #------- UCJ operator for all las
    H = ffsim.MolecularHamiltonian(h1e_MO,h2e_MO,h0e_MO)
    n_reps = 6
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=t2, t1=mc.t1, n_reps=n_reps, optimize=True)
    nelec = (nelec_a,nelec_b)
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    hamiltonian = ffsim.linear_operator(H, norb=norb, nelec=nelec)
    num_orbitals = norb
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)] 
    alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)] 
    interaction_pairs = (alpha_alpha_indices, alpha_beta_indices)    
    
    def fun(x):
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
        x,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=False)
        final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
        fidelity = (np.abs(np.vdot(final_state, las_ci.reshape(-1))))**2
        print('current fid', fidelity)
        return 1.0 - fidelity

    result = scipy.optimize.minimize(fun, x0 =operator.to_parameters(interaction_pairs=interaction_pairs),
    method="L-BFGS-B")


    ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
    result.x,
    norb=norb,
    n_reps=n_reps,
    interaction_pairs=interaction_pairs,
    with_final_orbital_rotation=False)
    
    #qlas = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nelec)
    from qiskit import QuantumCircuit, QuantumRegister
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Initialize quantum device backend
    backend = GenericBackendV2(2 * norb)
    # basis_gates=["cp", "xx_plus_yy", "p", "x"])
    # Create a pass manager for circuit transpilation
    LEVEL = 3
    pass_manager = generate_preset_pass_manager(optimization_level=LEVEL, backend=backend)
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    #circuit = circuit.decompose(reps=1)
    transpiled = pass_manager.run(circuit)
    print("Optimization level ",LEVEL,transpiled.count_ops(),transpiled.depth())
    print("two qubit gate depth", transpiled.depth(lambda instruction: instruction.operation.num_qubits == 2))
    '''
    qc = append_measurements(circuit)
    X = QASM_simulator(shots=10_000,alpha=0)
    r = X.run_circuit(qc)
    counts = r
    print(r)
    '''
    return transpiled

