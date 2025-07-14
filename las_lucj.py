import numpy as np
import time

from get_hamiltonian import get_hamiltonian
from get_init_state import get_init_di_las

# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# Qiskit imports
import qiskit_nature
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
qiskit_nature.settings.use_pauli_sum_op = False
from qiskit.primitives import Estimator
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal


def qiskit_operator_energy(params,qubitOp,psi):
    estimator = Estimator()
    job = estimator.run([psi], [qubitOp], [params])
    job_result = job.result() 
    return job_result


#----------------Here we perform the calculation with user-defined molecule
xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 1.6 0.1
             H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = '6-31g', output='h4_6-31g_{equ}.log',symmetry=False, verbose=lib.logger.DEBUG)
 
# Do RHF
mf = scf.RHF(mol).run()
print("HF energy: ", mf.e_tot)
 
#LASSCF                                                                                                                                                                     
las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(loc_mo_coeff)
loc_mo_coeff = las.mo_coeff
print("LASSCF energy: ", las.e_tot)
 
ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,4,4)
mc.kernel(loc_mo_coeff)
cas_h1e, e_core = mc.h1e_for_cas()
 
eri_cas = mc.get_h2eff(loc_mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
 
#Do nuVQE with LAS
hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
#init_state = get_init_di_las(las)
'''
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
mapper = JordanWignerMapper()
ansatz = UCCSD(ncas,mc.nelecas,mapper,initial_state=init_state)
params = np.zeros(ansatz.num_parameters)
energy =  qiskit_operator_energy(params,hamiltonian,init_state)
print('VQE energy with las init state')
'''
#---Post LASSCF with LUCJ and SQD
from LUCJ_sampler import LUCJ_circuit
laslucj_circuit = LUCJ_circuit(mc.ncas,mc.nelecas[0],mc.nelecas[1],cas_h1e,eri,layout='default')
from qiskit_aer import AerSimulator
simulator = AerSimulator(method='matrix_product_state')
qc = transpile(laslucj_circuit, simulator)
r = simulator.run(qc,shots=100_000).result().get_counts()
bitstring_matrix_full, probs_array_full = counts_to_arrays(r)
# SQD options
ITERATIONS = 5

# Eigenstate solver options
NUM_BATCHES = 1
SAMPLES_PER_BATCH = 100
MAX_DAVIDSON_CYCLES = 200
# Self-consistent configuration recovery loop
energy_hist = np.zeros((ITERATIONS, NUM_BATCHES))  # energy history
spin_sq_hist = np.zeros((ITERATIONS, NUM_BATCHES))  # spin history
occupancy_hist = []
avg_occupancy = None
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    solve_fermion,
)

for i in range(ITERATIONS):
    print(f"\nStarting configuration recovery iteration {i}")
    # On the first iteration, we have no orbital occupancy information from the
    # solver, so we just post-select from the full bitstring set based on Hamming weight.
    if avg_occupancy is None:
        bitstring_matrix_tmp = bitstring_matrix_full
        probs_array_tmp = probs_array_full

    # If there is average orbital occupancy information, use it to refine the full set of noisy configurations
    else:
        bitstring_matrix_tmp, probs_array_tmp = recover_configurations(
            bitstring_matrix_full,
            probs_array_full,
            avg_occupancy,
            num_alpha,
            num_beta,
            rand_seed=rng,
        )

    # Throw out configurations with an incorrect particle number in either the spin-up or spin-down systems
    batches = postselect_and_subsample(
        bitstring_matrix_tmp,
        probs_array_tmp,
        hamming_right=num_alpha,
        hamming_left=num_beta,
        samples_per_batch=SAMPLES_PER_BATCH,
        num_batches=NUM_BATCHES,
        rand_seed=rng,
    )

    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    e_tmp = np.zeros(NUM_BATCHES)
    s_tmp = np.zeros(NUM_BATCHES)
    occs_tmp = []
    coeffs = []
    for j in range(NUM_BATCHES):
        strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
        print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
        energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
            batches[j],
            core_hamiltonian,
            electron_repulsion_integrals,
            open_shell=open_shell,
            spin_sq=spin_sq,
            max_davidson=MAX_DAVIDSON_CYCLES,
        )
        energy_sci += nuclear_repulsion_energy
        e_tmp[j] = energy_sci
        s_tmp[j] = spin
        occs_tmp.append(avg_occs)
        coeffs.append(coeffs_sci)

    # Combine batch results
    avg_occupancy = tuple(np.mean(occs_tmp, axis=0))

    # Track optimization history
    energy_hist[i, :] = e_tmp
    spin_sq_hist[i, :] = s_tmp
    occupancy_hist.append(avg_occupancy)
    print()

print('las-lucj-sqd energy is', np.min(energy_hist))
