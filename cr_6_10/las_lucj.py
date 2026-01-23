import numpy as np
import time

#from get_hamiltonian import get_hamiltonian
#from get_init_state import get_init_di_las

# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo,tools
from pyscf.mcscf import avas

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

from qiskit.primitives import Estimator
from scipy.optimize import minimize


#----------------Here we perform the calculation with user-defined molecule
with open ('CrCr_expt.xyz', 'r') as f:
    carts = f.read ()
frag_name_list = [ 'cr1', 'cr2' ]
my_basis = {'Cr': 'def2-tzvp', 'O': 'def2-svp', 'N': 'def2-svp', 'C': 'def2-svp', 'H': 'def2-svp'}
mol = gto.M (atom = carts, basis = my_basis, symmetry = False, charge = 3 , spin = 6, verbose=lib.logger.INFO , output = 'KD_ls.log')
# Do RHF
dm0 = np.load('crcr_sing_svp_dm.npy')
mf = scf.RHF(mol)
mf.kernel(dm0)
print("HF energy: ", mf.e_tot)
nuclear_repulsion_energy = mol.energy_nuc()   
ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Cr 3d','Cr 4d'] ,minao=mol.basis)
tools.molden.from_mo(mol,'as_increase_crcr_avas_3d.molden',guess_mo_coeff)
mo_list = [64,65,66,67,68, 69, 70, 71,72,73]
#LASSCF                                                                                                     
norb = 10 
nelec = 6 
norb_f = (5,5)
jw_nelec = (3,3)
nelec_f = ((3,0),(0,3))
las = LASSCF (mf, (5,5), ((3,0),(0,3)),spin_sub=(4,4))
frag_atom_list = [[0],[1]]
sort_mo = las.sort_mo(mo_list,mo_coeff=guess_mo_coeff)
#tools.molden.from_mo(mol,'as_increase_crcr_sortmo.molden',sort_mo)
loc_mo_coeff = las.localize_init_guess(frag_atom_list, sort_mo)
#tools.molden.from_mo(mol,'as_increase_crcr_locmocoeff.molden',loc_mo_coeff)
#np.save('CrCr_LS_mo',las.mo_coeff)
#np.save('CrCr_LS_ci', las.ci)
las.max_cycle_macro= 200
las.kernel(loc_mo_coeff)
print("LASSCF energy: ", las.e_tot)
tools.molden.from_mo(mol,'as_increase_crcr_las.molden',las.mo_coeff)
np.save('kd_6_10_lasmo',las.mo_coeff)
#np.save('kf_6_10_lasci', las.ci) 
ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,10,(3,3))
mc.mo_coeff =las.mo_coeff
cas_h1e, e_core = mc.h1e_for_cas()
eri_cas = mc.get_h2eff(las.mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
mc.kernel()
print('CASCI with LAS orbital', mc.e_tot)
#np.save('CrCr_LS_mc_ci', mc.ci) 
#Do nuVQE with LAS
#hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
from LUCJ_loader import LUCJ_load
h1_las = las.h1e_for_las()
eri_las =  las.get_h2eff(las.mo_coeff)
frag_ops = []
for f in range(len(las.ncas_sub)):
    print('current fragment', f)
    h1_frag = h1_las[f][0][0]
    h2_frag = las.get_h2eff_slice(eri_las, f)
    qlas,ucj_op = LUCJ_load(las.ncas_sub[f],nelec_f[f][0],nelec_f[f][1],h1_frag,h2_frag,las.ci[f][0])
    frag_ops.append(ucj_op)
# this is the AS operator
from LUCJ_sampler import LUCJ_circuit
ucj3 = LUCJ_circuit(10,3,3,cas_h1e,eri)

from qiskit import QuantumCircuit, QuantumRegister
import ffsim
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
qr = QuantumRegister(2 * norb, "q")
circ = QuantumCircuit(qr)
circ.append(ffsim.qiskit.PrepareHartreeFockJW(norb, jw_nelec), qr[:])
offset = 0
for idx, op in enumerate(frag_ops):
    frag_qubits = list(range(offset, offset + norb_f[idx])) \
                + list(range(norb + offset, norb + offset + norb_f[idx]))
    ucj_gate_f = ffsim.qiskit.UCJOpSpinUnbalancedJW(op)
    circ.append(ucj_gate_f, [qr[i] for i in frag_qubits])
    offset += norb_f[idx]
circ.barrier()
circ.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj3),qr)
circ.measure_all()
backend = GenericBackendV2(2 * norb)
# basis_gates=["cp", "xx_plus_yy", "p", "x"])
# Create a pass manager for circuit transpilation
LEVEL = 3 
pass_manager = generate_preset_pass_manager(optimization_level=LEVEL, backend=backend)
pass_manager.pre_init = ffsim.qiskit.PRE_INIT
#circuit = circuit.decompose(reps=1)
transpiled = pass_manager.run(circ)
from qiskit import qpy

#with open("test.qpy", "wb") as file:
#    qpy.dump(transpiled, file)
print("Optimization level ",LEVEL,transpiled.count_ops(),transpiled.depth())
print("two qubit gate depth", transpiled.depth(lambda instruction: instruction.operation.num_qubits == 2)) 

from qiskit_aer import AerSimulator
simulator = AerSimulator()
r = simulator.run(transpiled,shots=200_000).result().get_counts()
#print('this is result from simulations',r)
'''
# Initialize ffsim Sampler
sampler = ffsim.qiskit.FfsimSampler(seed=0)

# Form PUB, submit job, retrieve job result, and extract first (and only) PUB result
pub = (circ,)
job = sampler.run([pub], shots=100_000)
result = job.result()
pub_result = result[0]

# Get counts
counts = pub_result
#.data.meas.get_counts()
'''
from qiskit_addon_sqd.counts import counts_to_arrays
bitstring_matrix_full, probs_array_full = counts_to_arrays(r)
#print(' bitstring_matrix_full', bitstring_matrix_full)
# SQD options
ITERATIONS = 5
open_shell = False
spin_sq =0
# Eigenstate solver options
NUM_BATCHES = 5
SAMPLES_PER_BATCH = 120
MAX_DAVIDSON_CYCLES = 200
num_alpha = mc.nelecas[0]
print('num_alpha',num_alpha)
num_beta = mc.nelecas[1]
print('num_beta',num_beta)
#exit()
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
rng = np.random.default_rng(24)
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
    print('batches',batches)
    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    e_tmp = np.zeros(NUM_BATCHES)
    s_tmp = np.zeros(NUM_BATCHES)
    occs_tmp = []
    coeffs = []
    for j in range(NUM_BATCHES):
        print('current batch', batches[j])
        strs_a, strs_b = bitstring_matrix_to_ci_strs(batches[j])
        print(f"Batch {j} subspace dimension: {len(strs_a) * len(strs_b)}")
        energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
            batches[j],
            cas_h1e,
            eri,
            open_shell=open_shell,
            spin_sq=spin_sq,
            max_cycle=MAX_DAVIDSON_CYCLES,
        )
        #+= nuclear_repulsion_energy
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
