import numpy as np


# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# quantcum imports
from get_zigzag_physical_layout import get_zigzag_physical_layout 
from hardware_simulator import hardware_simulator
from qiskit_ibm_runtime import QiskitRuntimeService

from LUCJ_loader import LUCJ_load
from LUCJ_sampler import LUCJ_circuit
from qiskit import QuantumCircuit, QuantumRegister
import ffsim

# sqd imports
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    solve_fermion,
)
#----------------Here we perform the calculation with user-defined molecule
mol_name='Al_Fe_Fe.xyz' # Al-Fe-Fe MOF node
basis={'C': 'cc-pvdz','H': 'cc-pvdz','O': 'cc-pvtz','Al': 'cc-pvtz','Fe': 'cc-pvtz'}
output='alfefe.log'
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol=gto.M(atom=mol_name,verbose=4,spin=9,charge=0,basis=basis,max_memory=10000,output=output) #spin = 2S
mol.use_gpu = False
 
mf=scf.RHF(mol)
mf=mf.density_fit()
mf.max_cycle=1
#np.save('hf',mf.mo_coeff)
mf.mo_coeff=np.load('hf.npy')
mf.kernel()
 
mo_guess=np.load('las_1011_ud.npy')
print("HF energy: ", mf.e_tot)
nuclear_repulsion_energy = mol.energy_nuc()   
#LASSCF                                                                                                     
norb = 10  
nelec = 11  
norb_f = (5,5)
jw_nelec = (5,6)
nelec_f = ((5,1),(0,5))
las = LASSCF (mf, (5,5), ((5,1),(0,5)),spin_sub=(5,6))
las.max_cycle_macro= 200 
las.kernel(mo_guess)

print("LASSCF energy: ", las.e_tot)



#tools.molden.from_mo(mol,'as_increase_crcr_las.molden',las.mo_coeff)
ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,10,(5,6))
mc.fix_spin_(ss=3/4)
mc.mo_coeff =las.mo_coeff
cas_h1e, e_core = mc.h1e_for_cas()
eri_cas = mc.get_h2eff(las.mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
mc.kernel()
print('CASCI with LAS orbital', mc.e_tot)
#np.save('CrCr_LS_mc_ci', mc.ci) 


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
ucj3 = LUCJ_circuit(10,5,6,cas_h1e,eri)

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

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=133)
print('backend',backend)
initial_layout, _ = get_zigzag_physical_layout(norb, backend=backend)
LAY = initial_layout
circuits = [circ]
H = hardware_simulator()
H.args['backend'] = backend
#H.args['device_name'] = backend
H.args['initial_layout'] = LAY
H.args['dynamical_decoupling'] = True
H.args['twirling'] = True
H.args['shots'] = 200_000
H.args['circuits_per_job'] = 100
H.submit_jobs(circuits,H.get_service())
#H.submit_jobs(circuits,H.get_service('Qiksit Runtime-lassqd'))
H.print_arguments() 
with open("job_id.txt", "r") as f:
    job_id = f.read().strip()
print('job_id',job_id)
#service = QiskitRuntimeService(channel="ibm_cloud", token='<IBM_QUANTUM_API_KEY>', instance='crn:v1:bluemix:public:quantum-computing:us-east:a/ed5d7d2fb3b249c6baea6864058116cb:36e451b2-5596-4f04-8689-05eced668a0e::')
service =  QiskitRuntimeService()
job = service.job(job_id)
job_result = job.result()
pub_result = job_result[0]
counts = job_result[0].data.meas.get_counts()
r =counts

bitstring_matrix_full, probs_array_full = counts_to_arrays(r)
#print(' bitstring_matrix_full', bitstring_matrix_full)
# SQD options
ITERATIONS = 18
open_shell = False
spin_sq =0
# Eigenstate solver options
NUM_BATCHES = 20
SAMPLES_PER_BATCH = 200
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
    #print('batches',batches)
    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
    e_tmp = np.zeros(NUM_BATCHES)
    s_tmp = np.zeros(NUM_BATCHES)
    occs_tmp = []
    coeffs = []
    for j in range(NUM_BATCHES):
        #print('current batch', batches[j])
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
    print('energy_hist',energy_hist, flush= True)
    spin_sq_hist[i, :] = s_tmp
    print('spin_sq_hist', spin_sq_hist, flush = True)
    occupancy_hist.append(avg_occupancy)
    print()

print('las-lucj-sqd energy is', np.min(energy_hist))
print('total energy is', mc.e_tot - mc.e_cas + np.min(energy_hist))
