import numpy as np
import time


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
#LASSCF                                                                                                     
norb = 10 
nelec = 6 
norb_f = (5,5)
jw_nelec = (3,3)
nelec_f = ((3,0),(0,3))
las = LASSCF (mf, (5,5), ((3,0),(0,3)),spin_sub=(4,4))
mo = np.load('kd_6_10_lasmo.npy')
las.max_cycle_macro= 200
las.kernel(mo)
print("LASSCF energy: ", las.e_tot)

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
    print('ucj_op',ucj_op)
    np.save(f'frag{f}_ucj',ucj_op)
    frag_ops.append(ucj_op)
# this is the AS operator
from LUCJ_sampler import LUCJ_circuit
ucj3 = LUCJ_circuit(norb,3,3,cas_h1e,eri)

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
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as Sampler
noise_model = NoiseModel()
cx_depolarizing_prob = 0.2
noise_model.add_all_qubit_quantum_error(
    depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
)
 
noisy_estimator = Sampler(
    options=dict(backend_options=dict(noise_model=noise_model))
)
job = noisy_estimator.run([transpiled],shots=300_000)
result = job.result()[0]          # PubResult for the first circuit
meas = result.data.meas           # BitArray
counts = meas.get_counts()  
r = counts
#simulator = AerSimulator()
#r = simulator.run(transpiled,shots=300_000).result().get_counts()
#print('this is result from simulations',r)
#print(' bitstring_matrix_full', bitstring_matrix_full)
# SQD options

ITERATIONS = 5
open_shell = False
spin_sq =0
# Eigenstate solver options
NUM_BATCHES = 5
SAMPLES_PER_BATCH = 150
MAX_DAVIDSON_CYCLES = 200
num_alpha = mc.nelecas[0]
print('num_alpha',num_alpha)
num_beta = mc.nelecas[1]
print('num_beta',num_beta)
from functools import partial

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch

# SQD options
#energy_tol = 1e-3
#occupancies_tol = 1e-3
#max_iterations = 5

# Eigenstate solver options
num_batches = 10
samples_per_batch = 150
symmetrize_spin = True
carryover_threshold = 1e-4
max_cycle = 200

# Pass options to the built-in eigensolver. If you just want to use the defaults,
# you can omit this step, in which case you would not specify the sci_solver argument
# in the call to diagonalize_fermionic_hamiltonian below.
sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

# List to capture intermediate results
result_history = []


def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")
    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")

bit_array = r
num_orbitals = norb
nelec = jw_nelec
result = diagonalize_fermionic_hamiltonian(
    cas_h1e,
    eri,
    bit_array,
    samples_per_batch=samples_per_batch,
    norb=num_orbitals,
    nelec=nelec,
    num_batches=num_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,
    sci_solver=sci_solver,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    callback=callback,
    seed=rng,
)




x1 = range(len(result_history))
min_e = [
    min(result, key=lambda res: res.energy).energy for result in result_history
]

print('las-lucj-sqd energy is', min_e)
print('total energy is', min_e +nuclear_repulsion_energy)
