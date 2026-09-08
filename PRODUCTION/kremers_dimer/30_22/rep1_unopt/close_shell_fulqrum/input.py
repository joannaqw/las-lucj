import numpy as np

#from get_hamiltonian import get_hamiltonian
#from get_init_state import get_init_di_las

# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo,tools
import numpy as np
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op
import time
 
import scipy.sparse.linalg as spla
from collections import OrderedDict
import fulqrum as fq
from fulqrum.core.sqd import (
    postselect_by_hamming_right_and_left,
    subsample,
    recover_configurations,
    get_carryover_full_strs,
)
# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from las_lucj.ucj_frag import ucj_frag_circuit
from get_zigzag_physical_layout import get_zigzag_physical_layout
from hardware_simulator import hardware_simulator
from qiskit_ibm_runtime import QiskitRuntimeService
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
#ncas,nelecas,guess_mo_coeff=avas.kernel(mf,['Cr 3d','Cr 4d'] ,minao=mol.basis)
#tools.molden.from_mo(mol,'as_increase_crcr_avas_3d.molden',guess_mo_coeff)
#mo_list = [64,65,66,67,68, 69, 70, 71,72,73]
mol2, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = tools.molden.load('cr2_30_22-las.molden')
#LASSCF                                                                                                     
norb = 22
nelec = 30 
norb_f = (11,11)
jw_nelec = (15,15)
nelec_f = ((9,6),(6,9))
las = LASSCF (mf, (11,11), ((9,6),(6,9)),spin_sub=(4,4))
#mo = np.load('kd_6_10_lasmo.npy')
#frag_atom_list = [[0],[1]]
#sort_mo = las.sort_mo(mo_list,mo_coeff=guess_mo_coeff)
#tools.molden.from_mo(mol,'as_increase_crcr_sortmo.molden',sort_mo)
#loc_mo_coeff = las.localize_init_guess(frag_atom_list, sort_mo)
#tools.molden.from_mo(mol,'as_increase_crcr_locmocoeff.molden',loc_mo_coeff)
#np.save('CrCr_LS_mo',las.mo_coeff)
#np.save('CrCr_LS_ci', las.ci)
las.max_cycle_macro= 200
las.kernel(mo_coeff)
print("LASSCF energy: ", las.e_tot)
#print('LASSCF core energy', las.energy_nuc())
#print('LASSCF estates', las.e_states)
#print('LASSCF elec', las.states_energy_elec())
ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,22,(15,15))
mc.mo_coeff =las.mo_coeff
cas_h1e, e_core = mc.h1e_for_cas()
eri_cas = mc.get_h2eff(las.mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
ecore = mc.get_h1eff()[1]
print('ecore',ecore)
#np.save('CrCr_LS_mc_ci', mc.ci) 
#Do nuVQE with LAS
#hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
'''
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
print('before ucj3',flush =True)
ucj3 = LUCJ_circuit(22,15,15,cas_h1e,eri)
print('ucj3',flush = True)
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
print('before ucj_optimized frag', flush = True)
#circuit_opt = ucj_frag_circuit(
#        [frag_ops[0], frag_ops[1]], [(9,6), (6,9)], ucj3
#)
#print('after ucj_optimized frag', flush =True)
#circuit_opt.measure_all()  
'''
'''
service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, simulator=False, min_num_qubits=133
)
print('backend',backend)
initial_layout, _ = get_zigzag_physical_layout(norb, backend=backend)
LAY = initial_layout
circuits = [circ]
H = hardware_simulator()

H.args['backend'] = backend
H.args['initial_layout'] = LAY 
H.args['dynamical_decoupling'] = True
H.args['twirling'] = True
H.args['shots'] = 500_000
H.args['circuits_per_job'] = 100 
H.submit_jobs(circuits,H.get_service())
H.print_arguments() 
from qiskit_ibm_runtime import QiskitRuntimeService
with open("job_id.txt", "r") as f:
    job_id = f.read().strip()
print('job_id',job_id)
service = QiskitRuntimeService()
'''
service = QiskitRuntimeService()
job_id ='d7u0gqcf3ras73b9qks0'
job = service.job(job_id)
job_result = job.result()
pub_result = job_result[0]
counts = job_result[0].data.meas.get_counts()

raw_bitstrings = []
raw_probs = []

for bs, cnt in counts.items():
    raw_bitstrings.append(bs)
    raw_probs.append(cnt)

raw_probs = np.asarray(raw_probs, dtype=float)
raw_probs /= raw_probs.sum()


num_orbitals = cas_h1e.shape[0]

fermionic_op = integrals_to_fq_fermionic_op(
    one_body_integrals=cas_h1e,
    two_body_integrals=eri,
)

print(f"Num fermionic terms: {fermionic_op.num_terms}")

fulqrum_operator = fermionic_op.extended_jw_transformation()

print(f"Num qubit terms after extended JW: {fulqrum_operator.num_terms}")

def unique_alpha_beta_combined(bitstrings):
    """A utility function for getting unique alpha and beta halves from full bitstrings.

    Args:
        bitstrings (list[str]): A list of full bitstrings consisting of both alpha and beta parts.

    Returns:
        A dictionary with combined unique ``alpha`` and ``beta`` half strings.
    """
    if not bitstrings:
        return OrderedDict()

    unique_ab = OrderedDict()
    num_spatial_orb = len(bitstrings[0]) // 2

    for bs in bitstrings:
        a = bs[num_spatial_orb:]
        b = bs[:num_spatial_orb]

        unique_ab[a] = 1
        unique_ab[b] = 1

    return unique_ab

def make_hf_half_string(norb, nelec):
    """Example: norb=5, nelec=3 -> '00111'."""
    return "0" * (norb - nelec) + "1" * nelec

# SQD options
ITERATIONS = 13
NUM_BATCHES = 12
SAMPLES_PER_BATCH = 250

# Fulqrum eigensolver / carryover options
EIGSH_TOL = 1e-5
CARRYOVER_THRESHOLD = 1e-4
MAX_EIGSH_ITER = None

num_alpha = mc.nelecas[0]
num_beta = mc.nelecas[1]

print("num_alpha", num_alpha, flush=True)
print("num_beta", num_beta, flush=True)
print("num_orbitals", num_orbitals, flush=True)

if num_alpha != num_beta:
    raise ValueError(
        "This tutorial-style shared-half-string implementation assumes "
        "num_alpha == num_beta. For open-shell systems, use separate alpha/beta pools."
    )

rng = np.random.default_rng(24)
seed = 24

energy_hist = np.zeros((ITERATIONS, NUM_BATCHES))
spin_sq_hist = np.full((ITERATIONS, NUM_BATCHES), np.nan)
occupancy_hist = []

avg_occupancy = None
carryover_full_strs = []

hf_hs = make_hf_half_string(num_orbitals, num_alpha)

for i in range(ITERATIONS):
    iter_start = time.perf_counter()
    print(f"\nStarting Fulqrum closed-shell SQD iteration {i}", flush=True)

    # 1. Postselect or recover configurations
    if avg_occupancy is None:
        bitstrings, probs = postselect_by_hamming_right_and_left(
            raw_bitstrings,
            raw_probs,
            num_alpha,
            num_beta,
        )
    else:
        bitstrings, probs = recover_configurations(
            raw_bitstrings,
            raw_probs,
            avg_occupancy[0],
            avg_occupancy[1],
            num_alpha,
            num_beta,
            seed + i,
        )

    print("Number of postselected/recovered bitstrings:", len(bitstrings), flush=True)

    if len(bitstrings) == 0:
        raise RuntimeError(
            "No bitstrings survived postselection/recovery. "
            "Check bit ordering and hamming_right/hamming_left convention."
        )

    # 2. Generate batches
    batches = [
        subsample(
            bitstrings,
            probs,
            min(SAMPLES_PER_BATCH, len(bitstrings)),
            seed + j + 1000 * i,
        )
        for j in range(NUM_BATCHES)
    ]

    e_tmp = np.zeros(NUM_BATCHES)
    occs_tmp = []
    carryover_candidates = []

    for j, batch in enumerate(batches):
        print(f"\nIteration {i}, batch {j}", flush=True)

        # 3. Tutorial-style shared half-string pool
        half_strs_dict = OrderedDict()

        # Add HF half string
        half_strs_dict[hf_hs] = 1

        # Add carryovers
        unique_ab_co = unique_alpha_beta_combined(carryover_full_strs)
        for bs in unique_ab_co.keys():
            half_strs_dict[bs] = 1

        # Add recovered/subsampled bitstrings
        unique_ab_recovered = unique_alpha_beta_combined(batch)
        for bs in unique_ab_recovered.keys():
            half_strs_dict[bs] = 1

        half_strs = list(half_strs_dict.keys())
        half_strs.sort()

        print("  num total half strs:", len(half_strs), flush=True)
        print(
            "  subspace dimension:",
            len(half_strs),
            "x",
            len(half_strs),
            "=",
            len(half_strs) ** 2,
            flush=True,
        )

        # 4. Build Fulqrum subspace
        # Closed-shell tutorial style:
        # alpha and beta share the same half-string pool.
        S = fq.Subspace([half_strs, half_strs])

        # 5. Project Hamiltonian
        t0 = time.perf_counter()

        Hsub = fq.SubspaceHamiltonian(fulqrum_operator, S)
        Hsub_csr_linop = Hsub.to_csr_linearoperator_fast(verbose=False)

        t1 = time.perf_counter()
        print(f"  projection took {t1 - t0:.4f} s", flush=True)

        # 6. Initial guess from lowest diagonal element
        diag_vec = Hsub.diagonal_vector()
        min_idx = np.where(diag_vec == diag_vec.min())[0]

        v0 = np.zeros(len(S), dtype=Hsub.dtype)
        v0[min_idx] = 1.0
        v0 /= np.linalg.norm(v0)

        # 7. Solve projected Hamiltonian
        t2 = time.perf_counter()

        eigvals, eigvecs = spla.eigsh(
            Hsub_csr_linop,
            k=1,
            which="SA",
            tol=EIGSH_TOL,
            v0=v0,
            maxiter=MAX_EIGSH_ITER,
        )

        t3 = time.perf_counter()
        print(f"  eigensolve took {t3 - t2:.4f} s", flush=True)

        energy = float(eigvals[0])
        eigvec = eigvecs[:, 0]

        e_tmp[j] = energy

        print(f"  batch {j} electronic energy: {energy}", flush=True)

        # 8. Compute average orbital occupancies
        avg_occs = S.get_orbital_occupancies(
            probs=np.abs(eigvec) ** 2,
            norb=num_orbitals,
        )

        occs_tmp.append(avg_occs)

        # 9. Extract carryover full strings from important CI amplitudes
        co_full_strs_and_weights = get_carryover_full_strs(
            S,
            np.abs(eigvec),
            CARRYOVER_THRESHOLD,
        )

        carryover_candidates.extend(co_full_strs_and_weights)

        print(
            f"  batch {j} carryover candidates:",
            len(co_full_strs_and_weights),
            flush=True,
        )

    # 10. Average occupancy across batches
    alpha_occs = np.mean([occs[0] for occs in occs_tmp], axis=0)
    beta_occs = np.mean([occs[1] for occs in occs_tmp], axis=0)
    avg_occupancy = (alpha_occs, beta_occs)

    occupancy_hist.append(avg_occupancy)
    energy_hist[i, :] = e_tmp

    # 11. Deduplicate carryover full strings, sorted by weight
    carryover_candidates.sort(key=lambda x: x[1], reverse=True)

    carryover_dict = OrderedDict()
    for bs, weight in carryover_candidates:
        if bs not in carryover_dict:
            carryover_dict[bs] = weight

    carryover_full_strs = list(carryover_dict.keys())

    print(f"\nIteration {i} energies:", e_tmp, flush=True)
    print(f"Iteration {i} min energy:", np.min(e_tmp), flush=True)
    print("Carryover full strings for next iteration:", len(carryover_full_strs), flush=True)

    iter_end = time.perf_counter()
    print(f"Iteration {i} took {iter_end - iter_start:.4f} s", flush=True)

print("Fulqrum closed-shell SQD electronic energy is", np.min(energy_hist), flush=True)
print('total energy', ecore+ np.min(energy_hist))
