from pyscf import gto, scf, mcscf, tools,ao2mo
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
#from mrh.my_pyscf import mcpdft    

mol=gto.M(atom='int.xyz',basis='def2tzvp',verbose=4,output='int_out_2822_pdft.log',charge=0,spin=4)
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.max_cycle = 1
mf.mo_coeff = np.load('mf.npy')
mf.kernel()
#np.save("mf.npy", mf.mo_coeff)
_,_,mo_coeff,_,_,_ = tools.molden.load('int_lasscf_2822_v2_2frag.molden')

las= LASSCF(mf,(10,3,3,3,3),((8,4),(2,2),(2,2),(2,2),(2,2)),spin_sub=(5,1,1,1,1),verbose=4)
las.mo_coeff = mo_coeff
las.lasci()
print('E(LAS)=',las.e_tot)
#las.localize_init_guess(([26,28,29,33],[6,12,13],[7,14,15],[10,20,21],[11,22,23]),mo_coeff)# mo_occ = mf.mo_occ)
#molden.from_mo(mol, 'local_ts_2822.molden',mo_coeff)
#las.max_cycle_macro=100
#from mrh.my_pyscf.mcscf.lasscf_async import old_aa_sync_kernel
#las = old_aa_sync_kernel.patch_kernel(las)
#if gpu_run: las = las(use_gpu=gpu)
#las.relax_params['conv_tol_grad'] = las.conv_tol_grad
#las.relax_params['min_cycle_macro'] = 10
#las.relax_params['ah_level_shift']=1e-5
#las.kernel(mo_coeff)
#mc = mcpdft.LASSCF(las, 'tPBE',(10,3,3,3,3),((8,4),(2,2),(2,2),(2,2),(2,2)))
#mc.kernel()
#print ("E(LAS-tPBE) =", mc.e_tot)
#---------- sanity checks --------------------
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
norb = 22
nelec = 28 
norb_f = (10,3,3,3,3)
jw_nelec = (16,12)
nelec_f = ((8,4),(2,2),(2,2),(2,2),(2,2))

ncore = las.ncore
ncas = las.ncas
ncas_sub = las.ncas_sub
# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,22,(16,12))
mc.mo_coeff =las.mo_coeff
cas_h1e, e_core = mc.h1e_for_cas()
eri_cas = mc.get_h2eff(las.mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
ecore = mc.get_h1eff()[1]
'''
#-------------------loading las----------------
from LUCJ_loader_updated import LUCJ_load
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
ucj3 = LUCJ_circuit(norb,16,12,cas_h1e,eri)

from las_lucj.ucj_frag import ucj_frag_circuit
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

circuit_opt = ucj_frag_circuit(
        [frag_ops[0], frag_ops[1],frag_ops[2],frag_ops[3],frag_ops[4]], [(8,4), (2,2),(2,2),(2,2),(2,2)], ucj3
)
print('after ucj_optimized frag', flush =True)
circuit_opt.measure_all()  
#----------------submit jobs ----------------------
service = QiskitRuntimeService()
backend = service.least_busy(
    operational=True, simulator=False, min_num_qubits=133
)
print('backend',backend)
initial_layout, _ = get_zigzag_physical_layout(norb, backend=backend)
LAY = initial_layout
circuits = [circuit_opt]
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
#service = QiskitRuntimeService()
'''
job_id ='dal1o4s62pvc739q8hug'
service =  QiskitRuntimeService()
job = service.job(job_id)
job_result = job.result()
pub_result = job_result[0]                                                                                                    
counts = job_result[0].data.meas.get_counts()
#print('counts', counts,flush=True)
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
 
def split_beta_alpha(full_bitstring):
    """Return (alpha_half, beta_half) assuming full bitstring = beta + alpha."""
    norb = len(full_bitstring) // 2
    beta = full_bitstring[:norb]
    alpha = full_bitstring[norb:]
    return alpha, beta
def unique_alpha_beta_separate(bitstrings):
    """Collect unique alpha and beta half strings separately."""
    alpha_dict = OrderedDict()
    beta_dict = OrderedDict()
 
    for bs in bitstrings:
        alpha, beta = split_beta_alpha(bs)
        alpha_dict[alpha] = 1
        beta_dict[beta] = 1
 
    return alpha_dict, beta_dict
 
 
def make_hf_half_string(norb, nelec):
    """Example: norb=5, nelec=3 -> '00111'."""
    return "0" * (norb - nelec) + "1" * nelec
 
# SQD options
ITERATIONS = 20
NUM_BATCHES = 20
SAMPLES_PER_BATCH = 4000
 
# Fulqrum / eigensolver options
EIGSH_TOL = 1e-6
CARRYOVER_THRESHOLD = 1e-4
MAX_EIGSH_ITER = None  # or e.g. 2000
 
num_alpha = mc.nelecas[0]
num_beta = mc.nelecas[1]
 
print("num_alpha", num_alpha, flush=True)
print("num_beta", num_beta, flush=True)
print("num_orbitals", num_orbitals, flush=True)
rng = np.random.default_rng(24)
seed = 24

energy_hist = np.zeros((ITERATIONS, NUM_BATCHES))
spin_sq_hist = np.full((ITERATIONS, NUM_BATCHES), np.nan)  # Fulqrum path below does not compute spin by default
occupancy_hist = []

avg_occupancy = None
carryover_full_strs = []

hf_alpha = make_hf_half_string(num_orbitals, num_alpha)
hf_beta = make_hf_half_string(num_orbitals, num_beta)
best_energy = float("inf")                                                                
best_state = None
from qiskit_addon_sqd.fermion import SCIState                                             
from pyscf import mcpdft   

for i in range(ITERATIONS):
    iter_start = time.perf_counter()
    print(f"\nStarting Fulqrum SQD / configuration recovery iteration {i}", flush=True)

    # 1. Postselect or recover configurations
    if avg_occupancy is None:
        bitstrings, probs = postselect_by_hamming_right_and_left(
            raw_bitstrings,
            raw_probs,
            num_alpha,
            num_beta,
        )
    else:
        # avg_occupancy is expected as (alpha_occs, beta_occs)
        bitstrings, probs = recover_configurations(
            raw_bitstrings,
            raw_probs,
            avg_occupancy[0],
            avg_occupancy[1],
            num_alpha,
            num_beta,
            seed,
        )

    print("Number of postselected/recovered bitstrings:", len(bitstrings), flush=True)

    if len(bitstrings) == 0:
        raise RuntimeError(
            "No bitstrings survived postselection/recovery. "
            "Check bit ordering and whether hamming_right/hamming_left should be swapped."
        )

    # 2. Generate batches
    # Fulqrum tutorial uses subsample(bitstrings, probs, n, seed).
    # Here we preserve NUM_BATCHES and SAMPLES_PER_BATCH.
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

        # 3. Build alpha/beta half-string dictionaries
        alpha_dict = OrderedDict()
        beta_dict = OrderedDict()

        # Always include HF
        alpha_dict[hf_alpha] = 1
        beta_dict[hf_beta] = 1


       # Add carryover from previous iteration
        co_alpha_dict, co_beta_dict = unique_alpha_beta_separate(carryover_full_strs)
        for a in co_alpha_dict:
            alpha_dict[a] = 1
        for b in co_beta_dict:
            beta_dict[b] = 1

        # Add current recovered/subsampled bitstrings
        rec_alpha_dict, rec_beta_dict = unique_alpha_beta_separate(batch)
        for a in rec_alpha_dict:
            alpha_dict[a] = 1
        for b in rec_beta_dict:
            beta_dict[b] = 1

        alpha_half_strs = list(alpha_dict.keys())[:SAMPLES_PER_BATCH]
        beta_half_strs = list(beta_dict.keys())[:SAMPLES_PER_BATCH]

        alpha_half_strs.sort()
        beta_half_strs.sort()

        print(
            f"alpha half strings: {len(alpha_half_strs)}, "
            f"beta half strings: {len(beta_half_strs)}",
            flush=True,
        )
        print(
            f"Subspace dimension: {len(alpha_half_strs)} x {len(beta_half_strs)} "
            f"= {len(alpha_half_strs) * len(beta_half_strs)}",
            flush=True,
        )

        # 4. Build Fulqrum subspace and projected Hamiltonian
        t0 = time.perf_counter()

        S = fq.Subspace([alpha_half_strs, beta_half_strs])

        Hsub = fq.SubspaceHamiltonian(fulqrum_operator, S)
        Hsub_csr_linop = Hsub.to_csr_linearoperator_fast(verbose=False)

        t1 = time.perf_counter()
        print(f"Projection took {t1 - t0:.4f} s", flush=True)

        # 5. Initial guess: determinant with lowest diagonal element
        diag_vec = Hsub.diagonal_vector()
        min_idx = np.where(diag_vec == diag_vec.min())[0]

        v0 = np.zeros(len(S), dtype=Hsub.dtype)
        v0[min_idx] = 1.0
        v0 /= np.linalg.norm(v0)

        # 6. Eigensolve
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
        print(f"Eigensolve took {t3 - t2:.4f} s", flush=True)

        energy = float(eigvals[0])
        eigvec = eigvecs[:, 0]

        if energy < best_energy:     
            best_energy = energy     
            best_state = SCIState(amplitudes=np.ascontiguousarray(
                        eigvec.reshape(len(beta_half_strs), len(alpha_half_strs)).T),
            ci_strs_a=np.array([int(s, 2) for s in alpha_half_strs], dtype=np.int64),
            ci_strs_b=np.array([int(s, 2) for s in beta_half_strs], dtype=np.int64),
            norb=mc.ncas,            
            nelec=mc.nelecas)           
        e_tmp[j] = energy

        print(f"Batch {j} electronic energy: {energy}", flush=True)

        # 7. Occupancies from eigenvector probabilities
        avg_occs = S.get_orbital_occupancies(
            probs=np.abs(eigvec) ** 2,
            norb=num_orbitals,
        )
        occs_tmp.append(avg_occs)

        # 8. Carryover candidates from this batch
        co_full_strs_and_weights = get_carryover_full_strs(
            S,
            np.abs(eigvec),
            CARRYOVER_THRESHOLD,
        )

        carryover_candidates.extend(co_full_strs_and_weights)

        print(
            f"Batch {j} carryover candidates: {len(co_full_strs_and_weights)}",
            flush=True,
        )

    # 9. Average occupancy across batches
    alpha_occs = np.mean([occs[0] for occs in occs_tmp], axis=0)
    beta_occs = np.mean([occs[1] for occs in occs_tmp], axis=0)
    avg_occupancy = (alpha_occs, beta_occs)

    occupancy_hist.append(avg_occupancy)
    energy_hist[i, :] = e_tmp

    # 10. Choose carryover configurations for next iteration
    # Deduplicate and keep sorted by largest weight.
    carryover_dict = OrderedDict()
    carryover_candidates.sort(key=lambda x: x[1], reverse=True)

    for bs, weight in carryover_candidates:
        if bs not in carryover_dict:
            carryover_dict[bs] = weight

    carryover_full_strs = list(carryover_dict.keys())  
    print(f"\nIteration {i} energies:", e_tmp, flush=True)
    print(f"Iteration {i} min energy:", np.min(e_tmp), flush=True)
    print(f"Carryover strings for next iteration: {len(carryover_full_strs)}", flush=True)

    iter_end = time.perf_counter()
    print(f"Iteration {i} took {iter_end - iter_start:.4f} s", flush=True)

print("Fulqrum LASSQD/SQD electronic energy is", np.min(energy_hist), flush=True)
print('total energy is', np.min(energy_hist)+ecore, flush = True)  

if best_state is None:
    raise RuntimeError("No FulQrum eigenstate was retained.")

casdm1s = np.asarray(best_state.rdm(rank=1, spin_summed=False))
casdm2 = best_state.rdm(rank=2, spin_summed=True)

pdft = mcpdft.CASCI(
    mf, "tPBE", mc.ncas, mc.nelecas, ncore=mc.ncore, grids_level=3
)
pdft.mo_coeff = mc.mo_coeff
e_ot = pdft.energy_dft(casdm1s=casdm1s, casdm2=casdm2)
e_pdft = pdft.energy_mcwfn(casdm1s=casdm1s, casdm2=casdm2) + e_ot

print("SQD total energy:", best_energy + ecore)
print("LAS-LUCJ-SQD-PDFT total energy:", e_pdft)
print("On-top energy:", e_ot)
