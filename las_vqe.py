import numpy as np
import time

from get_hamiltonian import get_hamiltonian
from operators import JHJ_operator, JJ_operator
from initial_state import get_initial_state

# PySCF imports
from pyscf import gto, scf, lib, mcscf, ao2mo

# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

# Qiskit imports
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit.primitives import Estimator
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal

def qiskit_operator_energy(params,qubitOp,psi):
    estimator = Estimator()
    params = params
    qubitOp = qubitOp
    psi = psi
    job = estimator.run([psi], [qubitOp], [params])
    job_result = job.result() 
    return job_result

#----------------Here we perform the calculation with user-defined molecule
xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 1.6 0.1
             H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g_{equ}.log',symmetry=False, verbose=lib.logger.DEBUG)
 
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
ncas_sub = las.ncas_sub

# CASCI h1 & h2 for VQE Hamiltonian
mc = mcscf.CASCI(mf,4,4)
mc.kernel(loc_mo_coeff)
cas_h1e, e_core = mc.h1e_for_cas()
 
eri_cas = mc.get_h2eff(loc_mo_coeff)
eri = ao2mo.restore(1, eri_cas,mc.ncas)
 
#Do nuVQE with LAS
hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, eri)
n_qubits =int(np.sum(ncas_sub)*2)
n_blocks = 2
num_vqe_params =((n_blocks+1)*n_qubits) #this is for ry+linear
print("vqe params:", num_vqe_params)
params=np.random.uniform(0., 2.*np.pi, size=num_vqe_params)
ansatz = TwoLocal(n_qubits, 'ry', 'cx', 'linear', reps=n_blocks, insert_barriers=True,initial_state = get_initial_state(ncas_sub, las))

def cost_func(params,n_qubits,num_vqe_params,hamiltonian,ansatz):
    energy_qiskit=qiskit_operator_energy(params,hamiltonian,ansatz).values
    return energy_qiskit

# Running nuVQE
t0 = time.time()
res = minimize(cost_func,params,args = (n_qubits,num_vqe_params,hamiltonian,ansatz),method='L-BFGS-B',options={'disp':True})
t1 = time.time()
print("Time taken for VQE: ",t1-t0)
print("VQE energies: ", res)
print("params:",res.x)
