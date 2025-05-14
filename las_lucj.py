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
init_state = get_init_di_las(las)

from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
mapper = JordanWignerMapper()
ansatz = UCCSD(ncas,mc.nelecas,mapper,initial_state=init_state)
params = np.zeros(ansatz.num_parameters)
energy =  qiskit_operator_energy(params,hamiltonian,init_state)
print('VQE energy with las init state')

#---Post LASSCF with LUCJ
laslucj_circuit = LUCJ_sampler(mc.ncas,mc.nelecas[0],mc.nelecas[1],cas_h1e,eri,layout='default')
    #energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
