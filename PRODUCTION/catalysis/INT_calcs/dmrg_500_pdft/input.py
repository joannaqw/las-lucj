import block2
import os
import pyscf
from pyscf import gto, scf, mcscf, tools,ao2mo
from pyscf import gto, scf, dmrgscf,mcpdft
import numpy as np
#from mrh.my_pyscf import mcpdft    
dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ''

mol=gto.M(atom='int.xyz',basis='def2tzvp',verbose=4,output='int_out_2822_pdft.log',charge=0,spin=4)
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.max_cycle = 1
mf.mo_coeff = np.load('mf.npy')
mf.kernel()
#np.save("mf.npy", mf.mo_coeff)
_,_,mo_coeff,_,_,_ = tools.molden.load('int_lasscf_2822_v2_2frag.molden')
#mc = dmrgscf.DMRGCI(mf, 22, (16,12))
mc = mcpdft.CASCI(mf, 'tPBE', 22, (16, 12))
mc.mo_coeff = mo_coeff
mc.max_memory = 20000
mc.canonicalization = False
mc.natorb = False
#las= LASSCF(mf,(10,3,3,3,3),((8,4),(2,2),(2,2),(2,2),(2,2)),spin_sub=(5,1,1,1,1),verbose=4)
#las.mo_coeff = mo_coeff
# maxM controls the maximum bond dimension (MPS truncation limit)
mc.fcisolver = dmrgscf.DMRGCI(mol)
mc.fcisolver.nroots = 1
mc.fcisolver.maxM = 500
mc.fcisolver.tol = 1e-8
mc.kernel()
print(f'Total DMRG-CASCI Energy (LAS orbitals): {mc.e_mcscf:.10f} Ha')
print(f'Total DMRG-PDFT Energy (tPBE): {mc.e_tot:.10f} Ha')
print(f'On-top functional contribution: {mc.e_ot:.10f} Ha')
