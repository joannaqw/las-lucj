import pyscf
import numpy as np
from pyscf import gto, lib, scf, mcscf, tools
from pyscf.mcscf import avas
from pyscf.tools import molden
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn

mol=gto.M(atom='ts1.xyz',basis='def2-tzvp',verbose=4,output='ts_out_8e8o.log',charge=0,spin=4)
mol.max_memory=500000
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.mo_coeff = np.load('mf.npy')
mf.max_cycle = 1
mf.kernel()
#np.save('mf.npy',mf.mo_coeff)

#old_mol = gto.M(atom='ts1.xyz',basis='def2-tzvp',verbose=4,charge=0,spin=4)

#old_mol,_,mo_init_coeff,_,_,_ = tools.molden.load('casscf_910_lowest.molden')
#mc_test=mcscf.CASCI(mf, 13,18)#norb,nelec
old_mol,_,mo_init_coeff,_,_,_ = tools.molden.load('casscf_88_sa2_ss.molden')
mc = mcscf.CASSCF(mf,8,(6,2))
#fe_mos  = [109, 110, 111, 112]
#n2o_mos = [107, 108, 113, 114]
cas_list = [107,108,109,110,111,112,113,114]
#cas_list = fe_mos + n2o_mos
mo_old_sorted = mc.sort_mo(
    cas_list, mo_coeff=mo_init_coeff, base=1
)
mo_coeff = mcscf.project_init_guess(
    mc,
    mo_old_sorted[:, :mc.ncore + mc.ncas],
    prev_mol=old_mol,
    priority='active',
)
#mo_coeff = mcscf.project_init_guess(mc, mo_init_coeff, prev_mol=old_mol)
mc.max_cycle_macro = 200
mc.kernel(mo_coeff)
molden.from_mcscf(mc, "casscf_88_sa2_ss_tzvp.molden", cas_natorb=True)
exit()
from mrh.my_pyscf.tools import molden
molden.from_sa_mcscf(mc, "sa2_state0.molden", state=0, cas_natorb=True)
molden.from_sa_mcscf(mc, "sa2_state1.molden", state=1, cas_natorb=True)

exit()
mo = mc.mo_coeff
mc2=mcscf.CASCI(mf,8,(6,2))
mc2.fcisolver = csf_solver (mol, smult=5)
mc2.fcisolver.nroots=10
mc2.kernel(mo)
for root, energy in enumerate(mc2.e_tot[:2]):
    print(f"Quintet CASCI root {root}: {energy:.12f} Eh",
          file=mol.stdout, flush=True)

#ao_labels=['Fe 3d','!O 2p','O1 2p', '!C 2p','!H 1s']
# int geometry
#ao_labels = ['Fe 3d',   # Fe
#             '!O 2p', 'N 2p']
#             '!C 2p', 
#             '!H 1s']    # reactive H (int)

# ts geometry: same, but the reactive H is atom 33
#ao_labels = ['26 Fe 3d', '28 O 2p', '24 O 2p', '33 H 1s']

#ncas, nelecas, mo_coeff = avas.avas(mf, ao_labels, minao=mol.basis,
#                                     openshell_option=3, threshold=0.1)
#ncas,nelecas,mo_coeff=avas.avas(mf,ao_labels,minao=mol.basis,#openshell_option=3)
#mc_test = mcscf.CASCI(mf,ncas,nelecas)
#molden.from_mo(mol, 'avas_ts1_one.molden',mo_coeff)
#exit()
#mo_list=[115,107,112, #n2o: pi pi*and sigma*
#         100,101,109,110,111,114] #fe3d + sigma
         #104,105,106] #O
#mo_list=[_+mc_test.ncore for _ in mo_list]

#las=asyn.LASSCF(mf,(9,),((7,3),),verbose=4)
#sorted_mo=las.sort_mo(mo_list,mo_coeff)
#molden.from_mo(mol, 'sorted_ts1_one_910.molden',sorted_mo)

#localized_mo1=las.set_fragments_(([17,19,29,30],),sorted_mo)
#molden.from_mo(mol, 'local_ts1_one_910.molden',localized_mo1)
#las.max_cycle_macro=200
#las.kernel(localized_mo1)
#molden.from_lasscf(las, 'ts1_lasscf_910_1frag.molden')
from pyscf.tools import molden
mc= mcscf.CASSCF(mf,7,(5,1))
mc.max_cycle_macro = 200
mc.kernel(mo_init_coeff)
molden.from_mcscf(mc, "casscf_localmo_67.molden", cas_natorb=True)

#mc2 = mcscf.CASSCF(mf,9,(7,3))
#mc2.max_cycle_macro = 200
#mc2.kernel(sorted_mo)
#molden.from_mcscf(mc2, "casscf_sortedmo.molden", cas_natorb=True)
