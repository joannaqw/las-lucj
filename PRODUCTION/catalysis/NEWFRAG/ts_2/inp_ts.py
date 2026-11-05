import pyscf
import numpy as np
from pyscf import gto, lib, scf, mcscf, tools
from pyscf.mcscf import avas
from pyscf.tools import molden
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn

mol=gto.M(atom='ts.xyz',basis='def2tzvp',verbose=4,output='ts_out_28e22o_v2.log',charge=0,spin=4)
mol.max_memory=500000
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.mo_coeff = np.load('mf.npy')
mf.max_cycle = 1
mf.kernel()
#_,_,mo_init_coeff,_,_,_ = tools.molden.load('local_int_2822.molden')
#mc_test=mcscf.CASCI(mf, 13,18)#norb,nelec
_,_,mo_init_coeff,_,_,_ = tools.molden.load('avas_ts_new2frag.molden')

#mo_coeff = mcscf.project_init_guess(mc_test, mo_init_coeff)
#ao_labels=['Fe 3d','!O 2p','O1 2p', '!C 2p','!H 1s']
# int geometry
#ao_labels = ['Fe 3d',   # Fe
#             '!O 2p',    # oxo O (use '28 O1 2p' if it is labeled O1 in int.xyz)
#             '!C 2p', 
#             '!H 1s']    # reactive H (int)

# ts geometry: same, but the reactive H is atom 33
#ao_labels = ['26 Fe 3d', '28 O 2p', '24 O 2p', '33 H 1s']

#ncas, nelecas, mo_coeff = avas.avas(mf, ao_labels, minao=mol.basis,
#                                     openshell_option=3, threshold=0.1)
#ncas,nelecas,mo_coeff=avas.avas(mf,ao_labels,minao=mol.basis,#openshell_option=3)
#mc_test = mcscf.CASCI(mf,ncas,nelecas)
#molden.from_mo(mol, 'avas_ts_new2frag.molden',mo_coeff)
mo_list=[3,4,5, #bottom o
         7,15, #c-h
         10,11,13,20,21, #topO-fe
         16,19, #3d
         23]#bottom o top o] 
#mo_list=[_+mc_test.ncore for _ in mo_list]

las=asyn.LASSCF(mf,(11,2),((10,6),(1,1)),spin_sub=(5,1),verbose=4)
sorted_mo=las.sort_mo(mo_list,mo_init_coeff)
molden.from_mo(mol, 'sorted_ts_1813.molden',sorted_mo)

localized_mo1=las.set_fragments_(([24,26,28],[29,30,31,32,33]),sorted_mo)
molden.from_mo(mol, 'local1_ts_1813.molden',localized_mo1)
las.max_cycle_macro=200
las.kernel(localized_mo1)
molden.from_lasscf(las, 'ts1_lasscf_1813_2frag.molden')

mc= mcscf.CASSCF(mf,18,(11,7))
mc.max_cycle_macro = 100
mc.kernel(localized_mo1)

localized_mo2=las.set_fragments_(([24,26],[28,29,30,31,32,33]),sorted_mo)
molden.from_mo(mol, 'local2_ts_1813.molden',localized_mo2)
#scheme 1: [Fe-O_bot] | [0-CH4]
las2=asyn.LASSCF(mf,(5,8), ((4,4),(7,3)),spin_sub=(1,5),verbose=4)
las2.max_cycle_macro=200
las2.max_cycle_macro=100
las2.kernel(localized_mo2)
molden.from_lasscf(las2, 'ts2_lasscf_1813_2frag_VW.molden')
