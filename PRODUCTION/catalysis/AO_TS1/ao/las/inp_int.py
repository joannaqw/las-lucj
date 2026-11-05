import pyscf
import numpy as np
from pyscf import gto, lib, scf, mcscf
from pyscf.mcscf import avas
from pyscf.tools import molden
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from pyscf.lib import param

mol=gto.M(atom='ao.xyz',basis='def2tzvp',verbose=4,output='ao_out_21_17.log',charge=0,spin=9)
mol.max_memory=500000
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.mo_coeff = np.load('mf.npy')
mf.max_cycle = 1
mf.kernel()
#np.save('mf.npy',mf.mo_coeff)
#ao_labels=['Fe 3d','!O 2p','O1 2p', '!C 2p','!H 1s']
# int geometry
ao_labels = ['Fe 3d',   # Fe
             '!O 2p',    # oxo O (us
             'N 2p']    # reactive H (int)

# ts geometry: same, but the reactive H is atom 33
#ao_labels = ['26 Fe 3d', '28 O 2p', '24 O 2p', '33 H 1s']

#ncas, nelecas, mo_coeff = avas.avas(mf, ao_labels, minao=mol.basis,
#                                     openshell_option=3, threshold=0.1)
ncas,nelecas,mo_coeff=avas.avas(mf,ao_labels,minao=mol.basis,openshell_option=3)
#mc_test = mcscf.CASCI(mf,ncas,nelecas)
#molden.from_mo(mol, 'avas_ao.molden',mo_coeff)
        #[:,mc_test.ncore:mc_test.ncore+ncas])
#exit()
mo_list=[ 108, 110,111, #oxygen ,111
        112, 117, 118, 119, 120, #fe left 
        106, 114,115, 116, 123, #Fe
        105,107,121,122 ] #n2o
         
#mo_list=[_+mc_test.ncore for _ in mo_list]

las=asyn.LASSCF(mf,(9,8),((7,3),(8,3)),verbose=4)
sorted_mo=las.sort_mo(mo_list,mo_coeff)
molden.from_mo(mol, 'sorted_ao_2117.molden',sorted_mo)

localized_mo1=las.set_fragments_(([17,19,29,30],[23,24]),sorted_mo)
molden.from_mo(mol, 'local_ao_2117.molden',localized_mo1)
las.max_cycle_macro=200
las.kernel(localized_mo1)
molden.from_lasscf(las, 'ao_lasscf_2117_2frag.molden')

mc= mcscf.CASSCF(mf,17,(15,6))
mc.max_cycle_macro = 100
mc.kernel(las.mo_coeff)



