gpu_run = 1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
if gpu_run:from gpu4mrh import patch_pyscf
import numpy as np
from pyscf import gto, lib, scf, mcscf
from pyscf.mcscf import avas
from pyscf.tools import molden
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from pyscf.lib import param
if gpu_run:
  gpu = libgpu.init()
  libgpu.set_verbose_(gpu,1)
  param.use_gpu = gpu
  param.custom_debug = False
  param.custom_fci = True
  param.mgpu_fci = True

mol=gto.M(atom='int.xyz',basis='def2tzvp',verbose=4,output='int_out_28e22o_v2.log',charge=0,spin=4)
if gpu_run: mol.use_gpu=gpu
mol.max_memory=500000
mf=scf.ROHF(mol)
mf=mf.density_fit()
mf=mf.newton()
mf.kernel()
ao_labels=['Fe 3d','!O 2p','O1 2p', '!C 2p','!H 1s']
ncas,nelecas,mo_coeff=avas.avas(mf,ao_labels,minao=mol.basis,openshell_option=3)
mc_test = mcscf.CASCI(mf,ncas,nelecas)
#molden.from_mo(mol, 'avas_int_O2p.molden',mo_coeff[:,mc_test.ncore:mc_test.ncore+ncas])
mo_list=[13, #Fe-double
         18, #Fe-O b
         20, #C-H b
         24, #O 2p nb
         14,15,16,17, #Pi b
         30,32,33,34, #Pi nb
         35,36,37,38, #3d
         39,40,41,42, #Pi*
         43, #Fe-O ab
         50] #C-h ab
mo_list=[_+mc_test.ncore for _ in mo_list]
las=asyn.LASSCF(mf,(10,3,3,3,3),((8,4),(2,2),(2,2),(2,2),(2,2)),spin_sub=(5,1,1,1,1),verbose=4,assert_no_dupes=False,use_gpu=gpu)
mo_coeff=las.sort_mo(mo_list,mo_coeff)
molden.from_mo(mol, 'sorted_int_2822.molden',mo_coeff)
mo_coeff=las.set_fragments_(([26,28,29,33],[6,12,13],[7,14,15],[10,20,21],[11,22,23]),mo_coeff, mo_occ = mf.mo_occ)
molden.from_mo(mol, 'local_int_2822.molden',mo_coeff)
las.max_cycle_macro=100
from mrh.my_pyscf.mcscf.lasscf_async import old_aa_sync_kernel
las = old_aa_sync_kernel.patch_kernel(las)
if gpu_run: las = las(use_gpu=gpu)
las.relax_params['conv_tol_grad'] = las.conv_tol_grad
las.relax_params['min_cycle_macro'] = 10
#las.relax_params['ah_level_shift']=1e-5
las.kernel(mo_coeff)
molden.from_lasscf(las, 'int_lasscf_2822_v2_2frag.molden')
if gpu_run: libgpu.destroy_device(gpu)
