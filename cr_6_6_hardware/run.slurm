#!/bin/bash -l            
#SBATCH --job-name=load_las 
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=130:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
##SBATCH --partition=lgagliardi-ld,lgagliardi-hm
##SBATCH --partition=build
#SBATCH --partition=lgagliardi-net
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=12
#SBATCH --error=sqd.err
 
ulimit -s unlimited
export OMP_NUM_THREADS=12
#export PYSCF_MAX_MEMORY=180000
#conda activate laslucj
conda activate lassqd
#module load mkl/2023.1
export PYTHONPATH=
export PYSCF_EXT_PATH=
export PYSCF=/project/lgagliardi/shared/Apps/pyscf
#/home/qjwang/.local/lib/python3.9/site-packages/pyscf
export MRH=/project/lgagliardi/qjwang:/project/lgagliardi/qjwang/mrh
export PYTHONPATH=$MRH$PYSCF:$PYTHONPATH

python las_lucj.py > hardware_run_2rep.out
