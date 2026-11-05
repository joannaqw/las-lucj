#!/bin/bash -l            
#SBATCH --job-name=cr66
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=23:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
##SBATCH --partition=lgagliardi-ld,lgagliardi-hm
##SBATCH --partition=build
#SBATCH --partition=lgagliardi-net
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=50
#SBATCH --error=sqd.err
 
#ulimit -s unlimited
#export OMP_NUM_THREADS=4
#export PYSCF_MAX_MEMORY=180000
#conda activate laslucj

#module load mkl/2023.1
#source ~/.bashrc     
conda activate lassqd

export PYTHONPATH=
export PYSCF_EXT_PATH=
export PYSCF=/project/lgagliardi/shared/Apps/pyscf
#/home/qjwang/.local/lib/python3.9/site-packages/pyscf
#export MRH=/project/lgagliardi/qjwang:/project/lgagliardi/qjwang/mrh
export MRH=/project/lgagliardi/shared/Apps/mrh
export PYTHONPATH=$MRH$PYSCF:$PYTHONPATH

python las_lucj.py > hardware_run_pdft.out
