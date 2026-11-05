#!/bin/bash -l            
#SBATCH --job-name=ts 
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=90:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
#SBATCH --partition=lgagliardi-ld,lgagliardi-hm,lgagliardi-amd
##SBATCH --partition=build
##SBATCH --partition=lgagliardi-net
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=32
#SBATCH --error=sqd.err
 
 
#ulimit -s unlimited
#export OMP_NUM_THREADS=14
#export PYSCF_MAX_MEMORY=180000
#conda activate laslucj
conda activate lassqd
#module load mkl/2023.1
export PYTHONPATH=
export PYSCF_EXT_PATH=
export PYSCF=/project/lgagliardi/shared/Apps/pyscf
#/home/qjwang/.local/lib/python3.9/site-packages/pyscf
export LASLUCJ=/project/lgagliardi/qjwang/las-lucj/src                                                                
export MRH=/project/lgagliardi/qjwang:/project/lgagliardi/qjwang/mrh
export PYTHONPATH=$LASLUCJ:$MRH:$PYSCF:$PYTHONPATH

python inp_ts.py > testrun_2k.out
