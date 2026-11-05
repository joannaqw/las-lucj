#!/bin/bash -l            
#SBATCH --job-name=intdmrg500
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=90:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
#SBATCH --partition=lgagliardi-ld,lgagliardi-hm
##SBATCH --partition=build
##SBATCH --partition=lgagliardi-net
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=16
#SBATCH --error=sqd.err


set -e

# Remove inherited paths that point to the shared/old PySCF.
unset PYTHONPATH

# Initialize Conda explicitly in this non-interactive shell.
eval "$(conda shell.bash hook)"
conda activate /home/qjwang/.conda/envs/lassqd
module load mkl
# Add pyscf-forge through PySCF's supported extension mechanism.
export PYSCF_EXT_PATH=/project/lgagliardi/shared/Apps/pyscf_ext/pyscf-forge

LASSQD_PYTHON=/home/qjwang/.conda/envs/lassqd/bin/python

# Print and verify the environment before starting the expensive calculation.
"$LASSQD_PYTHON" - <<'PY'
import sys
import pyscf
from pyscf import dmrgscf

print("Python executable:", sys.executable)
print("PySCF file:", pyscf.__file__)
print("PySCF paths:", list(pyscf.__path__))
print("dmrgscf file:", dmrgscf.__file__)
PY

/usr/bin/time --verbose "$LASSQD_PYTHON" input.py > dmrg_500.out


