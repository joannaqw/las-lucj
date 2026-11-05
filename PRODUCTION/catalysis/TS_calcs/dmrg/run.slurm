#!/bin/bash -l
#SBATCH --job-name=tsdmrg1k
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --partition=lgagliardi-ld,lgagliardi-hm
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=16
#SBATCH --error=sqd.err

set -e
unset PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate /home/qjwang/.conda/envs/lassqd

export PYSCF_EXT_PATH=/project/lgagliardi/shared/Apps/pyscf_ext/pyscf-forge
LASSQD_PYTHON="$CONDA_PREFIX/bin/python"

BLOCK2_MKL="$CONDA_PREFIX/lib/python3.11/site-packages/block2.libs/libmkl_def.so.1"
if [ ! -r "$BLOCK2_MKL" ]; then
    echo "Missing Block2 MKL library: $BLOCK2_MKL" >&2
    exit 1
fi

"$LASSQD_PYTHON" - <<'PY'
import sys
import pyscf
from pyscf import dmrgscf

print("Python executable:", sys.executable)
print("PySCF file:", pyscf.__file__)
print("PySCF paths:", list(pyscf.__path__))
print("dmrgscf file:", dmrgscf.__file__)
PY

/usr/bin/time --verbose "$LASSQD_PYTHON" input.py > dmrg_1k.out
