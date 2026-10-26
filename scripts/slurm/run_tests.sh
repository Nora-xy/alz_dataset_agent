#!/bin/bash
#SBATCH --job-name=alz_tests
#SBATCH --partition=common
#SBATCH --output=/hpc/group/396-Alzheim26/alz_dataset_agent/logs/tests_%j.out
#SBATCH --error=/hpc/group/396-Alzheim26/alz_dataset_agent/logs/tests_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

source /hpc/group/396-Alzheim26/xl461/miniconda3/etc/profile.d/conda.sh
conda activate alz

cd /hpc/group/396-Alzheim26/alz_dataset_agent

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "PWD: $(pwd)"

echo "Python:"
which python
python --version

echo "Package versions:"
python - <<'PY'
import sys
print("executable:", sys.executable)
import pandas, numpy, pytest, openpyxl
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("pytest:", pytest.__version__)
print("openpyxl:", openpyxl.__version__)
PY

echo "Ensure editable install with catalog extra:"
python -m pip install -e ".[catalog]"

echo "Run pytest:"
python -m pytest -q

echo "Finished: $(date)"
