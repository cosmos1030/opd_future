#!/bin/bash
#SBATCH --job-name=dx-ev
#SBATCH --output=/home/doyoonkim/projects/opd_horizon/logs/dx-ev_%j.log
#SBATCH --error=/home/doyoonkim/projects/opd_horizon/logs/dx-ev_%j.log
#SBATCH --account=users
#SBATCH --partition=PRO6000,A100,H200
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=10:00:00
# Author eval settings (paper Tables 8/9): aime24 aime25 hmmt25 amc23 minerva, n=64.
# CKPT unset evaluates the base model, which is the reference the paper improves on.
set -x
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
export HF_HOME=/home/shared/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
cd /home/doyoonkim/projects/distIL
M=/home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507
if [ -n "${CKPT:-}" ]; then
  [ -d "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 17; }
  BASE_MODEL=$M CHECKPOINT="$CKPT" OUTPUT_DIR="${OUT:-eval_results}" bash scripts/math/eval_math.sh
else
  BASE_MODEL=$M OUTPUT_DIR="${OUT:-eval_results_base}" bash scripts/math/eval_math.sh
fi
echo "EXIT=$?"
