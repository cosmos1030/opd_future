#!/bin/bash
#SBATCH --job-name=dx-r4
#SBATCH --output=/home/doyoonkim/projects/opd_horizon/logs/dx-r4_%j.log
#SBATCH --error=/home/doyoonkim/projects/opd_horizon/logs/dx-r4_%j.log
#SBATCH --account=users
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
# DistIL math reproduction, author settings unchanged (scripts/math/run_distil_4b.sh).
# This is the baseline for the one-line ablation: OPSD puts ce_per_token where SDPO
# puts student_log_probs in the REINFORCE term (opsd_trainer.py:554 vs
# core_algos.py:1278), so the math path optimizes something other than the paper's
# Proposition 1. Reproducing as-is first gives that comparison a reference point.
set -x
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
export HF_HOME=/home/shared/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
cd /home/doyoonkim/projects/distIL
MODEL=/home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507 \
OUTPUT_DIR=/home/doyoonkim/projects/distIL/outputs/repro_distil_4b \
NUM_PROCESSES=4 \
  bash scripts/math/run_distil_4b.sh
echo "EXIT=$?"
