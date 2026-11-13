#!/bin/bash
#SBATCH --job-name=gopd
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
#SBATCH --error=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.err
exec 2>&1

# The authors' own code, run on our pair before we trust our own version of it.
#
#   sbatch -J gopd_base  slurm_run.sh opd        # token-local OPD baseline
#   sbatch -J gopd_gamma slurm_run.sh gammaopd   # + discount, RBM, softsign, KL scaling
#
# Why not just our estimator: scripts/gammaopd.sh turns on five things --
# causal_opd, causal_gamma, use_reward_advantage (the paper's RBM, which mixes
# verifiable outcome reward into the advantage), scale_reverse_kl_by_abs_mean,
# and apply_softsign_to_opd_advantages -- and applies them in policy_loss, not
# in the advantage estimator. Ours implements the discount alone, which is
# their ablation, not their method.
set -u
ARM=${1:?"usage: sbatch slurm_run.sh <opd|gammaopd>"}
cd /home/doyoonkim/projects/GammaOPD

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate verl_opd

OPD=/home/doyoonkim/projects/OPD_thunlp
export STUDENT_MODEL=${STUDENT_MODEL:-$OPD/model/Qwen3-1.7B-SFT}
export TEACHER_MODEL=${TEACHER_MODEL:-$OPD/model/Qwen3-4B}
export TRAIN_FILE=${TRAIN_FILE:-$OPD/datasets/dapo-math-17k.parquet}
export VAL_FILE=${VAL_FILE:-$OPD/datasets/test_data/AIME24/test.parquet}

# Our box is 2 GPUs per job, not the 8 the defaults assume, and the response
# budget here is the 7168 the rest of this project uses so the numbers are
# comparable to the existing runs.
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/doyoonkim/projects/GammaOPD/outputs}

# This cluster's SLURM sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES.
# verl's worker setup treats that pair as a configuration error and raises
# before any model loads ("Please don't set ROCR_VISIBLE_DEVICES when
# HIP/CUDA_VISIBLE_DEVICES is set"). It is an AMD variable and means nothing on
# these NVIDIA nodes, so drop it before ray starts; the workers inherit from here.
unset ROCR_VISIBLE_DEVICES

export HF_HOME=/home/shared/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" logs "$OUTPUT_ROOT"

echo "=== GammaOPD official: arm=$ARM"
echo "  student=$STUDENT_MODEL"
echo "  teacher=$TEACHER_MODEL"
echo "  gpus=$N_GPUS_PER_NODE  batch=$TRAIN_BATCH_SIZE  resp_len=$MAX_RESPONSE_LENGTH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash "scripts/${ARM}.sh"
echo "##### END ($?) #####"
