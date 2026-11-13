#!/bin/bash
#SBATCH --job-name=gopd
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
#SBATCH --error=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.err
exec 2>&1

# The authors' code at the authors' settings. Nothing here overrides batch
# size, response length, optimizer or GPU count -- scripts/lib/launcher_common.sh
# defaults to 8 GPUs, train_batch_size 1024 and max_response_length 16384, and
# those defaults are the point. Only the four paths the README requires are set.
#
#   sbatch -J gopd_base  slurm_official.sh opd
#   sbatch -J gopd_gamma slurm_official.sh gammaopd
set -u
ARM=${1:?"usage: sbatch slurm_official.sh <opd|gammaopd>"}
cd /home/doyoonkim/projects/GammaOPD

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gopd

OPD=/home/doyoonkim/projects/OPD_thunlp
export STUDENT_MODEL=${STUDENT_MODEL:-$OPD/model/Qwen3-1.7B-SFT}
export TEACHER_MODEL=${TEACHER_MODEL:-$OPD/model/Qwen3-4B}
export TRAIN_FILE=${TRAIN_FILE:-$OPD/datasets/dapo-math-17k.parquet}
export VAL_FILE=${VAL_FILE:-$OPD/datasets/test_data/AIME24/test.parquet}

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
mkdir -p "$TMPDIR" logs

echo "=== GammaOPD official defaults: arm=$ARM"
echo "  student=$STUDENT_MODEL"
echo "  teacher=$TEACHER_MODEL"
echo "  train=$TRAIN_FILE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "scripts/${ARM}.sh"
echo "##### END ($?) #####"
