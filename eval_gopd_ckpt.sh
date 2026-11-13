#!/bin/bash
#SBATCH --job-name=ev_gopd
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/home/doyoonkim/projects/OPD_thunlp/logs/%x_%j.out
exec 2>&1
# Score a GammaOPD checkpoint with the SAME harness that produced the numbers
# already on record for this student/teacher pair -- verl validation on
# MATH-500 at val_kwargs.n=16. Re-scoring with a different harness would make
# the reproduction incomparable to:
#
#   SFT base (no training)        0.6658
#   token_reward_direct baseline  0.6811
#   baseline, batch-size matched  0.6724
#   GAE gamma .99 lam .95         0.6721
#   token_reward_direct_critic    0.6740
#
#   sbatch -J ev_opd150 --export=ALL,CKPT=/home/doyoonkim/projects/GammaOPD/outputs/hf/small_opd_step150 eval_gopd_ckpt.sh
set -u
cd /home/doyoonkim/projects/OPD_thunlp
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate verl_opd
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/home/shared/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}; mkdir -p "$TMPDIR" logs

export GPU_UTIL=${GPU_UTIL:-0.9}
export ACTOR_MODEL_PATH=${CKPT:?}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
bash eval_qwen3_baseline.sh
echo "##### END ($?) #####"
