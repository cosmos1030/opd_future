#!/bin/bash
#SBATCH --job-name=gae
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=900G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/doyoonkim/projects/OPD_thunlp/logs/%x_%j.out
exec 2>&1
# OPD reward -> token_level_rewards -> verl GAE + critic. gamma fixes the
# horizon; lam is how much the critic bootstraps:
#   lam=1.0  discounted MC return, critic only as a baseline
#   lam=0.95 mild bootstrap (what the earlier 0.6721 run used, at gamma .99)
#   lam=0.0  full one-step bootstrap, the critic carries everything
# The offline fidelity test put Adv-R2 at -2.80, which predicts this gets worse
# as lam falls. LOG_PROB_TOP_K stays 0 so the per-token reward is the sampled
# token, matching the GammaOPD horizon runs.
set -u
cd /home/doyoonkim/projects/OPD_thunlp
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate verl_opd
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/home/shared/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}; mkdir -p "$TMPDIR" logs
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export LOG_PROB_TOP_K=0
bash on_policy_distillation_qwen3_gae.sh
echo "##### END ($?) #####"
