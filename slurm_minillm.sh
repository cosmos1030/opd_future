#!/bin/bash
#SBATCH --job-name=mllm
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
exec 2>&1
# OPD with MiniLLM's recipe, as far as verl can express it.
#
# Matched (scripts/qwen2.5/minillm/train_1.5B_7B.sh + arguments.py defaults):
#   gamma 0.95, ppo_epochs 4, advantage whitening,
#   reward_scaling 0.5, cliprange_reward 100
#   reward itself already matches: MiniLLM's t_rewards + ent_rewards is
#   log p_T(a) - log p_S(a), the same per-token quantity verl uses.
#
# NOT matched, and why:
#   single_step_reg  - excluded by instruction
#   kd_ratio 0.5     - excluded by instruction (needs a pretrain corpus)
#   teacher_mixed_alpha 0.2 - structurally impossible here: it samples from a
#     token-level mixture of student and teacher and corrects with an importance
#     weight, and vLLM has no hook to blend another model's logits mid-generation.
#
# lr is swept rather than taken from MiniLLM's 5e-6, which was tuned for a
# different pair at max-length 512.
set -u
cd /home/doyoonkim/projects/GammaOPD
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gopd
unset ROCR_VISIBLE_DEVICES
_o=$(( (SLURM_JOB_ID % 50) * 200 ))
export MASTER_PORT=$(( 20000 + _o + 1 ))
export RAY_DASHBOARD_PORT=$(( 20000 + _o + 2 ))
export RAY_DASHBOARD_AGENT_GRPC_PORT=$(( 20000 + _o + 3 ))
export RAY_DASHBOARD_AGENT_LISTEN_PORT=$(( 20000 + _o + 4 ))
export RAY_METRICS_EXPORT_PORT=$(( 20000 + _o + 5 ))
export RAY_RUNTIME_ENV_AGENT_PORT=$(( 20000 + _o + 6 ))
export RAY_NODE_MANAGER_PORT=$(( 30000 + _o + 1 ))
export RAY_OBJECT_MANAGER_PORT=$(( 30000 + _o + 2 ))
_w=$(( (SLURM_JOB_ID % 10) * 2500 ))
export RAY_MIN_WORKER_PORT=$(( 35000 + _w ))
export RAY_MAX_WORKER_PORT=$(( 35000 + _w + 2400 ))
export RAY_OBJECT_STORE_BYTES=40000000000

OPD=/home/doyoonkim/projects/OPD_thunlp
export STUDENT_MODEL=$OPD/model/Qwen3-1.7B-SFT
export TEACHER_MODEL=$OPD/model/Qwen3-4B
export TRAIN_FILE=$OPD/datasets/dapo-math-17k.parquet
export VAL_FILE=/home/doyoonkim/projects/GammaOPD/data/aime24_val.parquet
export N_GPUS_PER_NODE=4
export TRAIN_BATCH_SIZE=128 PPO_MINI_BATCH_SIZE=128
export MAX_RESPONSE_LENGTH=8192 MAX_PROMPT_LENGTH=2048
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
export SAVE_FREQ=${SAVE_FREQ:-50} TEST_FREQ=-1 ROLLOUT_TP=1
export MAX_ACTOR_CKPT_TO_KEEP=2
export OUTPUT_ROOT=/home/doyoonkim/projects/GammaOPD/outputs
export HF_HOME=/home/shared/huggingface
export TOKENIZERS_PARALLELISM=false VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}; mkdir -p "$TMPDIR" logs

echo "=== MiniLLM-matched OPD  gamma=${GAMMA:-0.95} lr=${ACTOR_LR:?} ppo_epochs=4"
bash scripts/opd.sh \
  actor_rollout_ref.actor.policy_loss.causal_opd=True \
  actor_rollout_ref.actor.policy_loss.causal_gamma="${GAMMA:-0.95}" \
  actor_rollout_ref.actor.policy_loss.minillm_whiten=True \
  actor_rollout_ref.actor.policy_loss.minillm_reward_scaling=0.5 \
  actor_rollout_ref.actor.policy_loss.minillm_cliprange_reward=100 \
  actor_rollout_ref.actor.ppo_epochs=4
echo "##### END ($?) #####"
