#!/bin/bash
#SBATCH --job-name=opdgae
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
#SBATCH --error=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.err
exec 2>&1
# GAE on the OPD reward, inside the SAME codebase as the horizon sweep.
#
# The horizon arms (s50_h16_*) run opd.sh with causal_opd=True, i.e. a discounted
# MC suffix return and no critic. This arm feeds the identical per-token reward
# into verl's GAE with a critic, so the only difference from s50_h16 is the
# estimator -- same RADAR optimizer, same shuffle, same batch, same data. That
# was impossible across the OPD_thunlp runs, where optimizer and shuffle differ
# too and no lambda setting could separate critic from codebase.
#
#   LAM=1.0 -> discounted MC return, critic as baseline only
#   LAM=0.0 -> one-step bootstrap, critic carries the future
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
export MAX_ACTOR_CKPT_TO_KEEP=2 MAX_CRITIC_CKPT_TO_KEEP=1
export OUTPUT_ROOT=/home/doyoonkim/projects/GammaOPD/outputs
export HF_HOME=/home/shared/huggingface
export TOKENIZERS_PARALLELISM=false VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}; mkdir -p "$TMPDIR" logs

echo "=== OPD+GAE  lam=${LAM:?}  gamma=${GAE_GAMMA:-0.9375}  lr=${ACTOR_LR:-1e-5}"
bash scripts/opd.sh \
  algorithm.adv_estimator=gae \
  +algorithm.opd_reward_as_token_level=True \
  algorithm.gamma="${GAE_GAMMA:-0.9375}" \
  algorithm.lam="${LAM}" \
  actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=False \
  critic.enable=True \
  critic.model.path="$STUDENT_MODEL" \
  critic.model.use_remove_padding=True \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.optim.lr=1e-5
echo "##### END ($?) #####"
