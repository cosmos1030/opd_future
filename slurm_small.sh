#!/bin/bash
#SBATCH --job-name=gopds
#SBATCH --qos=normal
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
#SBATCH --error=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.err
exec 2>&1

# The authors' code and the authors' method flags, at 1/16 the token budget.
#
# Nothing about gamma-OPD itself is touched: scripts/gammaopd.sh still sets
# causal_opd, causal_gamma, use_reward_advantage, scale_reverse_kl_by_abs_mean
# and apply_softsign_to_opd_advantages, and the optimizer is still RADAR. Only
# the three knobs that set the bill are turned down, and both arms get the same
# ones, so "gammaopd beats opd" is still the thing being tested.
#
#   batch 1024 -> 128, response 16384 -> 8192, steps 150 kept
#   official 150*1024*16384 = 2.5B generated tokens; here 157M.
#
# ROLLOUT_TP drops to 1 because a 1.7B student does not need 4-way tensor
# parallel on an H200 -- that is throughput, not method.
#
#   sbatch -J gs_base  -p H200    slurm_small.sh opd
#   sbatch -J gs_gamma -p PRO6000 slurm_small.sh gammaopd
set -u
ARM=${1:?"usage: sbatch slurm_small.sh <opd|gammaopd>"}
cd /home/doyoonkim/projects/GammaOPD

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gopd

OPD=/home/doyoonkim/projects/OPD_thunlp
export STUDENT_MODEL=${STUDENT_MODEL:-$OPD/model/Qwen3-1.7B-SFT}
export TEACHER_MODEL=${TEACHER_MODEL:-$OPD/model/Qwen3-4B}
export TRAIN_FILE=${TRAIN_FILE:-$OPD/datasets/dapo-math-17k.parquet}
export VAL_FILE=${VAL_FILE:-$OPD/datasets/test_data/AIME24/test.parquet}

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TEST_FREQ=${TEST_FREQ:-10}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/doyoonkim/projects/GammaOPD/outputs}

# This cluster's SLURM sets ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES.
# verl's worker setup treats that pair as a configuration error and raises
# before any model loads ("Please don't set ROCR_VISIBLE_DEVICES when
# HIP/CUDA_VISIBLE_DEVICES is set"). It is an AMD variable and means nothing on
# these NVIDIA nodes, so drop it before ray starts; the workers inherit from here.
unset ROCR_VISIBLE_DEVICES

# Two arms can land on the same node (one per NVLink island), so every ray
# port must be unique per job or the second one dies on a bound port. Derive
# them from the job id, and keep the worker range far from the component ports
# -- launcher_common rejects an overlap outright.
_o=$(( (SLURM_JOB_ID % 50) * 200 ))
export MASTER_PORT=${MASTER_PORT:-$(( 20000 + _o + 1 ))}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-$(( 20000 + _o + 2 ))}
export RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-$(( 20000 + _o + 3 ))}
export RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$(( 20000 + _o + 4 ))}
export RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-$(( 20000 + _o + 5 ))}
export RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-$(( 20000 + _o + 6 ))}
_w=$(( (SLURM_JOB_ID % 10) * 2500 ))
export RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-$(( 35000 + _w ))}
export RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$(( 35000 + _w + 2400 ))}
export RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-$(( 30000 + _o + 1 ))}
export RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-$(( 30000 + _o + 2 ))}
export RAY_OBJECT_STORE_BYTES=${RAY_OBJECT_STORE_BYTES:-40000000000}

export HF_HOME=/home/shared/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" logs "$OUTPUT_ROOT"

echo "=== GammaOPD (official code, scaled): arm=$ARM"
echo "  student=$STUDENT_MODEL  teacher=$TEACHER_MODEL"
echo "  gpus=$N_GPUS_PER_NODE batch=$TRAIN_BATCH_SIZE resp=$MAX_RESPONSE_LENGTH steps=$TOTAL_TRAINING_STEPS tp=$ROLLOUT_TP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "scripts/${ARM}.sh" ${EXTRA_ARGS:-}
echo "##### END ($?) #####"
