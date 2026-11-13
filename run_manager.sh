#!/bin/bash
# One manager for the whole sweep: every run is either done, healthy, or gets
# (re)submitted. Replaces the patchwork that let two jobs sit dead for 7h12m.
#
# Per cycle, for each spec line:
#   1. result already in an eval log      -> done, skip
#   2. queued or running and log moving   -> healthy, skip
#   3. running but log idle > STALL_MIN   -> cancel, then resubmit next cycle
#   4. otherwise                          -> submit training, or merge+eval if
#                                            the checkpoint is already there
#
# squeue and sacct both report RUNNING for a job whose ray cluster has died, so
# log mtime is the only trustworthy liveness signal.
set -u
G=/home/doyoonkim/projects/GammaOPD
O=/home/doyoonkim/projects/OPD_thunlp
S=$HOME/gopd_watchdog
VF=$G/data/aime24_val.parquet
SPEC=${SPEC:-$S/runspec.txt}
PART=${PART:-PRO6000,H200}
STALL_MIN=${STALL_MIN:-25}
MAX_TRY=${MAX_TRY:-4}
mkdir -p "$S"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

result_of () { # $1 arm -> prints metric or empty
    local f
    f=$(ls -t "$O"/logs/ev_$1_step50_*.out 2>/dev/null | head -1)
    [ -n "$f" ] && grep -aoE "acc/mean@16:np\.float64\([0-9.]+\)" "$f" 2>/dev/null | tail -1 | sed 's/.*float64(//;s/)//'
}
jid_of () { squeue -u "$USER" -h -o "%i %j" | awk -v n="$1" '$2==n{print $1; exit}'; }
tries () { cat "$S/.try_$1" 2>/dev/null || echo 0; }
bump () { echo $(( $(tries "$1") + 1 )) > "$S/.try_$1"; }

submit_train () { # $1 arm  $2 kind  $3 gamma  $4 lr  $5 optim
    local arm=$1 kind=$2 gam=$3 lr=$4 opt=$5 extra="" oimpl="rad.optim" oname="RADAR"
    [ "$opt" = "adamw" ] && { oimpl="torch.optim"; oname="AdamW"; }
    if [ "$kind" = "gae1" ] || [ "$kind" = "gae0" ]; then
        local lam=1.0; [ "$kind" = "gae0" ] && lam=0.0
        (cd "$G" && sbatch --parsable -J "$arm" -p H200 \
            --export=ALL,LAM=$lam,ACTOR_LR=$lr,EXPERIMENT_NAME=$arm,GAE_GAMMA=$gam slurm_opdgae.sh)
        return
    fi
    [ "$gam" != "none" ] && extra="actor_rollout_ref.actor.policy_loss.causal_opd=True actor_rollout_ref.actor.policy_loss.causal_gamma=$gam"
    # MiniLLM-matched arm: same per-token OPD reward, but their advantage
    # pipeline -- whiten, reward_scaling 0.5, cliprange_reward 100, 4 ppo epochs.
    # single_step_reg and kd_ratio are deliberately left out.
    if [ "$kind" = "mllm" ]; then
        extra="$extra actor_rollout_ref.actor.policy_loss.minillm_whiten=True actor_rollout_ref.actor.policy_loss.minillm_reward_scaling=0.5 actor_rollout_ref.actor.policy_loss.minillm_cliprange_reward=100 actor_rollout_ref.actor.ppo_epochs=4"
    fi
    (cd "$G" && sbatch --parsable -J "$arm" -p "$PART" --gres=gpu:4 --mem=300G --time=10:00:00 --cpus-per-task=40 \
        --export=ALL,N_GPUS_PER_NODE=4,EXPERIMENT_NAME=$arm,VAL_FILE=$VF,EXTRA_ARGS="$extra",ACTOR_LR=$lr,SAVE_FREQ=50,TOTAL_TRAINING_STEPS=50,TEST_FREQ=50,MAX_ACTOR_CKPT_TO_KEEP=2,OPTIM_IMPL=$oimpl,OPTIM_NAME=$oname \
        slurm_small.sh opd)
}

submit_eval () { # $1 arm
    local arm=$1 hf=$G/outputs/hf/${arm}_step50 m="" dep=()
    if [ ! -f "$hf/config.json" ]; then
        m=$(cd "$G" && sbatch --parsable -J "m_$arm" -p A100,A6000,3090 \
            --export=ALL,ARM="$arm",STEP=50 merge_and_eval.sh)
        dep=(--dependency=afterok:"$m")
    fi
    (cd "$O" && sbatch --parsable -J "ev_${arm}_step50" -p PRO6000,A100,H200 --gres=gpu:2 \
        --mem=200G --cpus-per-task=12 "${dep[@]}" --export=ALL,CKPT="$hf" eval_gopd_ckpt.sh)
}

say "run manager up  spec=$SPEC  stall=${STALL_MIN}m  max_try=$MAX_TRY"
while true; do
    while read -r arm kind gam lr opt; do
        case "${arm:-}" in ""|\#*) continue ;; esac
        [ -n "$(result_of "$arm")" ] && continue

        ej=$(jid_of "ev_${arm}_step50"); [ -n "$ej" ] && continue
        mj=$(jid_of "m_$arm");           [ -n "$mj" ] && continue

        tj=$(jid_of "$arm")
        if [ -n "$tj" ]; then
            f=$(ls -t "$G"/logs/${arm}_${tj}.log 2>/dev/null | head -1)
            if [ -f "$f" ]; then
                age=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 60 ))
                if [ "$age" -ge "$STALL_MIN" ]; then
                    say "STALL $arm ($tj) idle ${age}m -> cancel"; scancel "$tj"
                fi
            fi
            continue
        fi

        if [ -f "$G/outputs/checkpoints/$arm/global_step_50/actor/fsdp_config.json" ]; then
            say "eval $arm -> $(submit_eval "$arm")"
        else
            n=$(tries "$arm")
            if [ "$n" -lt "$MAX_TRY" ]; then
                bump "$arm"; say "train $arm (try $((n+1))) -> $(submit_train "$arm" "$kind" "$gam" "$lr" "$opt")"
            fi
        fi
    done < "$SPEC"
    sleep 240
done
