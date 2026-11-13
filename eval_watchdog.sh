#!/bin/bash
# Resubmit merge/eval jobs that end without producing a number.
#
# Both harnesses here swallow failures: run_training and eval_qwen3_baseline.sh
# both end with `ray stop`, whose exit code overwrites python's, so a crashed
# job reports COMPLETED and prints "##### END (0) #####". sacct is therefore
# useless as a health check -- the only reliable signal is whether the log
# actually contains the metric.
#
# Checked every 3 minutes:
#   eval job  -> its log must contain val-core/MATH-500/acc/mean@16
#   merge job -> its target_dir must contain config.json
# A job that is gone from the queue and has neither gets resubmitted, up to
# MAX_RETRY, each retry with bounded memory/CPU so a 2-GPU eval cannot hold a
# whole training node.
set -u
G=/home/doyoonkim/projects/GammaOPD
O=/home/doyoonkim/projects/OPD_thunlp
STATE=$HOME/gopd_watchdog
MAX_RETRY=${MAX_RETRY:-3}
mkdir -p "$STATE"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# one line per target: <tag> <arm> <step>
TARGETS_FILE=${TARGETS_FILE:-$STATE/targets.txt}
touch "$TARGETS_FILE"

queued () { squeue -u "$USER" -h -o "%j" | grep -qx "$1"; }

has_metric () {   # $1 = eval job name
    local f
    f=$(ls -t "$O"/logs/"$1"_*.out 2>/dev/null | head -1)
    [ -n "$f" ] && grep -aq "val-core/MATH-500/acc/mean@16" "$f"
}

retries () { cat "$STATE/.r_$1" 2>/dev/null || echo 0; }
bump ()    { echo $(( $(retries "$1") + 1 )) > "$STATE/.r_$1"; }

diagnose () {      # $1 = eval job name -> prints a reason tag
    local f
    f=$(ls -t "$O"/logs/"$1"_*.out 2>/dev/null | head -1)
    [ -z "$f" ] && { echo none; return; }
    if grep -aq "less than desired GPU memory utilization" "$f"; then echo smallgpu
    elif grep -aq "Raylet is terminated\|killed by signal: Killed\|ActorUnavailableError" "$f"; then echo hostoom
    elif grep -aq "CUDA out of memory" "$f"; then echo cudaoom
    else echo unknown; fi
}

submit_pair () {   # $1 tag, $2 arm, $3 step
    local tag=$1 arm=$2 step=$3
    local hf=$G/outputs/hf/${arm}_step${step}
    local m_job=""
    if [ ! -f "$hf/config.json" ]; then
        m_job=$(cd "$G" && sbatch --parsable -J "m_$tag" -p A100,A6000,3090 \
            --export=ALL,ARM="$arm",STEP="$step" merge_and_eval.sh)
    fi
    local dep=()
    [ -n "$m_job" ] && dep=(--dependency=afterok:"$m_job")
    # React to WHY it died instead of throwing it at the same node again.
    # A6000 has 47.4 GiB and the eval asks 0.9 of it; that fails on startup
    # every single time, which is how three retries burned for nothing.
    local why parts util
    why=$(diagnose "ev_$tag")
    parts=PRO6000,A100,H200
    util=0.9
    case "$why" in
        smallgpu) parts=H200 ; util=0.7 ;;
        cudaoom)  parts=H200 ; util=0.6 ;;
        hostoom)  parts=H200,PRO6000 ;;
    esac
    local e_job
    e_job=$(cd "$O" && sbatch --parsable -J "ev_$tag" -p "$parts" --mem=400G --cpus-per-task=24 \
        "${dep[@]}" --export=ALL,CKPT="$hf",GPU_UTIL="$util" eval_gopd_ckpt.sh)
    bump "$tag"
    say "resubmitted $tag (arm=$arm step=$step) reason=$why parts=$parts util=$util merge=${m_job:-skip} eval=$e_job attempt $(retries "$tag")"
}

say "watchdog up: targets=$TARGETS_FILE max_retry=$MAX_RETRY"
while true; do
    while read -r tag arm step; do
        [ -z "${tag:-}" ] && continue
        case "$tag" in \#*) continue ;; esac
        has_metric "ev_$tag" && continue
        queued "ev_$tag" && continue
        queued "m_$tag" && continue
        # the training run may not have reached this step yet
        [ -d "$G/outputs/checkpoints/$arm/global_step_$step/actor" ] || continue
        [ "$(retries "$tag")" -ge "$MAX_RETRY" ] && continue
        submit_pair "$tag" "$arm" "$step"
    done < "$TARGETS_FILE"
    sleep 180
done
