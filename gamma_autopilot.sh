#!/bin/bash
# Keep the gamma-OPD sweep running unattended.
#
# The sweep is the experiment the archived work never ran. It closed on depth 2
# -- a term that correlates 0.63 with the 1-step one and is 2/3 its size -- and
# concluded temporal credit was inert. The lagwise data says gamma=0.99 is a
# different animal: 33x the action spread, correlation 0.41 with 1-step, SNR
# still 5.0. gamma-OPD (arXiv 2609.16937) reports +4.92% relative on exactly
# this 4B->1.7B pair, so the sweep either reproduces that or contradicts it,
# and either outcome is worth the GPUs.
#
# gamma=0 is token_reward_direct and gamma=1 is sequence-level, so the three
# known methods sit on one axis and the run-to-run noise is measured in the
# same pipeline rather than quoted from another paper.
set -u
D=/home/doyoonkim/projects/OPD_thunlp
STATE=$HOME/eval_watchdog
LOG_TAG=gamma
GAMMAS=${GAMMAS:-"0.0 0.9 0.99 1.0"}
FULL_STEPS=${FULL_STEPS:-139}     # one epoch at train_batch_size 128, ~5h on 2 GPUs
MAX_RETRY=3
mkdir -p "$STATE"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

running_or_pending () {   # $1 = job name
    squeue -u "$USER" -h -o "%j" | grep -qx "$1"
}

smoke_ok () {             # $1 = gamma; smoke log must show the estimator firing
    local g=$1 f
    f=$(ls -t "$D"/logs/smoke/gamma_${g}_*.log 2>/dev/null | head -1)
    [ -n "$f" ] && grep -q "\[gamma-opd\]" "$f"
}

submit_full () {          # $1 = gamma
    local g=$1 name=gfull_$1
    running_or_pending "$name" && return 0
    [ -f "$STATE/.gdone_$g" ] && return 0
    local n; n=$(cat "$STATE/.gretry_$g" 2>/dev/null || echo 0)
    if [ "$n" -ge "$MAX_RETRY" ]; then return 0; fi
    sbatch --export=ALL,ARM=token_reward_gamma,ADV_ESTIMATOR=token_reward_gamma,OPD_GAMMA=$g,MINI_BATCH_SIZE=32,SMOKE_STEPS=$FULL_STEPS \
           -J "$name" --partition=H200 --time=12:00:00 "$D/smoke_gated2step.sh" \
        && { echo $((n+1)) > "$STATE/.gretry_$g"; say "submitted full run gamma=$g (attempt $((n+1)))"; }
}

say "gamma autopilot up: gammas=[$GAMMAS] full_steps=$FULL_STEPS"
while true; do
    # 1. promote each gamma to a full run once its smoke has proved the estimator
    for g in $GAMMAS; do
        if smoke_ok "$g"; then submit_full "$g"; fi
    done

    # 2. judge finished full runs; a died run is invisible to a queue check
    for g in $GAMMAS; do
        name=gfull_$g
        running_or_pending "$name" && continue
        jid=$(sacct -X -n -S today -u "$USER" --name="$name" -o JobID 2>/dev/null | tail -1 | tr -d ' ')
        [ -z "$jid" ] && continue
        st=$(sacct -X -j "$jid" -o State -n 2>/dev/null | head -1 | awk '{print $1}')
        el=$(sacct -X -j "$jid" -o ElapsedRaw -n 2>/dev/null | head -1 | tr -d ' ')
        case "$st" in
            COMPLETED)
                if [ "${el:-0}" -gt 1800 ]; then
                    [ -f "$STATE/.gdone_$g" ] || say "gamma=$g finished (${el}s)"
                    touch "$STATE/.gdone_$g"
                else
                    say "gamma=$g exited after ${el}s -- too short to be a real run, retrying"
                    submit_full "$g"
                fi ;;
            FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT|CANCELLED*)
                say "gamma=$g ended $st -- retrying"; submit_full "$g" ;;
        esac
    done

    # 3. never leave the partition idle: H200 free and nothing of ours queued
    idle=$(sinfo -h -p H200 -o "%C" | awk -F/ '{print $2}')
    mine=$(squeue -u "$USER" -h -t RUNNING,PENDING | wc -l)
    if [ "${idle:-0}" -gt 0 ] && [ "$mine" -lt 4 ]; then
        for g in $GAMMAS; do
            [ -f "$STATE/.gdone_$g" ] && continue
            submit_full "$g"
        done
    fi
    sleep 300
done
