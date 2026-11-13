#!/bin/bash
# Kill and resubmit training jobs whose log stopped moving.
#
# Two jobs sat for 7h12m on an H200 node with 0% GPU after their raylet died:
# SLURM still counted them as RUNNING, so nothing in the queue advanced and I
# only noticed when asked. sacct and squeue both say RUNNING for a dead ray
# cluster; the only honest signal is whether the log file is still growing.
set -u
G=/home/doyoonkim/projects/GammaOPD
STALL_MIN=${STALL_MIN:-25}
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "stall watch up (threshold ${STALL_MIN}m)"
while true; do
    while read -r jid name; do
        [ -z "${jid:-}" ] && continue
        case "$name" in s50_*|opdgae_*|gs_*) ;; *) continue ;; esac
        f=$(ls -t "$G"/logs/${name}_${jid}.log 2>/dev/null | head -1)
        [ -f "$f" ] || continue
        age=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 60 ))
        [ "$age" -lt "$STALL_MIN" ] && continue
        say "STALLED: $name ($jid) log idle ${age}m -- cancelling"
        scancel "$jid"
        touch "$HOME/gopd_watchdog/.stalled_${name}"
    done < <(squeue -u "$USER" -h -t RUNNING -o "%i %j")
    sleep 300
done
