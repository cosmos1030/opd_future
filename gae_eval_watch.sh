#!/bin/bash
# Evaluate each GAE checkpoint the moment it is complete, before anything else
# touches it. The previous pair was deleted before it was ever scored.
set -u
O=/home/doyoonkim/projects/OPD_thunlp
G=/home/doyoonkim/projects/GammaOPD
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "gae eval watcher up"
while true; do
    for d in "$O"/checkpoint/gae_*g_0.9375-lam_*/; do
        [ -d "$d" ] || continue
        lam=$(basename "$d" | grep -oE "lam_[0-9.]+")
        lr=$(basename "$d" | grep -oE "lr_[0-9e.-]+" || echo lr_1e-5)
        for step in 50 100 150; do
            a="$d/global_step_$step/actor"
            tag="gaeX_${lam}_${lr}_${step}"
            # fsdp_config.json is written last, so its presence means the shard
            # set is complete and safe to merge.
            [ -f "$a/fsdp_config.json" ] || continue
            [ -f "$HOME/gopd_watchdog/.gae_$tag" ] && continue
            hf="$G/outputs/hf/$tag"
            m=$(cd "$G" && sbatch --parsable -J "m_$tag" -p A100,A6000,3090 \
                 --export=ALL,CKPT_SRC="$a",TARGET="$hf" merge_any.sh)
            e=$(cd "$O" && sbatch --parsable -J "ev_$tag" -p PRO6000,A100,A6000,H200 --gres=gpu:2 --mem=200G --cpus-per-task=12 \
                 --dependency=afterok:$m --export=ALL,CKPT="$hf" eval_gopd_ckpt.sh)
            touch "$HOME/gopd_watchdog/.gae_$tag"
            say "queued $tag  merge=$m eval=$e  ($a)"
        done
    done
    sleep 120
done
