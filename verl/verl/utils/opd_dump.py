"""Dump the exact tensors that define the OPD distillation reward.

Enabled by setting OPD_DUMP_DIR. On the first call each rank writes one .pt
file and then, unless OPD_DUMP_CONTINUE=1, the process exits so the job stops
right after the reward phase instead of paying for adv + actor update.

Why this exists: the offline branching experiment has to compute the same
per-candidate reward the trainer optimizes. Reading the code pins down the
formula but not the plumbing around it -- logit temperature, the prompt/response
alignment, where student_top_k_ids comes from, whether top-k log-probs are a
full-vocab log_softmax gathered at k or renormalized within the top-k set. Those
are silent failure modes, so we dump the real tensors once and diff the offline
reimplementation against them.
"""

import os

import torch


_DONE = False


def _cpu(x):
    # clone() is load-bearing: several of these tensors are views over shared
    # storage with differing dtypes, which torch.save refuses to serialise.
    if isinstance(x, torch.Tensor):
        x = x.detach().to(torch.float32) if x.is_floating_point() else x.detach()
        return x.cpu().clone().contiguous()
    return x


def maybe_dump(data, rm_scores, kl_val, norm_weights, extras=None):
    """Write one snapshot of the reward computation, then stop the run.

    Args:
        data: the DataProto handed to compute_distillation_reward.
        rm_scores: the (B, T, K) reward the actor will consume as advantages.
        kl_val: the pre-weighting (B, T, K) log-ratio term.
        norm_weights: the (B, T, K) weights multiplied into kl_val.
        extras: any additional named tensors worth capturing.
    """
    global _DONE
    out_dir = os.environ.get("OPD_DUMP_DIR")
    if not out_dir or _DONE:
        return

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    os.makedirs(out_dir, exist_ok=True)

    # Everything needed to recompute rm_scores from scratch offline, plus the
    # raw ids so the same positions can be re-scored with a plain HF forward.
    wanted = [
        "input_ids",
        "responses",
        "attention_mask",
        "position_ids",
        "response_mask",
        "student_top_k_ids",
        "student_top_k_log_probs",
        "teacher_on_student_log_probs",
        "teacher_top_k_ids",
        "teacher_top_k_log_probs",
        "overlap_mask",
        "teacher_in_student_mask",
        "old_log_probs",
    ]
    batch = {k: _cpu(data.batch[k]) for k in wanted if k in data.batch}

    payload = {
        "batch": batch,
        "meta_info": {k: v for k, v in data.meta_info.items() if isinstance(v, (int, float, str, bool))},
        "rm_scores": _cpu(rm_scores),
        "kl_val": _cpu(kl_val),
        "norm_weights": _cpu(norm_weights),
        "extras": {k: _cpu(v) for k, v in (extras or {}).items() if v is not None},
    }

    path = os.path.join(out_dir, f"reward_dump_rank{rank}.pt")
    torch.save(payload, path)
    _DONE = True
    print(f"[opd_dump] wrote {path}", flush=True)
    for k, v in batch.items():
        print(f"[opd_dump]   batch/{k}: {tuple(v.shape)} {v.dtype}", flush=True)
    print(f"[opd_dump]   rm_scores: {tuple(payload['rm_scores'].shape)}", flush=True)
    print(f"[opd_dump]   meta_info: {payload['meta_info']}", flush=True)

    if os.environ.get("OPD_DUMP_CONTINUE") != "1":
        print("[opd_dump] dump complete, stopping run (set OPD_DUMP_CONTINUE=1 to keep going)", flush=True)
        os._exit(0)
