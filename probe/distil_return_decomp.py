#!/usr/bin/env python3
"""Variance decomposition of DistIL's future-credit returns.

Answers: is there room for a baseline / value critic in DistIL's Term 2?

DistIL (arXiv:2606.05152, Proposition 1) adds to the forward-CE loss a REINFORCE
term with NO baseline:

    token_reward[t] = -sum_v pi_T(v|s_t) log pi_theta(v|s_t)      (clamped >= 0)
    future_returns[t] = (sum_{s>t} token_reward[s]) / (T - t - 1)
    pg_per_token[t] = log pi_theta(a_t|s_t) * future_returns[t]

The same decomposition was run for OPD's reverse-KL returns (see
opd_horizon/CRITIC_AND_SPEC_2026-09-27.md) and killed the critic there: the
action-dependent share of the variance was 1.55%, so a critic would have needed
R^2 >= 0.9845 against a measured 0.25, and its advantage scored R^2 = -2.797,
i.e. worse than using zero. Length normalisation was tested there and did not
change the structure (1.55% -> 1.69%).

DistIL differs in one way that could matter: OPD's per-token reward is a
single-sample log-ratio on the token that was actually drawn, while DistIL's is
an expectation over the top-K vocabulary. That removes the action-sampling noise
at the scoring position itself, so N_pi should be structurally smaller here.
This script measures whether that is enough.

Method (mirrors the OPD branch design):
  Phase 0  one reference rollout per prompt; cut it at the highest-entropy
           position to get a shared prefix whose next-token distribution has
           real spread.
  Phase 1  take the student's top-A next tokens at that branch state as forced
           actions, with their policy probabilities.
  Phase 2  for each (branch, forced action) generate K continuations.
  Phase 3  score every continuation with DistIL's exact reward and return.
  Phase 4  decompose:
             Q(s,a)  = mean_k return
             I_pi(s) = sum_a pi(a|s) (Q(s,a) - Qbar)^2     action signal
             N_pi(s) = sum_a pi(a|s) Var_k(return)/K       MC noise in Qhat
             share   = mean_s I_pi / Var_total
             required R^2 = 1 - share

Single GPU. vLLM is freed before the HF scoring pass, so peak memory is either
the vLLM engine or the two 4B models, not both.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_PROMPT_LEN = 2048


def add_tail(log_probs):
    """DistIL's top-K tail mass term (core_algos.py): append log(1 - sum p_i) so
    the truncated distribution still sums to 1."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    tail_log = torch.log(-torch.expm1(log_s.clamp(max=-1e-7)))
    return torch.cat([log_probs, tail_log], dim=-1)


def distil_token_reward(s_logits, t_logits, topk):
    """-sum_v pi_T(v) log pi_theta(v) over the STUDENT's top-K support, clamped
    at 0 -- exactly compute_distil_self_distillation_loss's token_rewards.

    The student's indices drive the gather: dp_actor passes
    topk_indices=student_topk_indices into the teacher forward.
    """
    s_logp_all = F.log_softmax(s_logits.float(), dim=-1)
    t_logp_all = F.log_softmax(t_logits.float(), dim=-1)
    idx = s_logits.float().topk(topk, dim=-1).indices
    s_d = add_tail(s_logp_all.gather(-1, idx))
    t_d = add_tail(t_logp_all.gather(-1, idx))
    r = -(t_d.exp() * s_d).sum(-1)
    return r.clamp(min=0.0)


def load_prompts(path, n_prompts, tokenizer, seed):
    """SciKnowEval chemistry prompts, rendered through the chat template the
    training run used (enable_thinking=False, matching user.yaml)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    out = []
    for r in rows:
        # mirrors SDPO/data/preprocess.py make_map_fn: optional system turn,
        # then the raw `prompt` string as the user turn.
        system = r.get("system")
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": r["prompt"]})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= MAX_PROMPT_LEN:
            out.append(ids)
        if len(out) >= n_prompts:
            break
    return out


def build_branches(student_path, prompt_ids, max_new_tokens, seed, n_actions, min_tail):
    """Phase 0 + 1: reference rollout, highest-entropy cut, top-A forced actions."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print("Phase 0: reference rollout per prompt", flush=True)
    llm = LLM(model=student_path, dtype="bfloat16", gpu_memory_utilization=0.45,
              trust_remote_code=True, max_model_len=MAX_PROMPT_LEN + max_new_tokens,
              enforce_eager=True)
    sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=max_new_tokens, seed=seed)
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in prompt_ids], sp)
    refs = [(p, list(o.outputs[0].token_ids)) for p, o in zip(prompt_ids, outs)
            if len(o.outputs[0].token_ids) >= min_tail + 16]
    del llm
    torch.cuda.empty_cache()

    print(f"Phase 1: branch point + top-{n_actions} actions for {len(refs)} refs", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    branches = []
    with torch.no_grad():
        for p_ids, ref in refs:
            full = torch.tensor([p_ids + ref], dtype=torch.long, device="cuda")
            logits = student(input_ids=full).logits[0]
            lo = max(4, len(ref) // 4)
            hi = len(ref) - min_tail
            if hi <= lo:
                continue
            cand = logits[len(p_ids) - 1 + lo: len(p_ids) - 1 + hi].float()
            probs = F.softmax(cand, dim=-1)
            ent = -(probs * (probs + 1e-12).log()).sum(-1)
            b = lo + int(ent.argmax().item())
            prefix = p_ids + ref[:b]
            # policy over the forced actions, renormalised to the top-A support
            at_branch = F.softmax(logits[len(p_ids) - 1 + b].float(), dim=-1)
            topv, topi = at_branch.topk(n_actions)
            branches.append({
                "prefix": prefix,
                "actions": topi.tolist(),
                "pi": (topv / topv.sum()).tolist(),
                "entropy": float(ent[b - lo].item()),
                "pi_raw_mass": float(topv.sum().item()),
            })
    del student
    torch.cuda.empty_cache()
    print(f"  {len(branches)} branches, mean entropy "
          f"{np.mean([b['entropy'] for b in branches]):.3f}, mean top-{n_actions} mass "
          f"{np.mean([b['pi_raw_mass'] for b in branches]):.3f}", flush=True)
    return branches


def rollout_forced(student_path, branches, k, max_new_tokens):
    """Phase 2: K continuations from each (prefix + forced action)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"Phase 2: {len(branches)} branches x {len(branches[0]['actions'])} actions "
          f"x K={k} continuations", flush=True)
    seqs, meta = [], []
    for bi, b in enumerate(branches):
        for ai, a in enumerate(b["actions"]):
            seqs.append(b["prefix"] + [a])
            meta.append((bi, ai))
    llm = LLM(model=student_path, dtype="bfloat16", gpu_memory_utilization=0.45,
              trust_remote_code=True, max_model_len=MAX_PROMPT_LEN + 2 * max_new_tokens,
              enforce_eager=True)
    sp = SamplingParams(n=k, temperature=0.6, top_p=0.95, max_tokens=max_new_tokens)
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in seqs], sp)
    recs = []
    for (bi, ai), seq, out in zip(meta, seqs, outs):
        for comp in out.outputs:
            recs.append({"bi": bi, "ai": ai, "ctx": seq, "cont": list(comp.token_ids)})
    del llm
    torch.cuda.empty_cache()
    print(f"  {len(recs)} continuations", flush=True)
    return recs


def score(recs, student_path, teacher_path, topk, chunk):
    """Phase 3: DistIL return for each continuation.

    The return at the branch position is the length-normalised mean of the
    token rewards STRICTLY AFTER the forced action -- matching DistIL's shift
    (future_returns[t] excludes token_reward[t]).
    """
    print("Phase 3: scoring with student + teacher", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    out = []
    with torch.no_grad():
        for n, r in enumerate(recs):
            ids = r["ctx"] + r["cont"]
            if len(r["cont"]) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            gen_start = len(r["ctx"])
            rewards = []
            # logits[i-1] scores token i; we need positions gen_start .. end-1
            s_all = student(input_ids=x).logits[0, gen_start - 1:-1, :]
            t_all = teacher(input_ids=x).logits[0, gen_start - 1:-1, :]
            for i in range(0, s_all.shape[0], chunk):
                rewards.append(distil_token_reward(s_all[i:i + chunk], t_all[i:i + chunk], topk))
            rew = torch.cat(rewards)                      # [cont_len]
            ret = float(rew.mean().item())                # (sum_{s>t} r_s)/(T-t-1)
            out.append({"bi": r["bi"], "ai": r["ai"], "ret": ret, "len": len(r["cont"])})
            if (n + 1) % 100 == 0:
                print(f"  scored {n + 1}/{len(recs)}", flush=True)
            del x, s_all, t_all
    del student, teacher
    torch.cuda.empty_cache()
    return out


def decompose(scored, branches, k):
    """Phase 4: I_pi / N_pi / share / required R^2."""
    by = {}
    for s in scored:
        by.setdefault((s["bi"], s["ai"]), []).append(s["ret"])

    all_rets, I_list, I_naive_list, N_list, Q_spread = [], [], [], [], []
    for bi, b in enumerate(branches):
        pis, Qs, Vs = [], [], []
        for ai, pi in enumerate(b["pi"]):
            rs = by.get((bi, ai))
            if not rs or len(rs) < 2:
                continue
            pis.append(pi)
            Qs.append(float(np.mean(rs)))
            Vs.append(float(np.var(rs, ddof=1)) / len(rs))
            all_rets.extend(rs)
        if len(Qs) < 2:
            continue
        w = np.array(pis) / np.sum(pis)
        Q = np.array(Qs)
        Qbar = float((w * Q).sum())
        naive = float((w * (Q - Qbar) ** 2).sum())
        Npi = float((w * np.array(Vs)).sum())
        # Qhat_a is a K-sample mean, so the weighted spread of the ESTIMATES is
        # biased up by their own sampling noise:
        #   E[sum_a w_a (Qhat_a - Qhatbar)^2] = I_pi + (1 - sum_a w_a^2) * N_pi
        # Reporting the naive number would read the MC floor as action signal.
        I_naive_list.append(naive)
        I_list.append(naive - (1.0 - float((w ** 2).sum())) * Npi)
        N_list.append(Npi)
        Q_spread.append(float(Q.max() - Q.min()))

    rng = np.random.default_rng(0)
    arr = np.array(I_list)
    boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(10000)])
    I_pi = float(np.mean(I_list))
    I_pi_naive = float(np.mean(I_naive_list))
    N_pi = float(np.mean(N_list))
    var_tot = float(np.var(np.array(all_rets), ddof=1))
    share = I_pi / var_tot if var_tot > 0 else float("nan")
    return {
        "n_branches_used": len(I_list),
        "n_returns": len(all_rets),
        "K": k,
        "return_mean": float(np.mean(all_rets)),
        "return_sd": float(np.std(all_rets, ddof=1)),
        "Var_total": var_tot,
        "I_pi_naive_biased": I_pi_naive,
        "I_pi": I_pi,
        "I_pi_CI95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "I_pi_P_gt_0": float(np.mean(boot > 0)),
        "N_pi": N_pi,
        "I_over_N": I_pi / N_pi if N_pi > 0 else float("inf"),
        "action_share_of_var": share,
        "action_share_CI95": [float(np.percentile(boot, 2.5)) / var_tot,
                              float(np.percentile(boot, 97.5)) / var_tot],
        "required_critic_R2": 1.0 - share,
        "mean_Q_spread": float(np.mean(Q_spread)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True, help="SciKnowEval chemistry train.json (JSONL)")
    ap.add_argument("--n_prompts", type=int, default=24)
    ap.add_argument("--n_actions", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=100, help="distillation_topk used in training")
    ap.add_argument("--min_tail", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64, help="positions per reward chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="probe/distil_decomp.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    prompts = load_prompts(args.data, args.n_prompts, tok, args.seed)
    print(f"{len(prompts)} prompts", flush=True)

    branches = build_branches(args.student, prompts, args.max_new_tokens,
                              args.seed, args.n_actions, args.min_tail)
    recs = rollout_forced(args.student, branches, args.k, args.max_new_tokens)
    scored = score(recs, args.student, args.teacher, args.topk, args.chunk)
    stats = decompose(scored, branches, args.k)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"stats": stats, "scored": scored, "branches": branches,
                "args": vars(args)}, args.out)

    print("\n=== DistIL future-return decomposition ===")
    for key, val in stats.items():
        print(f"  {key:24s} {val}")
    print("\nOPD reference (CRITIC_AND_SPEC_2026-09-27.md, horizon T, length-normalised):")
    print("  action_share_of_var      0.0169")
    print("  required_critic_R2       0.9831")
    print("  measured critic R2       ~0.25   -> advantage R2 -2.797 (worse than zero)")


if __name__ == "__main__":
    main()
