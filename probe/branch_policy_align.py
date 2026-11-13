"""Does the future term make the policy prefer low-future-cost actions?

The decomposition in FUTURE_CREDIT.md section 2 asks how much of the return's
variance is attributable to the action. This asks the complementary question,
which is the one the future term is actually supposed to answer: at the same
state, with the same future costs, does an arm trained WITH the term put more
probability on the cheaper action than an arm trained WITHOUT it?

Design. Branch points and the future costs Qhat(s,a) are built ONCE from a fixed
reference model, so both arms are scored against the same states and the same
yardstick. Each arm then contributes only a single forward pass per branch point
to give pi_arm(a|s). The comparison is paired by branch point, which is where the
power comes from: the expensive Qhat is shared, so the number of branch points
can be large without paying for it twice.

Statistic, per branch point and per arm:

    E_pi[Q] = sum_a w_a Qhat(s,a),   w = pi_arm renormalised over the candidates

Lower is better: the policy puts mass on actions with lower future teacher-
matching cost. If the future term installs the mechanism Proposition 1 claims,
E_pi_on[Q] < E_pi_off[Q], paired over branch points.

This does not depend on the 210-question accuracy resolution at all.
"""
import argparse, json, importlib.util, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_spec = importlib.util.spec_from_file_location(
    "_decomp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "distil_return_decomp.py"))
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def policy_at_branches(model_path, branches):
    """pi_arm(a|s) over each branch point's fixed candidate actions."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    out = []
    with torch.no_grad():
        for n, b in enumerate(branches):
            x = torch.tensor([b["prefix"]], dtype=torch.long, device="cuda")
            logits = model(input_ids=x).logits[0, -1].float()
            p = F.softmax(logits, dim=-1)
            v = p[torch.tensor(b["actions"], device="cuda")]
            out.append({
                "w": (v / v.sum()).tolist(),      # renormalised over the candidates
                "mass": float(v.sum().item()),    # how much total mass the candidates hold
            })
            if (n + 1) % 50 == 0:
                print(f"  policy {n + 1}/{len(branches)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def expected_cost(pols, Q, valid):
    """E_pi[Q] per branch point, over the branches where Q is defined."""
    return np.array([float(np.dot(pols[i]["w"], Q[i])) for i in valid])


def paired(a, b, n_boot=20000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return (d.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)),
            float((boot < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="fixed model that defines states and Qhat")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--arms", required=True, nargs="+",
                    help="name=path pairs, e.g. on=models/on30-merged off=models/off30-merged")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--n_actions", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--min_tail", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="probe/branch_align.pt")
    ap.add_argument("--reuse", default="", help="reuse branches+Q from a previous .pt")
    a = ap.parse_args()

    arms = dict(s.split("=", 1) for s in a.arms)

    if a.reuse and os.path.exists(a.reuse):
        print(f"reusing states and Qhat from {a.reuse}")
        blob = torch.load(a.reuse, weights_only=False)
        branches, Q, valid = blob["branches"], blob["Q"], blob["valid"]
    else:
        tok = AutoTokenizer.from_pretrained(a.ref, trust_remote_code=True)
        prompt_ids = _d.load_prompts(a.data, a.n_prompts, tok, a.seed)
        branches = _d.build_branches(a.ref, prompt_ids, a.max_new_tokens, a.seed,
                                     a.n_actions, a.min_tail)
        recs = _d.rollout_forced(a.ref, branches, a.k, a.max_new_tokens)
        scored = _d.score(recs, a.ref, a.teacher, a.topk, a.chunk)

        by = {}
        for s in scored:
            by.setdefault((s["bi"], s["ai"]), []).append(s["ret"])
        Q, valid = [], []
        for bi, b in enumerate(branches):
            qs = [by.get((bi, ai)) for ai in range(len(b["actions"]))]
            if any(q is None or len(q) < 2 for q in qs):
                Q.append(None); continue
            Q.append([float(np.mean(q)) for q in qs])
            valid.append(bi)
        print(f"  {len(valid)}/{len(branches)} branch points with all actions scored")

    pols = {}
    for name, path in arms.items():
        print(f"Phase 4: policy of arm '{name}'", flush=True)
        pols[name] = policy_at_branches(path, branches)

    torch.save({"branches": branches, "Q": Q, "valid": valid, "pols": pols,
                "args": vars(a)}, a.out)

    print()
    print("=" * 72)
    print("E_pi[Qhat] per branch point  (lower = policy prefers cheaper futures)")
    print("=" * 72)
    Ql = [Q[i] for i in valid]
    spread = np.array([max(q) - min(q) for q in Ql])
    print(f"  branch points      {len(valid)}")
    print(f"  mean Q spread      {spread.mean():.5f}")
    print(f"  uniform-policy E[Q]{np.mean([np.mean(q) for q in Ql]):9.5f}")
    print()
    E = {n: expected_cost(pols[n], Q, valid) for n in arms}
    for n in arms:
        print(f"  {n:<12} E_pi[Q] = {E[n].mean():.5f}   "
              f"candidate mass = {np.mean([pols[n][i]['mass'] for i in valid]):.3f}")
    names = list(arms)
    if len(names) == 2:
        x, y = names[0], names[1]
        d, lo, hi, p = paired(E[x], E[y])
        print()
        print(f"  {x} - {y} = {d:+.6f}   CI95 [{lo:+.6f}, {hi:+.6f}]   "
              f"P({x} < {y}) = {p:.3f}")
        print(f"  {'significant' if hi < 0 or lo > 0 else 'not distinguishable from zero'}")


if __name__ == "__main__":
    main()
