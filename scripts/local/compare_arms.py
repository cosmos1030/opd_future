"""Consolidated step-30 comparison across arms, with across-question error bars.

Aggregates come from the trainer logs; error bars come from the per-question
valdump JSONLs, which only some arms have.
"""
import json, re, glob, sys
import numpy as np

ARMS = [
    ("base",            "학습 전",            59449, None,       None),
    ("fwdCE on  5e-5",  "forward CE 미래항 O", 59174, "5e-5",     None),
    ("fwdCE off 5e-5",  "forward CE 미래항 X", 59175, "5e-5",     None),
    ("fwdCE on  2e-5",  "forward CE 미래항 O", 59196, "2e-5",     None),
    ("fwdCE off 2e-5",  "forward CE 미래항 X", 59267, "2e-5",     None),
    ("revKL     5e-5",  "reverse KL α=1.0",   59468, "5e-5",     "revkl"),
    ("baseline  5e-5",  "credit 성분만",       59486, "5e-5",     "baseline"),
    ("common    5e-5",  "공통 모드만",         59487, "5e-5",     "common"),
]

def agg_from_log(jid, want_step=30):
    paths = glob.glob(f"logs/*_{jid}.log")
    if not paths:
        return None
    rows = {}
    for line in open(sorted(paths)[-1], errors="ignore"):
        if "val-core" not in line:
            continue
        s = re.search(r"step:(\d+)", line)
        m = re.search(r"acc/mean@16:np\.float64\(([0-9.]+)\)", line)
        j = re.search(r"acc/maj@16/mean:np\.float64\(([0-9.]+)\)", line)
        b = re.search(r"acc/best@16/mean:np\.float64\(([0-9.]+)\)", line)
        if m and j and b:
            rows[int(s.group(1)) if s else 0] = (
                float(m.group(1)) * 100, float(j.group(1)) * 100, float(b.group(1)) * 100)
    if not rows:
        return None
    step = want_step if want_step in rows else max(rows)
    return step, rows[step], rows

def per_question(tag, step):
    """Return per-question mean accuracy vector, keyed by prompt."""
    p = f"valdump/{tag}/{step}.jsonl"
    try:
        rows = [json.loads(l) for l in open(p)]
    except OSError:
        return None
    by = {}
    for r in rows:
        by.setdefault(r["input"], []).append(float(r["acc"]))
    return by

def paired(a, b, n_boot=20000, seed=0):
    """Paired bootstrap over questions on the per-question accuracy difference."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 30:
        return None
    da = np.array([np.mean(a[k]) for k in keys])
    db = np.array([np.mean(b[k]) for k in keys])
    d = da - db
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    return (len(keys), d.mean() * 100, np.percentile(boot, 2.5) * 100,
            np.percentile(boot, 97.5) * 100, float((boot > 0).mean()))

print("=" * 78)
print("STEP 30 — all arms, Qwen3-4B-Instruct-2507, SciKnowEval chemistry, n=16")
print("=" * 78)
print(f"{'arm':<17}{'설명':<22}{'mean@16':>9}{'maj@16':>9}{'best@16':>9}  {'Δmean vs base':>13}")
print("-" * 78)
base_agg = None
for tag, desc, jid, lr, dump in ARMS:
    r = agg_from_log(jid)
    if r is None:
        print(f"{tag:<17}{desc:<22}{'—':>9}{'—':>9}{'—':>9}   (아직 없음)")
        continue
    step, (mean, maj, best), _ = r
    if tag == "base":
        base_agg = mean
    delta = f"{mean - base_agg:+6.2f}" if base_agg is not None and tag != "base" else ""
    note = "" if step == 30 else f"  [step {step}]"
    print(f"{tag:<17}{desc:<22}{mean:9.2f}{maj:9.2f}{best:9.2f}  {delta:>13}{note}")

print()
print("=" * 78)
print("문항별 오차 — valdump JSONL이 있는 arm만. 210문항 paired bootstrap.")
print("=" * 78)
base_q = per_question("base", 0)
if base_q is None:
    print("  base dump 없음")
    sys.exit(0)
print(f"  base: {len(base_q)}문항, mean@16 {np.mean([np.mean(v) for v in base_q.values()])*100:.2f}")
print()
for tag, desc, jid, lr, dump in ARMS:
    if not dump:
        continue
    for step in (30, 25, 20, 15):
        q = per_question(dump, step)
        if q:
            break
    if not q:
        print(f"  {tag:<17} dump 없음")
        continue
    own = np.mean([np.mean(v) for v in q.values()]) * 100
    res = paired(q, base_q)
    if res is None:
        print(f"  {tag:<17} step {step}: 공통 문항 부족")
        continue
    n, d, lo, hi, p = res
    sig = "*" if lo > 0 or hi < 0 else " "
    print(f"  {tag:<17} step {step:>2}  mean@16 {own:6.2f}   "
          f"vs base {d:+6.2f}  CI95 [{lo:+6.2f}, {hi:+6.2f}] {sig}  P(>0)={p:.3f}")
