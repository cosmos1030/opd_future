"""Pairwise across-question comparison at step 30, for every arm with a valdump.

Each arm contributes a 210-vector of per-question accuracy (mean over its 16
samples). Differences are paired by question and bootstrapped over questions,
which is the only error bar that is valid here: the maj@16/std the trainer logs
is a *within-question* bootstrap and says nothing about across-question spread.
"""
import json, glob, os
import numpy as np

DUMPS = [
    ("base",      "valdump/base/0.jsonl",     "학습 전"),
    ("off@30",    "valdump/off30/0.jsonl",    "forward CE 미래항 X (59175 재평가)"),
    ("baseline",  "valdump/baseline/30.jsonl","credit 성분만 (R-b)"),
    ("revKL@5e-5","valdump/revkl/30.jsonl",   "reverse KL α=1.0 lr 5e-5"),
    ("revKL@2e-5","valdump/revkl-lr2e5/30.jsonl","reverse KL α=1.0 lr 2e-5 (59912)"),
]

def load(p):
    if not os.path.exists(p):
        return None
    by = {}
    for line in open(p):
        r = json.loads(line)
        by.setdefault(r["input"], []).append(float(r["acc"]))
    return {k: float(np.mean(v)) for k, v in by.items()}

def paired(a, b, n_boot=20000, seed=0):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return len(keys), d.mean()*100, np.percentile(boot, 2.5)*100, np.percentile(boot, 97.5)*100

arms = []
for tag, path, desc in DUMPS:
    q = load(path)
    if q is None:
        print(f"  [없음] {tag}: {path}")
        continue
    arms.append((tag, q, desc))

print("=" * 86)
print("STEP 30 문항별 짝지음 비교 — SciKnowEval chemistry, 210문항 x 16샘플")
print("=" * 86)
for tag, q, desc in arms:
    print(f"  {tag:<10} {np.mean(list(q.values()))*100:6.2f}   n={len(q):3d}   {desc}")

print()
print(f"{'A - B':<22}{'Δ mean':>8}{'CI95':>22}{'유의':>6}")
print("-" * 86)
for i in range(len(arms)):
    for j in range(len(arms)):
        if i >= j:
            continue
        ta, qa, _ = arms[i]
        tb, qb, _ = arms[j]
        n, d, lo, hi = paired(qa, qb)
        sig = "✓" if lo > 0 or hi < 0 else "–"
        print(f"{ta+' - '+tb:<22}{d:+8.2f}   [{lo:+6.2f}, {hi:+6.2f}]      {sig}")

print()
print("=" * 86)
print("같은 가중치를 다시 평가했을 때의 흔들림 (측정 바닥)")
print("=" * 86)
print("  off@30  학습 로그의 step-30 검증 :  mean@16 69.88   maj@16 71.54")
print("  off@30  같은 체크포인트 재평가    :  mean@16 70.18   maj@16 73.00")
print("  차이                              :         +0.30          +1.46")
print()
print("  참고: step 30에서 on - off = 70.21 - 69.88 = +0.33 (mean@16)")
print("  → 같은 가중치를 두 번 재면 나오는 차이와 같은 크기다.")
