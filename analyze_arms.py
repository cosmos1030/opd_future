"""Paired arm comparison. Each seed is one paired observation: same init, same data
order, same rollouts at step 1 (verified bit-identical), differing only in the
advantage estimator. Three seeds can falsify, not establish -- all-same-sign is
p=0.25 under a null of random signs."""
import glob, re, sys
import numpy as np

KEYS = ["critic/rewards/mean", "actor/entropy", "actor/grad_norm", "actor/pg_loss"]

def series(job):
    f = glob.glob(f"logs/full/*_{job}.log")
    if not f:
        return None
    out = {k: {} for k in KEYS}
    for line in open(f[0], errors="ignore"):
        m = re.search(r"step:(\d+) - ", line)
        if not m:
            continue
        s = int(m.group(1))
        for k in KEYS:
            mm = re.search(rf"{re.escape(k)}:(?:np\.float64\()?(-?[\d.eE+-]+)", line)
            if mm:
                out[k][s] = float(mm.group(1))
    return out

PAIRS = [(0, 51467, 51468), (1, 51476, 51477), (2, 51478, 51479)]
rows = []
for seed, jb, jg in PAIRS:
    b, g = series(jb), series(jg)
    if not b or not g:
        print(f"seed {seed}: missing logs"); continue
    steps = sorted(set(b[KEYS[0]]) & set(g[KEYS[0]]))
    if len(steps) < 5:
        print(f"seed {seed}: only {len(steps)} paired steps, skipping"); continue
    r = {}
    for k in KEYS:
        bb = np.array([b[k][s] for s in steps if s in b[k] and s in g[k]])
        gg = np.array([g[k][s] for s in steps if s in b[k] and s in g[k]])
        n = min(len(bb), len(gg))
        r[k] = (bb[:n], gg[:n])
    rows.append((seed, len(steps), r))
    d = r["critic/rewards/mean"][1] - r["critic/rewards/mean"][0]
    wob = np.diff(r["critic/rewards/mean"][0]).std()
    print(f"seed {seed}: {len(steps)} paired steps | reward delta mean {d.mean():+.5f} "
          f"last10 {d[-10:].mean():+.5f} | baseline step-wobble sd {wob:.5f} "
          f"| ratio {abs(d.mean())/max(wob,1e-12):.2f}")

if len(rows) >= 2:
    print("\n" + "=" * 72)
    print(f"{'metric':<22}" + "".join(f"{'seed '+str(s):>13}" for s, _, _ in rows) + f"{'signs':>9}")
    for k in KEYS:
        ds = []
        for _, _, r in rows:
            bb, gg = r[k]
            ds.append(gg.mean() - bb.mean())
        sg = "".join("+" if d > 0 else "-" for d in ds)
        print(f"{k:<22}" + "".join(f"{d:>+13.5f}" for d in ds) + f"{sg:>9}")
    dr = [r["critic/rewards/mean"][1].mean() - r["critic/rewards/mean"][0].mean() for _, _, r in rows]
    wobs = [np.diff(r["critic/rewards/mean"][0]).std() for _, _, r in rows]
    print(f"\nreward: paired deltas {['%+.5f' % d for d in dr]}")
    print(f"        all same sign: {len(set(np.sign(dr))) == 1}"
          f"   (p=0.25 under a random-sign null with 3 seeds)")
    print(f"        |mean delta| vs baseline step-wobble: "
          f"{[f'{abs(d)/w:.2f}x' for d, w in zip(dr, wobs)]}")
    print(f"        a ratio below ~1 means the arm gap is smaller than the run's own"
          f" step-to-step noise")
