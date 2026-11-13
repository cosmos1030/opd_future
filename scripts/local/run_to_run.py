"""Run-to-run variation of one identical configuration, from the 59911 wreck.

Job 59911 re-ran the lr 5e-5 `on` arm with the *same* seed (42), the same lr,
the same 4 GPUs and the same code as 59174, and died on its wall at step ~24.
Its four validation points are therefore a replicate of 59174 under identical
settings, differing only in CUDA/vLLM nondeterminism and node contention.

That makes the right comparison available for the first time: is the on-vs-off
gap larger than the gap between two runs that should be the same run?

Because the seed is identical, this is a LOWER bound on seed-to-seed variation.
Caveats: n=4 matched steps, steps within a run are autocorrelated, and 59911
never reached step 30, so the headline step is not covered.
"""
import re, numpy as np

P = "logs/DistIL-science-chemistry--home-doyoonkim-projects-distIL-models-Qwen3-4B-Instruct-2507"
RUNS = {"on(59174)": f"{P}_59174.log",
        "off(59175)": f"{P}_59175.log",
        "on(59911)": f"{P}-on30dump_59911.log"}

def curve(f):
    out = {}
    for line in open(f, errors="ignore"):
        a = re.search(r"acc/mean@16:np\.float64\(([0-9.]+)\)", line)
        b = re.search(r"acc/maj@16/mean:np\.float64\(([0-9.]+)\)", line)
        s = re.search(r"step:(\d+)", line)
        if a and b and s:
            out[int(s.group(1))] = (float(a.group(1)) * 100, float(b.group(1)) * 100)
    return out

c = {k: curve(v) for k, v in RUNS.items()}
steps = sorted(set.intersection(*(set(v) for v in c.values())))
on1, off, on2 = c["on(59174)"], c["off(59175)"], c["on(59911)"]

print(f"matched steps: {steps}\n")
for i, name in ((0, "mean@16"), (1, "maj@16")):
    e = np.array([on1[s][i] - off[s][i] for s in steps])
    n = np.array([on1[s][i] - on2[s][i] for s in steps])
    print(f"{name}")
    print(f"  on - off (the claimed effect) : mean {e.mean():+.2f}  mean|.| {np.abs(e).mean():.2f}  max|.| {np.abs(e).max():.2f}")
    print(f"  on - on  (same config, twice) : mean {n.mean():+.2f}  mean|.| {np.abs(n).mean():.2f}  max|.| {np.abs(n).max():.2f}")
    print(f"  ratio of discrepancies        : {np.abs(n).mean()/np.abs(e).mean():.1f}x\n")
