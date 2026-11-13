"""Re-score every valdump with a fallback answer extractor.

The shipped scorer (SDPO/verl/utils/reward_score/feedback/mcq.py) does

    answer = text.split("<answer>")[-1].split("</answer>")[0].strip()

so when the model answers without an <answer> tag -- "Final answer: A." --
split returns the WHOLE response and the comparison to the gold letter fails.
Those samples score 0 even when the letter is right.

strict  = the shipped rule, unchanged (this is what training rewarded)
lenient = strict, and if that yields no single letter, a fallback chain
"""
import json, os, re, sys
import numpy as np

L = ("A", "B", "C", "D")

def strict(text):
    a = text.split("<answer>")[-1].split("</answer>")[0].strip()
    return a if a in L else None

FALLBACKS = [
    r"<answer>\s*\(?([ABCD])\)?[\.\s]*</answer>",      # tag with stray parens/period
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*\*{0,2}\(?([ABCD])\)?\b",
    r"\\boxed\{\s*\(?([ABCD])\)?\s*\}",
    r"\*\*\s*\(?([ABCD])\)?\s*\*\*\s*(?:is\s+correct|is\s+the\s+answer)?\s*\.?\s*$",
    r"^\s*\(?([ABCD])\)?\s*[\.\)]?\s*$",               # whole tail is one letter
]

def lenient(text):
    s = strict(text)
    if s:
        return s, "strict"
    tail = text[-1500:]
    for i, pat in enumerate(FALLBACKS):
        m = re.findall(pat, tail, re.IGNORECASE | re.MULTILINE)
        if m:
            return m[-1].upper(), f"fb{i}"
    return None, "none"

def run(dump_root="valdump"):
    out = []
    for arm in sorted(os.listdir(dump_root)):
        d = os.path.join(dump_root, arm)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jsonl"):
                continue
            step = int(f[:-6])
            rows = [json.loads(l) for l in open(os.path.join(d, f))]
            by_q_s, by_q_l = {}, {}
            how = {}
            for r in rows:
                gt = str(r["gts"]).strip() if not isinstance(r["gts"], list) else str(r["gts"][0]).strip()
                s = strict(r["output"])
                p, tag = lenient(r["output"])
                how[tag] = how.get(tag, 0) + 1
                by_q_s.setdefault(r["input"], []).append(float(s == gt))
                by_q_l.setdefault(r["input"], []).append(float(p == gt))
            acc_s = np.mean([np.mean(v) for v in by_q_s.values()]) * 100
            acc_l = np.mean([np.mean(v) for v in by_q_l.values()]) * 100
            out.append(dict(arm=arm, step=step, n=len(by_q_s), strict=acc_s,
                            lenient=acc_l, delta=acc_l - acc_s,
                            rescued=sum(c for t, c in how.items() if t.startswith("fb")) / len(rows) * 100,
                            unanswered=how.get("none", 0) / len(rows) * 100))
    return out

if __name__ == "__main__":
    res = run()
    print(f"{'arm':22s}{'step':>5}{'strict':>8}{'lenient':>9}{'Δ':>7}{'구제%':>7}{'무응답%':>8}")
    for r in res:
        mark = "  <<<" if r["delta"] > 3 else ""
        print(f"{r['arm']:22s}{r['step']:5d}{r['strict']:8.2f}{r['lenient']:9.2f}"
              f"{r['delta']:+7.2f}{r['rescued']:7.1f}{r['unanswered']:8.1f}{mark}")
    json.dump(res, open("/tmp/claude-1031/-home-doyoonkim-projects/76301a6a-7fef-4ff6-a2d0-7778c58bd707/scratchpad/rescore.json", "w"))
