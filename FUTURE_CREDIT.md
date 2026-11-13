# Is DistIL's future term credit assignment, or a position-wise common mode?

2026-10-28, revised 2026-11-07, open. Paused mid-grid on 2026-11-06; §8.5 says what is unrun. Student/teacher are Qwen3-4B-Instruct-2507 with an EMA teacher
(`teacher_update_rate=0.01`), SciKnowEval **chemistry**, batch 32, rollout n=8,
response 8192, validated every 5 steps on 210 held-out prompts at n=16 /
temperature 0.6 through the verl harness. All training arms below are 30 steps
unless noted. Offline measurements are on 24 forced-action branch points.

**Why this exists.** [opd_horizon/SYNTHESIS.md](../opd_horizon/SYNTHESIS.md)
closed with: sequence-level temporal credit under reverse KL is real but does not
scale as a training signal, because the action-dependent share of the future
return is ~1.7% and a critic would need R² ≈ 0.983 to extract it, while the best
measured critic reaches R² ≈ 0.25. DistIL ([arXiv:2606.05152](https://arxiv.org/abs/2606.05152))
runs the same idea under a **forward CE** reward and reports that it works. Either
the divergence matters, or the term is doing something other than what
Proposition 1 says it does.

The short version, after four of its own headlines were overturned by
measurement: **the future term's effect (+0.33 to +2.0 mean@16) is smaller than
the spread between two runs of the same configuration with the same seed (2.74),
its action-dependent share of the return is 0.22% (95% CI [0.12%, 0.33%]), the
forward-CE-beats-reverse-KL result that seemed to separate DistIL from OPD
disappears once reverse KL is given its own learning rate (+0.54, CI [−4.52,
+5.60]), and the future term's effect, pooled over the last three checkpoints instead of
read off one, is +0.08 / +0.60 / −0.67 under three different divergences — zero
in all three.** The single-checkpoint numbers that this document reported for
four revisions turn out to flip sign between consecutive validation steps of the
*same two runs*: forward KL at lr 5e-5 gives +30.06 at step 10 and −5.74 at step
30, four of six checkpoints "significant" in both directions, with neither arm
collapsed. Checkpoint choice is a larger error term than generation noise (0.64),
run-to-run spread at fixed seed (2.74), or the 1.28 between two arms that are
mathematically the same objective (`fwdCE off` and `fwdKL off`, 70.18 vs 71.46). None of that is a finding of "no effect" — it is a finding that this
testbed cannot resolve the effect it was built to measure. Sections 3–4 were
built to separate credit from common mode, and §4 failed for a reason worth
recording. §6 is the reframe that followed: the question has to be asked at the
objective, on a task where a future term has somewhere to act.

**§8 is that attempt, on physics, and it carries the one positive mechanism
result in this document.** Term 2 splits exactly into an action-dependent half
and an action-independent one, and run alone at matched learning rate the
action-independent half drives response length to the 8192 cap with `clip_ratio`
0.98 while the action-dependent half holds flat at `clip_ratio` 0.00. The part of
the published future term that collapses the policy is precisely the part a
baseline would remove. Two further corrections live there: this document was
reading mean@16 against a paper that plots Best@16 (§8.1), and the physics
learning-rate brackets were open when I reported a contrast off them (§8.2) —
that contrast is withdrawn.

**A caveat that governs everything below.** Every arm I designed up to §4 —
`off`, `baseline`, `common`, `shuffle` — destroys some part of the signal and
checks whether performance drops. Not one of them would *amplify* a real effect,
so the whole battery can only ever return "we could not see it". That is a
confirmation-shaped design, and it is why §6 exists. Five times now a new
measurement has moved a headline in this document, and all five moved *against*
the effect I had written up — §1 by the generation floor and then by run-to-run
spread, §5 by the learning-rate sweep, then by the closed reverse-KL bracket,
and finally by the checkpoint-flip result that withdrew most of the preceding
intervals. The design predicted that asymmetry.

## 1. The future term's effect is smaller than the spread between two identical runs

DistIL's Term 2 is a score function on the sampled token
(`core_algos.py:1290-1307`; it was line 1278 before the mode switch below was added):

    pg_per_token = student_log_probs * future_returns

`future_returns[t]` is the sum of per-token rewards strictly after t, divided by
`remaining`, clamped at ≥ 0, with **no baseline**. `DISTIL_PG_MODE=off` replaces
the whole thing with zeros; everything else is held fixed.

Validation at step 30, in percent:

| | mean@16 | maj@16 | best@16 |
|---|---|---|---|
| base, untrained (job 59449) | 50.03 | 53.70 | 83.45 |
| lr 5e-5, future term **on** (59174) | **70.21** | **73.06** | 94.19 |
| lr 5e-5, future term off (59175) | 69.88 | 71.54 | 89.22 |
| lr 2e-5, future term **on** (59196) | **69.35** | **72.71** | 85.30 |
| lr 2e-5, future term off (59267) | 67.35 | 69.14 | 86.53 |
| lr 1e-4, either (59268/59269) | diverged | — | — |

DistIL itself is worth **+20pp** over base. The future term's own share is
**+1.5pp mean@16 / +1.6pp maj@16** at lr 5e-5 and **+2.0 / +3.6pp** at lr 2e-5.

Paired over every validated checkpoint (steps 5…30, both learning rates, 12
pairs), `on` beats `off` on **9/12** for mean@16 and **9/12** for maj@16; the
mean paired maj@16 gap is +1.35pp at 5e-5 and +2.15pp at 2e-5. The direction is
consistent; the magnitude is not stable step to step, which is what a 210-prompt
validation set at n=16 looks like.

**Correction to an earlier reading.** I first called this difference "within
noise" by treating `val-core/…/maj@16/std` as a between-question standard
deviation and dividing by √210. `verl/trainer/ppo/metric_utils.py:600-642` shows
it is the mean of *within-question* bootstrap standard deviations, which says
nothing about the error on the across-question mean. That understated the
measurement noise by 4–5×, in the wrong direction for the claim I made. The
"within noise" conclusion is withdrawn **as reasoned**. The subsection below
re-establishes it for lr 5e-5 by a method that is actually valid; the original
argument was wrong even where its conclusion happened to land.

**lr 1e-4 is outside the trainable range**, for both arms. At step 20
`success_sample_fraction` hits 0.0 — no sampled rollout solves its problem, so no
distillation targets exist, and `ce_loss`, `pg_loss` and `grad_norm` are all
exactly 0.0 while the model rambles to the 8192-token cap (603 → 2041 s/it).
That is a collapse of the data-generation loop, not of the loss. The lr axis is
therefore closed with two working settings, and the sign of the effect is the
same in both.

### The measurement floor, measured (2026-10-31)

Job 59918 re-evaluated the **off@30 checkpoint itself** — same weights, same 210
questions, same n=16 / temperature 0.6 — through a fresh vLLM instance. It
returned mean@16 **70.18** where the training log had recorded **69.88** for the
same checkpoint. So a re-draw of the generations alone moves the headline number
by +0.30.

Resampling the 16 samples within each question (questions held fixed, 5000
draws) turns that anecdote into a distribution:

| statistic | sd of one evaluation | sd of the difference of two | 95% limit on \|difference\| |
|---|---|---|---|
| mean@16 | 0.45 | **0.64** | 1.22 |
| maj@16 | 0.96 | **1.36** | — |

The observed +0.30 sits well inside the predicted 0.64, so the noise model
checks out. Reading §1's table against it:

| | Δ mean@16 | in σ | Δ maj@16 | in σ |
|---|---|---|---|---|
| lr 5e-5, on − off | +0.33 | 0.5 σ | +1.52 | 1.1 σ |
| lr 2e-5, on − off | +2.00 | **3.1 σ** | +3.57 | **2.6 σ** |

**At lr 5e-5 the gap is exactly the size of re-evaluating the same weights
twice.** At lr 2e-5 it is not. The earlier section title — "the effect is not
measurement noise" — was true of one of the two settings and I stated it of
both; it is corrected above.

Two things this floor does *not* license:

- It is **not** a finding of "no effect". A floor of 0.64 σ means the experiment
  cannot see +0.33, not that +0.33 is absent. Detecting an effect that size at
  80% power would take ≈30 independent replicate evaluations per arm
  (7.85·σ²/Δ² with σ=0.64, Δ=0.33), i.e. ~28 hours of 2×H200 time to settle
  0.33pp. The accuracy endpoint is the wrong instrument, not the verdict.
- It counts **generation noise only**. Training-seed variation — a second run of
  the same arm — is not measured here and can only widen it. Every single-run
  arm-vs-arm comparison in this document inherits that gap. The next subsection
  closes it, and the gap turns out to be the dominant one.

### Run-to-run variation, from the wreck of 59911 (2026-10-31)

Job 59911 re-ran the lr 5e-5 `on` arm and died on its wall at step ~24, leaving
no weights but four validation points. It used the **same seed (42)**, the same
lr, the same 4 GPUs and the same code as the original `on` arm (59174). It is
therefore a replicate of 59174 under identical settings, differing only in CUDA
and vLLM nondeterminism and in node contention. That makes the comparison that
matters available for the first time — *is the on-vs-off gap bigger than the gap
between two runs that should have been the same run?*

| step | on − off (the claimed effect) | | on − on (same config, twice) | |
|---|---|---|---|---|
| | mean@16 | maj@16 | mean@16 | maj@16 |
| 5 | +0.18 | −0.45 | −4.73 | −2.61 |
| 10 | −0.03 | −0.61 | +0.92 | +2.09 |
| 15 | +2.08 | +1.82 | −1.25 | +0.07 |
| 20 | −0.39 | +5.46 | −4.05 | +2.02 |
| **mean \|·\|** | **0.67** | **2.09** | **2.74** | **1.70** |

On mean@16 two identical runs disagree with each other **4.1× more** than the
`on` arm disagrees with the `off` arm. On maj@16 they disagree by the same order
(0.8×). Neither reading leaves room to resolve a +0.33 / +1.52 effect.

Since the seed is *identical*, this is a **lower bound** on seed-to-seed
variation, not an estimate of it. And it is ~4× the generation floor measured
above (0.64 mean@16), which means the dominant noise source in this experiment
is the training run itself, not the evaluation — so the σ column in the table
above uses the wrong denominator. Against run-to-run spread, neither learning
rate clears the floor:

| | Δ mean@16 | vs generation sd 0.64 | vs run-to-run \|·\| 2.74 |
|---|---|---|---|
| lr 5e-5, on − off | +0.33 | 0.5 σ | **0.1×** |
| lr 2e-5, on − off | +2.00 | 3.1 σ | **0.7×** |

**Caveats, and they are not small.** n = 4 matched steps; steps within a run are
autocorrelated, so this is a crude spread and not a clean sd. 59911 never reached
step 30, so the headline step is *not* covered — steps 5–20 are the noisy early
phase and may overstate the late-training spread. A proper answer needs a second
full run to step 30, which is now the highest-value queued item rather than a
footnote under "still open". What this does establish is that the error bars used
everywhere above are the wrong ones, and that the correction goes against the
effect, for the third time in this document.

## 2. But the action-dependent share of the return is 0.22%

`probe/distil_return_decomp.py`. The branch design is the one from
[opd_horizon/ACTION_CREDIT_2026-09-27.md](../opd_horizon/ACTION_CREDIT_2026-09-27.md):
roll out once per prompt, cut at the highest-entropy position, force each of the
top-4 tokens there, then sample K continuations from each forced action with a
shared prefix. What changes is the reward — DistIL's, not OPD's:

```python
def distil_token_reward(s_logits, t_logits, topk):
    s_logp_all = F.log_softmax(s_logits.float(), dim=-1)
    t_logp_all = F.log_softmax(t_logits.float(), dim=-1)
    idx = s_logits.float().topk(topk, dim=-1).indices    # STUDENT's top-K, as in training
    s_d = add_tail(s_logp_all.gather(-1, idx))
    t_d = add_tail(t_logp_all.gather(-1, idx))
    return (-(t_d.exp() * s_d).sum(-1)).clamp(min=0.0)   # forward CE, clamped
```

Per branch point s, with w = π(a|s) renormalised over the forced actions:

    I_pi(s) = Σ_a w_a (Q(s,a) − Q̄)²      action signal
    N_pi(s) = Σ_a w_a Var_k(return)/K     MC noise in Q̂
    share   = mean_s I_pi / Var_total

**The bias correction is the whole result.** Q̂_a is a K-sample mean, so the
weighted spread of the *estimates* is inflated by their own sampling noise:

    E[ Σ_a w_a (Q̂_a − Q̂̄)² ] = I_pi + (1 − Σ_a w_a²) · N_pi

Reporting the raw spread reads the Monte-Carlo floor as action signal. Here it
inflates the answer nearly fivefold.

Student = `on` at step 50. Two independent runs at different sample budgets; the
second (job 59473) is ~8× the samples per action and is the one to quote:

| | K=16, 24 branches, 512 tok | **K=128, 32 branches, 1024 tok** |
|---|---|---|
| returns scored | 1536 | 16384 |
| return mean / sd | 0.9149 / 0.1570 | 0.8280 / 0.1767 |
| Var_total | 0.024657 | 0.031240 |
| I_pi, naive (biased) | 5.682e-4 → 2.30% | 1.272e-4 → 0.41% |
| I_pi, bias-corrected | 1.188e-4 → **0.48%** | 6.936e-5 → **0.222%** |
| 95% bootstrap CI on the share | [−0.250%, +1.344%] | **[+0.122%, +0.329%]** |
| P(I_pi > 0) | 0.888 | **1.000** |
| N_pi | 6.446e-4 | 8.048e-5 |
| N_pi · K (K-invariant) | 1.0314e-2 | 1.0301e-2 |
| required critic R² | 0.9952 | **0.9978** |
| mean Q spread across actions | 0.0570 | 0.0258 |

The two runs agree where they must: `N_pi · K`, the per-sample noise variance,
is 1.031e-2 in both — an 8× change in K moves N_pi by exactly 8×, which is the
estimator behaving as designed. The K=16 point estimate (0.48%) sits just above
the K=128 interval, which is what an estimate with CI [−0.25%, +1.34%] is
entitled to do. The prompt sets and continuation lengths also differ, so these
are not quite the same estimand.

**What the high-power run changes.** At K=16 the action signal was not
distinguishable from zero (P = 0.888). At K=128 it is: P(I_pi > 0) = 1.000, CI
strictly positive. So the credit is **real** — and it is **0.22% of the return's
variance**, requiring a critic at R² = 0.9978 to extract. OPD's comparable
numbers (horizon T, length-normalised) are share **1.69%**, required R² 0.9831,
measured critic R² ≈ 0.25, advantage R² = −2.797.

That sharpens the puzzle rather than resolving it: **DistIL has roughly 8× *less*
action-attributable credit than OPD does**, and OPD's future term did not help
while DistIL's does. Whatever drives §1, more credit is not it.

**My hypothesis is dead.** I expected DistIL's expectation-based reward
(−Σ_v π_T log π_θ over the student's top-K) to be a lower-variance estimator than
OPD's single-sample `log p_T(a_t) − log p_S(a_t)`, and therefore to leave a
larger action-dependent share. It leaves a smaller one. Changing the divergence
does not buy attributable credit.

Reproduce:

    source /opt/anaconda3/2022.05/etc/profile.d/conda.sh; conda activate distil
    python -I probe/distil_return_decomp.py \
        --student models/on50-merged --teacher models/Qwen3-4B-Instruct-2507 \
        --data SDPO/datasets/sciknoweval/chemistry/train.json \
        --n_prompts 24 --n_actions 4 --k 16 --max_new_tokens 512 --topk 100

Raw returns and branch metadata are saved to `probe/distil_decomp.pt`; the stats
can be recomputed from it alone with `decompose(d['scored'], d['branches'], 16)`.

The K=128 run is `probe/distil_decomp_hi.pt`, submitted as
`scripts/local/decomp_hi.sh` (job 59473, one A100, ~2.6 h).

## 3. The contradiction

If only 0.22% of the future return's variance is attributable to the action, the
gradient of Term 2 is almost entirely a function of the *state*, not of what the
policy did. Then removing it should cost nothing — and at lr 5e-5 it does cost
nothing that this experiment can see (+0.33 mean@16, 0.5 σ). The contradiction
lives entirely at **lr 2e-5**, where removing it costs +2.0 mean / +3.6 maj@16,
or 3.1 σ / 2.6 σ. One setting out of two, on a single seed, is a thin place to
stand a mechanism argument — which is the point of §6.

One explanation died already. **It is not exploration.** The obvious story — the
future term keeps the policy broad and that is where the accuracy comes from — is
falsified: over the 30 compared steps at lr 5e-5, the `on` arm's `actor/entropy`
is *lower* than `off`'s (1.0092 vs 1.0217, mean difference −0.0125, `on` higher on
only 13/30 steps).

What is left: the term may work as a **position-wise common mode**. R_t is
dominated by "how much teacher-matching cost remains from here", which is a
property of t and of the state, not of a_t. Multiplying log π_θ(a_t) by that
quantity reweights the local CE gradient by position — late tokens, where little
remains, get down-weighted; early tokens get up-weighted. That is a curriculum
over positions, not credit assignment, and it has nothing to do with Proposition 1.

## 4. The split: an additive decomposition of Term 2

Both readings predict the same `on`-vs-`off` gap, so the gap cannot separate them.
Split Term 2 additively instead:

    log π_θ(a_t) · R_t  ≡  log π_θ(a_t) · (R_t − b(s_t))  +  log π_θ(a_t) · b(s_t)
                              └── credit ──┘                  └── common mode ──┘

`b(s_t)` is the **leave-one-out mean over the other sequences at the same
position**. Excluding a sequence's own return is what makes b a state-only
quantity: it cannot depend on a_t, so the first term stays an unbiased score
function and the second carries no action information at all.

In `verl/trainer/ppo/core_algos.py`:

```python
_pg_mode = os.environ.get("DISTIL_PG_MODE", "on")
if _pg_mode == "off":
    pg_per_token = torch.zeros_like(student_log_probs)
elif _pg_mode in ("baseline", "common"):
    with torch.no_grad():
        m = loss_mask.to(future_returns.dtype)
        masked_R = future_returns * m
        cnt = m.sum(dim=0, keepdim=True)
        tot = masked_R.sum(dim=0, keepdim=True)
        b = ((tot - masked_R) / (cnt - m).clamp(min=1.0)) * m
        metrics["distil/baseline_mean"] = ((b * m).sum() / m.sum().clamp(min=1.0)).item()
    pg_per_token = student_log_probs * ((future_returns - b) if _pg_mode == "baseline" else b)
else:
    pg_per_token = student_log_probs * future_returns
```

Verified standalone before submitting: `baseline + common == on` elementwise, the
LOO value matches a direct computation, masked cells are exactly zero, and a
sequence's own return is excluded from its own baseline.

| arm | Term 2 | job |
|---|---|---|
| on | `log π · R` | 59174 (done) |
| off | `0` | 59175 (done) |
| baseline | `log π · (R − b)` — credit only | 59486 |
| common | `log π · b` — common mode only | 59487 |

All four: lr 5e-5, 30 steps, `save_freq=30`, per-question dumps to
`valdump/<arm>/`, 4× RTX PRO 6000, everything else identical.

**The reading, fixed before the results land:**

- **`common` reproduces most of `on`'s gain** → the future term is not credit.
  Consistent with 0.48%, and Proposition 1's credit-assignment account is not the
  mechanism that produces the number in the paper. The OPD conclusion stands.
- **`baseline` reproduces it** → the credit is real and the decomposition in §2
  is missing something. The OPD conclusion has to be revisited, and the open
  question becomes why forward CE has credit that reverse KL does not.
- **Both partially** → two mechanisms, and the ratio is the answer.

Either way the §1 observation survives. What is at stake is what to call it.

### What actually happened: the split does not run

**`common` diverged at step 13** and never produced a usable arm. `grad_norm`
went 2.54 → 0.107 → exactly 0.0, and accuracy was 0.00 from step 15 to the 12h
wall:

    step  5  mean@16 49.91   maj@16 63.49
    step 10  mean@16 63.60   maj@16 66.81
    step 15  mean@16  0.00   maj@16  0.00
    step 20  mean@16  0.00   maj@16  0.00
    step 25  mean@16  0.00   maj@16  0.00

The identity `log π·R ≡ log π·(R−b) + log π·b` is exact, but that does not make
each half a trainable objective. `b(s_t)` is *identical across sequences at a
given position* by construction, so `log π·b` applies the same positive weight
to every sampled token and suppresses all of them indiscriminately — it is a
uniform likelihood-decay term with no contrast. The design error was assuming an
additive split of a gradient yields two gradients you can each descend. It does
not, and the `common` arm is withdrawn rather than reported.

**`baseline` ran to step 30 and is indistinguishable from `off`.** Paired over
the 210 questions against the off@30 re-evaluation:

| A − B | Δ mean@16 | CI95 | distinguishable |
|---|---|---|---|
| base − off@30 | −19.70 | [−24.52, −14.97] | yes |
| base − baseline | −18.90 | [−23.60, −14.26] | yes |
| base − revKL | −5.74 | [−10.80, −0.71] | yes |
| **off@30 − baseline** | **+0.80** | **[−2.02, +3.57]** | **no** |
| off@30 − revKL | +13.96 | [+9.49, +18.57] | yes |
| baseline − revKL | +13.15 | [+8.15, +18.36] | yes |

(`scripts/local/pairwise30.py`, 20000 bootstrap draws over questions.)

So the credit-only arm neither reproduces nor loses `on`'s gain — it lands on
`off`, with a ±2.8pp paired interval that spans the entire effect being argued
about. **§4 cannot answer its question.** The replacement arm `shuffle`
(`DISTIL_PG_MODE=shuffle`, which permutes `future_returns` across sequences
within a position, preserving the per-position multiset while destroying the
sequence↔return pairing) is job 59914, but it is subject to the same floor and
the same confirmation-shaped design noted at the top.

### Selection rule changes the headline

Step 30 is not each arm's best checkpoint. The forward CE arms are still rising
at 30; reverse KL peaks at step 10 and then collapses ~31pp by step 15.

| comparison | at fixed step 30 | at each arm's own best |
|---|---|---|
| fwdCE − revKL, mean@16 | +13.99 | **+5.36** |
| fwdCE − revKL, maj@16 | +11.57 | **+5.99** |
| on − off | unchanged | unchanged |

The forward-CE-beats-reverse-KL headline is between **2× and 3× smaller** under
best-checkpoint selection than at a fixed step. The future-term gap is not
affected either way. This is the same trap as the paper's own
best-checkpoint-up-to-100 reporting in §5, and the same one recorded in
`opd_horizon/HORIZON_MECHANISM_2026-10-01.md` §1.

## 5. Forward CE vs reverse KL is a learning-rate artifact

`scripts/science/run_sdpo_science.sh:47` sets `ALPHA=0.5`. In
`compute_self_distillation_loss` (`core_algos.py:1139-1146`) alpha is the mixing
weight in `F.kl_div(..., log_target=True)`: `alpha=0.0` is forward KL, `alpha=1.0`
is reverse KL, and `0.5` is a generalized JSD. So the baseline is **not reverse
KL**. On top of that the SDPO script runs lr 1e-5 against DistIL's 5e-5 and
`teacher_update_rate` 0.05 against 0.01. Three things move at once; the effect of
the divergence cannot be read off it.

**Checked against the paper itself, not the repo (2026-11-03).**
[arXiv:2606.05152](https://arxiv.org/abs/2606.05152), Agrawal, Fein-Ashley and
Rashidinejad; the repo owner is the first author, so it is the official code.
Appendix C (Science) lists SDPO at lr 1x10^-5 against DistIL at 5x10^-5, EMA
0.05 against 0.01, Top-K 100 for both, and the divergence as **Jensen-Shannon
for both arms**. The body says: "For Science and Coding benchmarks, we follow
the implementation of SDPO and use their reported hyperparameters for
baselines." So the lr gap and the untuned baseline are stated in the paper, not
inferred from the code.

**A correction to what I wrote earlier.** I had recorded "README says reverse KL,
code runs JSD" as a repo defect. It is not: Appendix C says Jensen-Shannon too,
so the code matches the appendix. The inconsistency is internal to the paper —
the body (SS2, SS6.3) calls the baseline "SDPO (reverse-KL)" while the appendix
specifies JSD. I had asserted this about "the paper" before reading it, from the
repo alone.

Jobs 59468 (`alpha=1.0`) and 59469 (`alpha=0.0`) are that comparison with
everything else held at DistIL's settings, including
`full_logit_distillation=True, distillation_topk=100`. Keeping full logits on
matters: with it off, only sampled-token logprobs are available and the
single-sample reverse-KL estimator is the only computable thing, which would
change the divergence **and** the information available at the same time.

**Reverse KL (59468) is done, and the divergence matters enormously.** Same lr
5e-5, same `teacher_update_rate=0.01`, same batch, same 30 steps, full logits on:

| step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| mean@16 | 59.20 | 64.85 | 33.72 | 54.05 | 57.65 | **56.22** |
| maj@16 | 67.16 | 67.37 | 53.61 | 60.22 | 60.90 | **61.49** |

At step 30 that is **+6.2 mean@16 / +7.8 maj@16 over base**, against forward CE's
**+20.2 / +19.4**. Reverse KL recovers under a third of the gain, and it is not a
clean curve either — step 15 collapses to 33.72 mean@16 and the arm never returns
to its step-10 level. (The run ends with a `DataLoader worker ... killed by
signal` traceback at teardown, after the step-30 validation and the checkpoint
save; the job state is COMPLETED and all 30 steps are present.)

### The gap closes when reverse KL gets its own learning rate (2026-10-31)

The paragraph that stood here claimed the paper's direction "survives being
controlled, and by a much larger margin than the paper reports." That was wrong,
and wrong in the specific way flagged at the top of this document: I compared
reverse KL at *forward CE's* tuned learning rate and read the resulting collapse
as a property of the divergence.

Job 59912 is the same arm at lr 2e-5, everything else identical:

| step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| mean@16 | 52.53 | 60.68 | 65.77 | 64.94 | 66.96 | **69.64** |
| maj@16 | 54.99 | 63.26 | 66.66 | 65.66 | 70.42 | **72.72** |

Monotone, no collapse at step 15, and at step 30 it lands on forward CE.
Paired across the same 210 questions, 20000 bootstrap draws:

| A − B | Δ mean@16 | CI95 | distinguishable |
|---|---|---|---|
| off@30 − revKL lr 5e-5 | +13.96 | [+9.49, +18.57] | yes |
| **off@30 − revKL lr 2e-5** | **+0.54** | **[−4.52, +5.60]** | **no** |
| baseline − revKL lr 2e-5 | −0.27 | [−5.51, +4.91] | no |
| revKL lr 5e-5 − revKL lr 2e-5 | −13.42 | [−18.69, −8.24] | yes |

The entire +13.96 "forward CE beats reverse KL" effect is the distance between
two learning rates of the *same* divergence. lr 5e-5 is simply past reverse KL's
stability edge: the step-15 collapse to 33.72 is a divergence event, not a
property of the objective.

So the correct statement is the weaker one: **the paper's headline comparison is
uncontrolled, and controlling it does not reproduce the claim.** At one learning
rate each, with the lr tuned per arm, forward CE and reverse KL are
indistinguishable on this task at step 30. Two points per arm is still not a
sweep — 59913 (lr 1e-5) adds a third, and nothing here rules out that forward CE
tolerates a wider lr band, which would be a real and reportable advantage. But it
would be an *optimization* advantage, not evidence that the forward-CE future
term carries credit the reverse-KL objective cannot.

This also removes the one result that appeared to separate DistIL from OPD. It
does not reinstate the opposite conclusion; it returns the question to open.
Forward KL (59469) is the third divergence point and is still held.

### The full grid: divergence x future term (2026-11-03)

The paper only ever runs two cells of a 2x2. Its "DistIL" is forward CE **with**
the future term; its "SDPO" baseline is reverse KL **without** it. The divergence
and the future term move together, so neither of its two claims — that forward CE
is the better divergence, and that the future term helps — is isolated by the
comparison it reports. Two cells were never run by anyone: reverse KL *with* the
future term, and forward KL at all.

`DISTIL_DIV` (`core_algos.py:1258-1281`) selects which divergence supplies Term 1
*and* the per-token reward that Term 2 integrates:

```python
_div = os.environ.get("DISTIL_DIV", "ce")
if _div == "revkl":
    div_per_token = F.kl_div(teacher_distill_log_probs, student_distill_log_probs,
                             reduction="none", log_target=True).sum(-1)
elif _div == "ce":
    div_per_token = -(teacher_distill_log_probs.exp() * student_distill_log_probs).sum(-1)
ce_per_token  = torch.clamp(div_per_token, min=0.0)          # Term 1
token_rewards = torch.clamp(div_per_token.detach(), min=0.0)  # integrand of Term 2
```

With the variable unset the path is byte-identical to the published one — the old
code recomputed the same CE expression inside the `no_grad` block, so folding it
into `div_per_token.detach()` changes nothing. The `revkl` branch is the same
expression as `compute_self_distillation_loss`'s `alpha=1.0` branch; checked
against the closed form `(s.exp() * (s - t)).sum(-1)` and for non-negativity
(`scripts/local/t_div.py`).

Step-30 mean@16, in percent. Base is 50.03 (59449) / 50.48 (59467's per-question
dump). Every cell is **one run**:

| divergence | future term | lr 5e-5 | lr 2e-5 |
|---|---|---|---|
| forward CE | **on** | 70.21 (59174) | 69.35 (59196) |
| forward CE | off | 69.88 (59175) | 67.35 (59267) |
| reverse KL | off | 56.22 (59468) | 69.64 (59912) |
| reverse KL | **on** | **66.25** (60384) | **65.89** (60385) |
| forward KL | off | 60445 — running | 60446 — queued |

(Submitted 2026-11-03, 4x H200, 8h wall, `save_freq=15`, per-question dumps to
`valdump/<arm>/`. Scripts: `scripts/local/{revkl-fut,fwdkl}-lr{5e5,2e5}.sbatch`.
The first four submissions, 60380-60383, all died in `init_model` on a CPU-RAM
OOM: I wrote fresh sbatch files requesting 16 CPUs / 80 GB where every working
arm in this document used 32 / 400 GB. Resubmitted as 60384-60387.)

### Reverse KL with the future term, lr 5e-5 (60384, 2026-11-03)

| step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| mean@16 | 50.18 | 66.61 | 67.32 | 63.33 | 64.76 | **66.25** |
| maj@16 | 60.84 | 67.48 | 70.15 | 65.02 | 67.91 | **67.11** |

Against reverse KL *without* the future term at the same lr (59468: 56.22 /
61.49) that is **+10.03 mean@16 / +5.62 maj@16** — four times the run-to-run
spread, and the largest future-term effect anywhere in this document.

**But it is almost certainly stability, not credit.** 59468 is the arm that
collapsed to 33.72 mean@16 at step 15 and never recovered: lr 5e-5 is past
reverse KL's stability edge (§5). 60384's curve has no such collapse. So what
the future term appears to buy here is survival at a learning rate the bare
objective cannot take — the same confound that produced the bogus +13.96 in §5,
with the arms swapped.

**lr 2e-5 was the test that decides it**, and it came back against the effect
(60385, 2026-11-03). There reverse KL without the future term is healthy and
monotone (59912: 69.64). With the future term it is **65.89**.

| A - B | delta mean@16 | CI95 | distinguishable |
|---|---|---|---|
| revKL+future - revKL, lr 2e-5 | **-3.75** | [-8.69, +1.13] | no |
| revKL+future - revKL, lr 5e-5 | +10.03 | [+4.38, +15.68] | yes |
| revKL+future lr 2e-5 - fwdCE off lr 5e-5 | -4.29 | [-8.10, -0.45] | yes |
| revKL+future lr 2e-5 - revKL+future lr 5e-5 | -0.36 | [-5.74, +4.97] | no |

So the +10.03 is a stability artifact exactly as predicted: it is the distance to
a *collapsed* arm, not the value of the future term. On a healthy arm the point
estimate is **negative**, and the interval spans zero. The reverse-KL row says
the same thing the forward-CE row says, only with the sign flipped and the
interval wider.

This is the fourth headline in this document moved by a measurement, and the
fourth to move against the effect. It also closes the question the §6 reframe was
built around in the cheapest possible way: the future term does not do something
under forward CE that it fails to do under reverse KL. Under both divergences,
at the learning rate where the arm is actually trainable, its effect is inside
the noise.

(60385 died in its step-30 checkpoint save when `/home` filled to 0 bytes, but
all six validation dumps including step 30 were already written, so the arm is
complete as a measurement. The disk was recovered by deleting 210 GB of
`optim_world_size_*_rank_*.pt` optimizer shards, which are resume-only state and
these runs must never be resumed — see "Traps in this repo".)

What the filled cells already say, before the new ones land:

- The **future term** moves mean@16 by +0.33 (lr 5e-5) and +2.00 (lr 2e-5). Two
  same-seed runs of one configuration differ by 2.74. The effect is inside that.
- The **divergence** looks worth 13.96 at a shared lr 5e-5 and 0.54 once each
  arm gets its own lr. That one is a learning-rate artifact.

The two new rows test the part neither of those touches: whether the future term
does anything *different* under reverse KL than under forward CE. If the paper's
Proposition-1 story is right, forward CE should be where the term pays and
reverse KL should be where it does not — that is the OPD result restated. If the
term is a position-wise common mode (§3), it should behave the same under both.

### The reverse-KL learning-rate bracket is closed (60447, 60448, 60449, 2026-11-04)

Two runs at 5e-5 and 2e-5 do not locate a peak; they only say the two points are
similar. The bracket needed a point below and a point above on both reverse-KL
rows. Those landed on 2026-11-04.

Step-30 mean@16, per-question dumps (`valdump/<arm>/30.jsonl`, 210 questions x
16 samples). Bold is each row's best:

| divergence | future term | lr 1e-5 | lr 2e-5 | lr 5e-5 | lr 1e-4 |
|---|---|---|---|---|---|
| forward CE | **on**  | — | 69.35 (59196) | **70.21** (59174, log only) | diverged |
| forward CE | off     | — | 67.35 (59267) | **70.18** (59175) | diverged |
| forward KL | off     | — | 62.11 (60446) | **71.46** (60445) | — |
| forward KL | **on**  | — | — | 60474 — running | — |
| reverse KL | off     | 62.23 (60447) | **69.64** (59912) | 56.22 (59468, collapsed) | — |
| reverse KL | **on**  | 63.48 (60448) | 65.89 (60385) | **66.25** (60384) | 0.0 (60449, collapsed) |

(59174's step-30 number is the training log's; 59911 died at step ~24 so that arm
has no step-30 dump and cannot enter a paired comparison. 59175's dump gives
70.18 where its training log gave 69.88 — the dump is the number used everywhere
in this document.)

**Above the bracket, reverse KL + future collapses outright.** 60449 at lr 1e-4
scored 61.19 at step 5 and **0.00** at steps 10 and 15, with `actor/entropy`
falling to 6.4e-6 and `global_seqlen/mean` pinned at the generation cap — the
policy stopped producing anything scoreable. Cancelled at step 18; the remaining
steps could not change the row.

**Below the bracket, the future term is worth +1.25.** 60448 at lr 1e-5 scored
63.48 against 62.23 for the same lr without it (60447), paired over questions:

| A - B | delta mean@16 | CI95 | distinguishable |
|---|---|---|---|
| revKL+future - revKL, lr 1e-5 | +1.25 | [-1.55, +4.14] | no |
| revKL+future - revKL, lr 2e-5 | -3.75 | [-8.69, +1.13] | no |
| revKL+future - revKL, lr 5e-5 | +10.03 | [+4.38, +15.68] | yes (comparator collapsed) |
| revKL+future best - revKL best | **-3.39** | [-6.82, +0.00] | no |

The two rows peak in different places — reverse KL alone at 2e-5 (69.64), reverse
KL with the future term at 5e-5 (66.25) — so the honest contrast is best against
best, and it is **-3.39 with an interval that touches zero from below**. At no
learning rate where both arms are trainable does the future term help. The only
positive reading in the row, +10.03, is the distance to a collapsed arm.

**Forward KL is now the best single cell in the grid, at 71.46.** It is also the
cheapest check on run-to-run variation this document has. `fwdCE off` and
`fwdKL off` are the *same objective*: CE = forward KL + H(pi_T), and H(pi_T) does
not depend on theta, so the two differ by a constant and have identical
gradients. They were nonetheless run on different code paths — 59175 on the
distil path with `DISTIL_PG_MODE=off`, 60445 on the sdpo path with `alpha=0.0` —
and scored 70.18 and 71.46. That **1.28** is a second, independent estimate of
run-to-run spread, from two runs that are mathematically obliged to agree. It
sits beside the 2.74 measured in §1, and it is larger than every future-term
effect in the forward-CE row.

Each divergence, at its own best learning rate, paired over questions:

| A - B | delta mean@16 | CI95 | distinguishable |
|---|---|---|---|
| forward KL - forward CE (both future-off) | +1.28 | [-2.08, +4.79] | no |
| forward KL - reverse KL | +1.82 | [-3.51, +7.14] | no |
| forward CE - reverse KL | +0.54 | [-4.52, +5.60] | no |
| forward KL - base | +20.98 | [+16.01, +25.98] | yes |

All three divergences land in the same place once each is given its own learning
rate, and all three are far above the untrained model. The paper's first claim —
that forward CE is the better divergence — is closed: it was a learning rate.

### Forward KL with the future term (60474, 2026-11-04)

The last cell, and the sharpest one. Term 2 of Proposition 1 integrates the
*value* of the per-token cost as a reward, so swapping CE for forward KL changes
Term 2 while leaving Term 1's gradient untouched — the two costs differ by
H(pi_T), which is theta-independent. Forward CE + future and forward KL + future
are therefore two trainable objectives with the same Term 1 and a Term 2 that
differs by exactly the teacher-entropy term, which is action-independent at each
position.

That is the credit-vs-common-mode split of §4 (which failed to run as a training
signal) posed instead as a comparison between two real objectives. If the future
term works by assigning credit to actions, removing an action-independent
component of its return should not matter. If it works as a position-wise common
mode, it should.

Verified numerically before launching (`scripts/local/t_div.py`): the `fwdkl`
branch matches `(t.exp() * (t - s)).sum(-1)`, is non-negative everywhere, and
`ce == fwdkl + H(pi_T)` holds — mean 2.8817 = 0.9076 + 1.9742.

Run at lr 5e-5, where both forward rows peak.

It finished at 65.71 (60474, 4h28m, entropy 1.03-1.23 throughout, no collapse).
Against forward KL *without* the future term at the same learning rate (71.46)
that is **-5.74, CI [-8.81, -2.71]** — the first future-term contrast anywhere in
this document whose interval excludes zero, and it is negative.

It does not survive the next paragraph.

### Every future-term contrast flips sign between checkpoints (2026-11-04)

60474 and 60445 are two healthy runs that differ only in the future term. Their
paired across-question difference at each of the six validation checkpoints:

| step | (+future) - (off) | CI95 | excludes 0 |
|---|---|---|---|
| 5  | -3.36 | [-5.71, -1.04] | yes |
| 10 | **+30.06** | [+26.07, +33.99] | yes |
| 15 | +0.24 | [-3.48, +3.87] | no |
| 20 | +6.25 | [+1.04, +11.58] | yes |
| 25 | -0.27 | [-3.57, +3.10] | no |
| 30 | **-5.74** | [-8.81, -2.71] | yes |

Four of six "significant", in both directions, from the same pair of runs. Step
10 and step 30 differ by 36 points. Neither arm collapsed: `actor/entropy` stays
near 1.0 in both for all 30 steps, and neither run's own curve is monotone
(60474: 48.15, 57.62, 67.80, 63.45, 69.88, 65.71).

The reverse-KL pairs do the same thing, more quietly:

| | step 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| revKL 2e-5 | -0.36 | +0.92 | +3.69 | +0.12 | **+5.42** (sig) | **-3.75** |
| revKL 1e-5 | -0.48 | +0.45 | +2.02 | +1.49 | **-4.76** (sig) | **+1.25** |

Each of those rows contains one checkpoint at which the future term is
"significantly" good or bad, and the sign of the step-30 number disagrees with
the sign of the step-25 number in both.

**This invalidates the error bar used throughout this document.** The paired
across-question bootstrap is a *within-checkpoint* interval: it answers "given
these two sets of weights, how sure am I which scores higher on this question
distribution". It says nothing about where along the run those weights were
taken, and that choice moves the answer further than the interval is wide. §1
measured two other components of the error — generation noise (0.64) and
run-to-run spread at fixed seed (2.74) — and the 1.28 between `fwdCE off` and
`fwdKL off` added a third. Checkpoint choice is a fourth, and it is the largest.

So the -5.74 above is withdrawn: it is one checkpoint of six, and three of its
neighbours say the opposite. The same applies to every single-checkpoint number
in this document, including the +10.03 (already attributed to a collapsed
comparator) and the +0.33 / +2.00 forward-CE figures in §1.

This is the fifth headline in this document overturned by a measurement, and the
first that is methodological rather than about the effect: it says that most of
the preceding intervals were measuring the wrong thing. The conclusion in §1 —
that this testbed cannot resolve the effect it was built to measure — survives,
and is now better supported than when it was written, for a worse reason.

A practical rule follows for anything downstream: **no arm-vs-arm claim from a
single fixed step.** Either select each arm's best checkpoint, which is what the
next subsection does and what the paper implicitly does, or report the spread
across checkpoints. Reading step 30 because it is last is the one thing that is
not defensible. The cost is zero — the dumps are already written every 5 steps.

### Best against best (2026-11-05)

The comparison the paper makes, and the only one that is fair to every arm, is
each configuration at *its own* best point — best learning rate **and** best
checkpoint. Reading step 30 fixes the checkpoint axis arbitrarily, and the
previous subsections did exactly that.

Every chemistry arm's validation curve, mean@16 in percent (210 questions x 16
samples per cell, `valdump/<arm>/<step>.jsonl`):

| arm | 5 | 10 | 15 | 20 | 25 | 30 | best |
|---|---|---|---|---|---|---|---|
| fwdCE +future 5e-5 (`on30`) | 49.26 | 64.17 | **70.57** | 65.77 | — | — | 70.57 @15 |
| fwdCE off (`off30`) | — | — | — | — | — | 70.18 | 70.18 (one point) |
| fwdKL +future 5e-5 | 48.15 | 57.62 | 67.80 | 63.45 | **69.88** | 65.71 | 69.88 @25 |
| fwdKL off 5e-5 | 51.52 | 27.56 | 67.56 | 57.20 | 70.15 | **71.46** | 71.46 @30 |
| fwdKL off 2e-5 | 55.39 | 64.26 | 68.10 | 65.03 | 61.16 | 62.11 | 68.10 @15 |
| revKL +future 2e-5 | 52.17 | 61.61 | 69.46 | 65.06 | **72.38** | 65.89 | 72.38 @25 |
| revKL +future 5e-5 | 50.18 | 66.61 | 67.32 | 63.33 | 64.76 | 66.25 | 67.32 @15 |
| revKL +future 1e-5 | 51.70 | 57.89 | 56.40 | 64.40 | 60.60 | 63.48 | 64.40 @20 |
| revKL +future 1e-4 | 61.19 | 0.00 | 0.00 | — | — | — | collapsed |
| revKL off 2e-5 | 52.53 | 60.68 | 65.77 | 64.94 | 66.96 | **69.64** | 69.64 @30 |
| revKL off 1e-5 | 52.17 | 57.44 | 54.37 | 62.92 | 65.36 | 62.23 | 65.36 @25 |
| revKL off 5e-5 | 59.20 | 64.85 | 33.72 | 54.05 | 57.65 | 56.22 | 64.85 @10 |

Row best against row best, paired over questions:

| contrast | delta | CI95 | excludes 0 |
|---|---|---|---|
| reverse KL: +future - off | **+2.74** | [-2.17, +7.68] | no |
| forward KL: +future - off | **-1.58** | [-5.27, +2.17] | no |
| forward CE: +future - off | **+0.39** | [-3.36, +4.17] | no |
| revKL +future - fwdKL off (best cells overall) | +0.92 | [-3.18, +5.18] | no |
| fwdKL off - fwdCE off (same objective) | +1.28 | [-2.08, +4.79] | no |

Three divergences, every interval across zero. **The future term has no effect
best-against-best either.** The two signs even disagree between rows: +2.74
under reverse KL, -1.58 under forward KL.

**The -3.39 reported in the preceding subsection is withdrawn.** It came from
calling 5e-5 reverse KL + future's best learning rate, which was true only at
step 30 (66.25 vs 65.89). Once the checkpoint axis is opened, that row's best is
**2e-5 at step 25, 72.38** — a different learning rate, and the contrast flips
from -3.39 to +2.74. The choice of "best learning rate" was itself a product of
the checkpoint I happened to read.

Two things make even the +2.74 unreadable. It is exactly the run-to-run spread
measured in §1 (2.74), so it is one noise unit by construction. And best-of
selection is biased upward by an amount that grows with the number of candidates:
reverse KL + future was offered 4 learning rates x 6 checkpoints, forward CE +
future only 1 learning rate x 4 checkpoints, because 59911 died at step ~24. The
arm with the most lottery tickets won. `off30` is worse still — a single step-30
re-evaluation with no curve at all, so the forward-CE row never got to select.

An honest version of this table needs each arm run to the same number of
checkpoints at the same set of learning rates, and a held-out split to select on
so that "best" is not chosen on the same 210 questions it is scored on. Neither
holds here. That is the §1 conclusion again, reached from a third direction.

### A scoring bug, and what it does and does not change (2026-11-05)

Three of the deep troughs in the curves above are not the model failing the
science. They are the grader failing to find an answer the model gave.

`SDPO/verl/utils/reward_score/feedback/mcq.py` extracts the letter as

```python
answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
```

When the response contains no `<answer>` tag at all, `split` returns the whole
response, `answer` becomes several thousand characters, and the comparison to
the gold letter fails. A response ending `"Thus, it is the only correct
reactant.\n\nFinal answer: A."` — correct, unambiguous — scores 0.

The same file has a second, cosmetic bug: `incorrect_format =
is_correct_format(solution)` sets the flag to 1 when the format is **correct**.
The field is inverted and must not be read; it is 98.6% on a step scoring 65 and
18.2% on a step scoring 16.

`scripts/local/rescore.py` re-scores every dump two ways: `strict` is the shipped
rule unchanged, `lenient` falls back to `Final answer: X`, `\boxed{X}`, `**X**`
and a bare trailing letter only when strict finds no single letter. Across ~85
arm-steps, three move:

| arm | step | strict | lenient | delta | rescued |
|---|---|---|---|---|---|
| phys-off-lr5e5 (physics) | 15 | 16.17 | **67.42** | +51.25 | 67.5% |
| revkl 5e-5 (59468) | 15 | 33.72 | **63.75** | +30.03 | 39.6% |
| fwdkl-lr5e5 | 10 | 27.56 | **44.14** | +16.58 | 22.3% |

Every other cell moves by less than 0.4. The bug does not leak a constant bias
into the scores; it fires only where the policy transiently stops emitting the
tag, which is a *format* collapse, not a knowledge collapse.

**The genuine collapses survive re-scoring.** `common` at steps 15/20/25,
`revkl-fut-lr1e4` (60449) at steps 10/15, and `revkl` 5e-5 at steps 20-30 are
degenerate repetition — tails like `"</answer>\n</answer>\n</answer>..."`,
`"<reasoning>\nA\n</reasoning>"` repeated to the cap, or a literal run of zeros.
No extractor recovers a letter from those, and `lenient` does not move them. §4's
divergence of `common` and §7's 60449 row stand.

**No best-against-best conclusion changes.** All three damaged cells were each
arm's *worst* checkpoint, so none of them was ever selected:

| row | best, strict | best, lenient |
|---|---|---|
| revKL +future | 72.38 (2e-5 @25) | **72.41** (2e-5 @25) |
| fwdKL off | 71.46 (5e-5 @30) | **71.49** (5e-5 @30) |
| fwdCE +future | 70.57 (`on30` @15) | **70.60** (`on30` @15) |
| fwdCE off | 70.18 (one point) | **70.18** |
| fwdKL +future | 69.88 (5e-5 @25) | **70.03** (5e-5 @25) |
| revKL off | 69.64 (2e-5 @30) | **69.64** (2e-5 @30) |

| contrast | strict | lenient |
|---|---|---|
| reverse KL: +future - off | +2.74 [-2.17, +7.68] | **+2.77** [-2.17, +7.68] |
| forward KL: +future - off | -1.58 [-5.27, +2.17] | **-1.46** [-5.15, +2.26] |
| forward CE: +future - off | +0.39 [-3.36, +4.17] | **+0.42** [-3.33, +4.20] |
| fwdKL off - fwdCE off (same objective) | +1.28 | **+1.31** |

That is luck, not design. Had a format collapse landed on a *peak* instead of a
trough, best-of selection would have silently compared answer-format stability
between arms and called it a divergence effect.

**It is also in the training signal.** This scorer supplies the reward, so a
correct answer written without the tag was rewarded 0 during training. Format
compliance and correctness are entangled in every number in this document, and
an arm that drifts away from the tag is punished for a reason that has nothing
to do with the objective being studied. That is a confound no re-scoring can
remove after the fact; it would need the fix applied before training.

**Why the forward-CE row has no curve, and why it cannot be given one.** 59174
(+future) and 59175 (off) were run on 2026-10-27, before `validation_data_dir`
was being passed, so neither wrote per-question dumps. `off30` is a single
`val_only` re-evaluation of 59175's surviving step-30 checkpoint (59918), and
`on30`'s four points come from 59911, the *retrain* that died at step ~24 — a
different run from 59174, whose own curve exists only in its training log.

The intermediate checkpoints are gone. 59175 used `save_freq=10` and so wrote
steps 10, 20 and 30, but verl keeps only the latest actor directory: today
`global_step_{10,20,40}` hold a 7 KB `data.pt` and nothing else, while only
`global_step_30` still has `actor/`. The weights were deleted by the run itself
on 2026-10-27, not by the disk cleanup of 2026-11-03.

Worse, **59174 and 59175 shared a checkpoint directory.** Both took the default
experiment name, which verl builds from the model path, so both resolved to
`.../DistIL-science-chemistry--home-doyoonkim-projects-distIL-models-Qwen3-4B-Instruct-2507/`.
59174 finished at 07:25 and 59175 began overwriting at 08:34. `global_step_40`
is 59174's leftover; `global_step_30` carries 59175's 10:31 timestamp. The two
arms are distinguishable now only by file mtime. Every later arm passes an
explicit suffixed experiment name and so has its own directory; this collision
affects only the two forward-CE runs, which is exactly the row that cannot be
repaired without rerunning both. That rerun was considered and declined on
2026-11-05 in favour of moving to physics (§7).

So the forward-CE row's +0.39 is not comparable to the other two rows: its
"+future" best was selected from 4 checkpoints at one learning rate and its
"off" best from a single point, against 24 and 17 candidates for the reverse-KL
row. Read it as unmeasured, not as a small effect.


## 6. The reframe

Everything above asks one question — *if I remove or damage the future term,
does accuracy drop?* — of a testbed that cannot answer it. Four problems, in
increasing order of how much they cost:

1. **The endpoint is too coarse.** The floor in §1 is σ=0.64 on mean@16. The
   effect under argument at lr 5e-5 is +0.33. No amount of care in arm design
   recovers an effect half the size of the measurement noise.
2. **Every arm subtracts.** `off`, `baseline`, `common`, `shuffle` all destroy
   signal and look for a drop. None would make a real effect *larger*, so the
   battery's best possible outcome is a null, and a null here is uninformative
   by (1).
3. **0.22% was over-read.** It is the action-attributable share of return
   variance at *forced-action branch points under a fixed reference policy*. I
   repeatedly wrote it as if it were "the amount of sequence information", which
   it is not.
4. **The task is close to worst case.** SciKnowEval chemistry is 4-way MCQ with a
   single-letter answer, and all four SciKnowEval domains have ground-truth
   length median 1. There is almost no sequence for sequence credit to act on.
   It was chosen because the repo ships it.

The sharpest omission: **the test that found the effect in OPD was never run
here.** `opd_horizon/HORIZON_MECHANISM_2026-10-01.md` §2 detected h16-over-h0 at
the level of the objective, with all four paired CIs excluding zero, while
accuracy showed nothing at all. The same instrument was available the whole time
and I kept reaching for accuracy.

### What replaces it

| | up to §5 | from §6 |
|---|---|---|
| question | does removing it hurt? | **what does it change?** |
| measurement | 210-question accuracy | **the objective J on held-out data, paired per token** |
| task | 4-way MCQ chemistry | **a task with long answers** |
| budget | step 30 | `on` is still rising at step 50 |

**(A) Branch-point policy alignment** (`probe/branch_policy_align.py`, job
59917). Build branch points and `Q̂(s,a)` **once** from a fixed reference model so
both arms are scored against the same yardstick; each arm then contributes one
forward pass per branch for `π_arm(a|s)`. The statistic is

    E_π[Q] = Σ_a w_a · Q̂(s,a)        w = π_arm renormalised over the candidate actions

— lower means the policy puts mass on actions with cheaper futures. If the
future term installs credit, `E_π_on[Q] < E_π_off[Q]`. Paired by branch point,
and **independent of the ±0.64 accuracy floor entirely**. Because the expensive
part (K continuations per branch) is shared, branch count can be pushed to 128
for roughly the cost of one arm — that is where the power comes from.

**(B) A long-answer task** (jobs 59915 / 59916). `tooluse`: multi-step action
sequences, answer length median 122 chars / max 1611, and a reward that is
action-sequence agreement, so it is **continuous with partial credit** and
earlier actions genuinely change what is reachable later. Built to
`SDPO/datasets/tooluse/{train,test}.parquet` (4046 / 68 rows) with
`python -I data/preprocess.py --data_source datasets/tooluse` run from `SDPO/`.
Caveat stated up front: **test is 68 questions**, smaller than chemistry's 210;
the continuous reward should lower per-question variance, but whether that
compensates is an empirical question to settle from the realised CI, not an
assumption.

## 7. Status

Done:

| job | arm | result |
|---|---|---|
| 59473 | decomposition, K=128 | share 0.222%, CI [0.122, 0.329], P(>0)=1.0 |
| 59468 | reverse KL, alpha=1.0 | step 30: 56.22 / 61.49, vs forward CE 70.21 / 73.06 |
| 59486 | baseline — credit only | step 30: 69.38; vs off +0.80, CI [−2.02, +3.57] — indistinguishable |
| 59487 | common — common mode only | **diverged at step 13**, withdrawn (§4) |
| 59467 | base, per-question dump | 3360 rows, 50.48%, matching 59449's 50.03% |
| 59918 | off@30 re-evaluation | 70.18 / 73.00 → the noise floor in §1 |
| 59912 | reverse KL, lr 2e-5 | step 30: **69.64 / 72.72** — ties forward CE; §5 overturned |
| 59911 | on@30 retrain | **TIMEOUT at step ~24**, no weights — but its 4 valdumps gave §1 the run-to-run spread |

Cancelled unrun on 2026-10-31, all of it chemistry: 59946 (on@30 retrain),
59947 (branch-point alignment), 59915/59916 (tooluse on/off), 59913 (reverse KL
lr 1e-5, TIMEOUT at step 25), 59914 (shuffle, killed at 25 min with no
validation point), 60016 (forward KL lr 2e-5, never started). Chemistry was the
wrong task to push further; see the next paragraph.

The 2x2 grid, filled (2026-11-03 to 2026-11-04):

| job | arm | result |
|---|---|---|
| 60384 | reverse KL **with** future term, lr 5e-5 | step 30: **66.25** / 67.11 — row best |
| 60385 | reverse KL **with** future term, lr 2e-5 | step 30: 65.89; vs no-future −3.75, CI [−8.69, +1.13] |
| 60445 | forward KL, no future term, lr 5e-5 | step 30: **71.46** — best single cell in the grid |
| 60446 | forward KL, no future term, lr 2e-5 | step 30: 62.11 |
| 60447 | reverse KL, no future term, lr 1e-5 | step 30: 62.23 — closes that row from below |
| 60448 | reverse KL **with** future term, lr 1e-5 | step 30: 63.48; vs no-future +1.25, CI [−1.55, +4.14] |
| 60449 | reverse KL **with** future term, lr 1e-4 | **collapsed**: 61.19 at step 5, 0.00 at steps 10 and 15, entropy 6.4e-6. Cancelled at step 18 — the row was decided |
| 60474 | forward KL **with** future term, lr 5e-5 | step 30: 65.71; vs no-future −5.74 at step 30 but **+0.08 pooled over steps 20/25/30** — the arm that exposed the checkpoint-flip problem |

60380–60383 died in `init_model` on a CPU-RAM OOM (16 CPUs / 80 GB requested
where every working arm here uses 32 / 400 GB); resubmitted as 60384–60387.
60386/60387 never started because `/home` hit 0 bytes free, which also killed
60385's final checkpoint save; resubmitted as 60445/60446 after reclaiming
210 GB of optimizer shards.

With 60447/60448/60449 the reverse-KL learning-rate bracket is closed on both
sides; 60474 filled the last cell. The grid's answer, once each contrast is
pooled over its last three checkpoints rather than read off step 30, is that the
future term does nothing under any of the three divergences: +0.08 / +0.60 /
−0.67, all intervals spanning zero. The single-checkpoint figures this document
carried for four revisions — including the −3.39 written two days ago — are
withdrawn in §5.

**The task was chosen on the wrong column.** The paper's per-subject DistIL - SDPO
gaps are physics **+8.4**, biology +3.0, chemistry **+0.6**, material +0.3. I
picked chemistry because its *base-to-trained* jump was the largest, which is a
different column from the contrast under study, and so spent the whole battery on
the second-smallest effect in the table. My own chemistry measurement reproduces
the paper's chemistry number exactly — paper 80.2 vs 80.8, mine 69.64 vs 70.18,
both +0.6 — which is evidence the pipeline is right and the subset is wrong.
Physics (80 test questions vs chemistry's 210) is where an 8.4-point contrast
would have somewhere to show against a 2.74 run-to-run spread. Recorded in the
`pick_condition_by_breakdown` memory note.

59911 hit its 5h wall at step ~24 on a contended PRO6000 node. `save_freq=30` on
a 30-step run saves only at the end, so a wall hit one step short costs the whole
run — the rule in memory is "never `save_freq=0`", and this is the same failure
with a different number. All four arms submitted on 2026-11-03 use
`save_freq=15`. Slurm refuses `scontrol update TimeLimit` upward for a
non-admin, so there is no in-flight rescue; the wall has to be right at
submission.

Reverse KL had been run at **one** learning rate while forward CE got three —
the mirror image of the paper's flaw in §5, and the exact trap recorded in
`opd_horizon/HORIZON_MECHANISM_2026-10-01.md` §1 ("learning rate was confounding
everything"). 59912 closed it and the §5 claim did not survive: at lr 2e-5
reverse KL ties forward CE. This is now the second headline in this document
revised by a measurement rather than by an argument — §1 by the noise floor,
§5 by the lr sweep — and both went the same direction, against the effect I had
written up. That is the pattern the top-of-document caveat predicts.

Still open:

- Out-of-domain transfer (biology / material / physics) from
  `ARM_ON_FINAL/global_step_50` and a matched `off` checkpoint.
- **A second full run to step 30.** This is now the highest-value item here, not
  a footnote: §1's run-to-run subsection shows two *same-seed* runs of the `on`
  arm disagree 4.1× more than `on` disagrees with `off` on mean@16, but it rests
  on 4 autocorrelated early steps and never reaches step 30. Until a replicate
  covers the headline step, every arm-vs-arm number in this document is reported
  against the wrong error bar. 59946 is that run.
- **A GAE critic is not planned.** At required R² = 0.9978 there is nothing for a
  critic to extract; OPD already measured advantage R² = −2.797 (worse than
  predicting the mean) and ridge R² ≈ 0.25 at 576k rows / 12k problems, with GAE
  γ=0.99 λ=0.95 landing at MATH-500 0.6721 against a batch-matched 0.6724. See
  [opd_horizon/CRITIC_AND_SPEC_2026-09-27.md](../opd_horizon/CRITIC_AND_SPEC_2026-09-27.md).
  This stays shut unless §4 comes back `baseline`.

## 8. Physics, and where the length collapse actually lives (2026-11-05 to 2026-11-07)

Three things happened here. One is a result; two are corrections to how this
document was being written.

### 8.1 The metric was wrong for four revisions

Everything above reports **mean@16**. The paper plots **Best@16** and **Maj@16**.
I had been comparing my numbers against a figure drawn on a different axis.

Re-read on Best@16, the chemistry numbers land where the paper's do: forward CE
with the future term 0.942 against the paper's ~0.94, SDPO 0.857 against ~0.86,
both at step 30. The pipeline was never the problem. Every interval in §§1–5 is
a mean@16 interval and stays one; the paper-comparison sentences that implied a
level mismatch are withdrawn.

Two details of the figure matter for anything read off it. In the physics panel
the curves overlap through step ~50, so the published +8.4 is a step-250 number
and I run 30 steps — 12% of its training. In biology and chemistry, by contrast,
a 16.5-point Best@16 gap is already open at step 10, the first plotted point,
which 320 prompts of training cannot produce.

### 8.2 The learning-rate bracket was open, again

I reported "the future term costs −6.41 in physics" from arms whose best lr was
2e-5 — the *lower* edge of a two-point grid {2e-5, 5e-5}. Every physics arm
peaked at that edge. The number could not be read and is withdrawn, together
with the chemistry Best@16 +0.050, whose two arms peaked at the *upper* edge and
one of which had run 50 steps against the other's 30, biasing a
best-of-checkpoint selection by candidate count.

This is the second time in this document — §5 already records forward-CE-beats-
reverse-KL dissolving once reverse KL got its own learning rate. The standing
rule is now in the `close_lr_bracket_first` memory note: print the lr axis before
the score, mark each arm closed or open, report only the closed ones, and submit
the missing lr in the same turn.

Physics was then filled out to 15 arms, 5 configurations x 3 learning rates, all
30 steps. Best@16, maximum over the six validation checkpoints:

| arm | 1e-5 | 2e-5 | 5e-5 | bracket |
|---|---|---|---|---|
| forward CE, future **on** | **92.05** | 87.66 | 84.10 | open (lower edge) |
| forward CE, future off | 90.73 | 90.73 | **91.60** | open (upper edge) |
| forward KL, future **on** | 85.52 | **90.33** | 90.25 | **closed** |
| forward KL, future off | **90.32** | 89.16 | 86.49 | open (lower edge) |
| reverse KL, future off | 89.75 | **90.60** | 88.96 | **closed** |

Only two arms are closed, and they differ by 0.27 — forward KL with the future
term 90.33, reverse KL without it 90.60.

That 0.27 is inside the measurement floor on this metric. `forward CE off` and
`forward KL off` are the same objective (§5), and at matched lr they differ by
0.41, 1.57 and 5.11 on Best@16. **The physics future-term contrast is not
readable from this grid.** The three missing cells — 5e-6 for the two
lower-edge arms, 1e-4 for the upper-edge one — were submitted as 60964/60965/60966
and cancelled unrun on 2026-11-06 (§8.5).

### 8.3 The sign is correct — the term is not inverted

Worth recording because it was the obvious suspect and it is not the answer.

`per_token_loss = ce_per_token + pg_per_token` where
`pg_per_token = student_log_probs * future_returns`
([core_algos.py:1372](SDPO/verl/trainer/ppo/core_algos.py#L1372)), passed through
`policy_loss = pg_loss` ([dp_actor.py:883](SDPO/verl/workers/actor/dp_actor.py#L883))
to `loss.backward()` and an optimizer step
([dp_actor.py:907](SDPO/verl/workers/actor/dp_actor.py#L907)). No negation
anywhere. Descending on `log pi * R` with `R >= 0` *lowers* the log-probability
of tokens with costly futures, which is the correct direction for minimizing
future cost.

The logs agree: `distil/pg_loss` is negative in every future-on run and grows
more negative as `returns_mean` grows (60929: −0.141 at R=0.435, −0.454 at
R=0.586). So the length blow-up is not a flipped sign.

### 8.4 The length collapse lives entirely in the common mode

This is the result. Term 2 splits additively and exactly:

```
on  =  log pi * R           =  log pi * (R - b)  +  log pi * b  =  baseline + common
```

where `b` is the leave-one-out mean of the return over the *other* sequences at
the same position — a state-only quantity that cannot depend on this sequence's
own action. `DISTIL_PG_MODE` runs each half alone. Chemistry, lr 5e-5, 30 steps,
matched pair (59486 `baseline`, 59487 `common`); their lengths agree to within
3% for the first 14 steps, so the runs are matched:

| step | 5 | 10 | 14 | 15 | 16 | 18 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| `baseline` length | 1189 | 4370 | 1681 | 2313 | 2342 | 3170 | 2645 | 3460 |
| `common` length | 1178 | 3556 | 2173 | 4439 | 6696 | 7726 | 8064 | 8022 |

`baseline` holds between 1.2k and 4.4k for all 30 steps with `clip_ratio` 0.00
throughout. `common` climbs from step 15 and pins at the 8192 cap, `clip_ratio`
0.40 → 0.98.

**The component that destroys the policy is exactly the one a baseline removes.**

The mechanism is direct. `b` is a non-negative weight identical across the
tokens sampled at a position, so every sampled token's log-probability is pushed
down by the same amount — an entropy-raising force with no preference about
*which* token was sampled, and in particular none about EOS. The `baseline` half
keeps that preference: a sampled EOS has no future, so its return is 0 and it is
not pushed down, while continuing tokens carry positive weight and are. That half
does not lengthen anything.

This corrects a claim made earlier in this work. "No baseline, so it lengthens"
is wrong as stated — an action-independent baseline does not change the expected
gradient, only the variance. The accurate statement is narrower and is what the
measurement supports: **the action-independent component of the future term is
what drives the length collapse, and a baseline removes it.** In expectation that
component is zero-mean; at batch 32, summed over ~2,000 tokens, it is not.

Caveat on scope. Three physics future-on arms blew up (forward CE 5e-5 at step
12, forward CE 2e-5 at step 19, forward KL 2e-5 at step 23; `clip_ratio` 0.84 to
1.00) and no future-off arm did. But onset is lr-dependent, 1e-5 did not blow up
within 30 steps, and SDPO at 5e-5 (60778) blew up at step 27 *without* a future
term. So "future-on blows up" is not itself established; what is established is
the `baseline`/`common` split above, which is a matched pair at one lr.

### 8.5 Two hypotheses closed, and what was left unrun

**The teacher's entropy is 75% of the published reward.** Measured per-token
costs: forward CE 0.840, forward KL 0.207, so H(pi_T) = 0.633. CE = forward KL +
H(pi_T) and H(pi_T) does not depend on theta, so Term 1's *gradient* is identical
between the two; only Term 2 differs, because Term 2 uses the cost's *value* as a
reward. Three quarters of what the published future term integrates is the
teacher's entropy at that position, which says nothing about which token the
student sampled.

**Diversity: rejected.** The hypothesis was that the future term helps by making
outputs more varied. Across six future-on/off pairs, future-on has equal or
*fewer* distinct answers per question, equal or lower answer entropy, and an
equal or *narrower* best-minus-mean gap. The sharpest case — chemistry reverse KL
2e-5, step 25 — goes the wrong way by 7.8 points. A side finding: forward-family
chemistry arms have a ~15% no-answer rate at step 5 against reverse KL's 0.2%,
and `fwdkl-lr5e5` step 10 reached 38%, which is why rescoring lifted that cell
from 27.56 to 44.14.

**Cancelled unrun 2026-11-06 23:50**, to free the cluster: 60934–60937 (chemistry
forward CE future-on, lr 1e-5/2e-5/5e-5/1e-4 — the first three were mid-run),
60938–60941 (chemistry forward CE future-off, same four lrs), 60942–60944
(physics reverse KL **with** the future term, lr 1e-5/2e-5/5e-5 — the cell the
paper never runs), 60964–60966 (the three physics bracket-closing lrs from §8.2).
All logs and valdumps of completed arms are kept; only these 14 are unrun.

The shortest path back in: physics is 15 arms deep and needs 3 more to close its
brackets; chemistry needs its 8 forward-CE cells to have a closed grid on
Best@16; and 60942–60944 remain the only arms that would test reverse KL *with*
the future term on the subject where the paper claims +8.4.

## Traps in this repo

Beyond the nine fixes needed to make it run at all (see the `distil_repo_defects`
memory note and `*.orig` backups beside each edited file), these bit us here.

**The MCQ grader loses answers that skip the `<answer>` tag.**
`verl/utils/reward_score/feedback/mcq.py` does
`text.split("<answer>")[-1].split("</answer>")[0]`, which returns the entire
response when the tag is absent, so `"Final answer: A."` scores 0. It fires on
whole checkpoints at a time and cost three cells 16 to 51 points (§5). It is
also the training reward, so the effect is baked into the runs, not only the
evaluation. `scripts/local/rescore.py` re-scores dumps after the fact; fixing
the training signal needs the patch applied before the run.

**`incorrect_format` is inverted** in the same file:
`incorrect_format = is_correct_format(solution)`. It is 1 when the format is
correct. Do not read that field.

**Two runs with default experiment names share a checkpoint directory.** verl
builds the default name from the model path, so 59174 and 59175 both wrote to
`.../DistIL-science-chemistry--home-...-Qwen3-4B-Instruct-2507/` and the second
overwrote the first (§5). Always pass an explicit suffixed experiment name.

**verl keeps only the latest actor checkpoint.** `save_freq=10` does not mean
three checkpoints survive; older `global_step_*/actor/` directories are removed
as the run proceeds, leaving a 7 KB `data.pt` behind. If an intermediate
checkpoint matters, dump its validation during the run — it cannot be recovered
afterwards.

Three more:

- **The EMA teacher is never checkpointed.** It lives in `ref_module_fsdp`
  (`fsdp_workers.py:908`), is EMA-updated in place (`dp_actor.py:132-151`), and is
  absent from the saved state. `resume_mode: auto` therefore resumes with the
  teacher silently reset to base. At rate 0.01 the teacher is ≈0.74·base +
  0.26·student by step 30, so **a resumed run optimizes a different objective than
  an uninterrupted one, with no warning.** Do not resume these runs; rerun them.
- **Two arms with the same `EXPERIMENT` name share `default_local_dir`**, and
  `max_actor_ckpt_to_keep: 1` means the second arm deletes the first's
  checkpoints. `run_distil_science.sh` now takes `EXP_SUFFIX` for this.
- `save_freq=0` leaves a finished run with no weights, which forecloses every
  follow-up that needs them — including everything in §2. All arms now use
  `save_freq=30`.
- **The configured `max_model_len` never reaches vLLM.**
  `verl/workers/rollout/vllm_rollout/vllm_async_server.py:205` unconditionally
  overwrites it with the model's `max_position_embeddings`, so the server starts
  with `--max_model_len 262144` while `--max_num_batched_tokens 10240` is passed
  correctly. vLLM then demands 18 GiB of KV cache for a single request. Invisible
  on H200/PRO6000 (0.5 × 141/96 GB leaves 40–62 GB of KV); fatal on A6000-48
  (14.18 GiB available → `EngineCore failed to start`). The one-line fix is
  `self.config.max_model_len = self.config.max_model_len or get_max_position_embeddings(...)`,
  but it is a **shared rollout path** — do not apply it while a comparison sweep
  is in flight. Raise `gpu_memory_utilization` for a one-off on a small-VRAM node.
- Scripts under `set -u` must write `${PYTHONPATH:-}`. The variable is set on some
  nodes and unset on others, so a bare `$PYTHONPATH` fails only after a job lands
  somewhere new.

## Cluster notes

PRO6000 is a **single node**. One 8-GPU 3-day job from another user blocks the
whole partition; this happened on 2026-10-30 and cost ~14 hours of zero progress
with nine jobs queued. Measured throughput for this workload, from our own
`sacct`: **H200 ≈ 3× PRO6000 ≈ 8× A100**, so when PRO6000 is full, A100 and A6000
are not an escape hatch for training — a 6.5h PRO6000 arm becomes ~17h on A100
and overruns a 12h wall. They are fine for inference-only merge/`val_only` jobs.
Submit with `--partition=H200,PRO6000` and set `--time` close to the real runtime,
or backfill will never slot the job in.
