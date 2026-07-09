# Long-horizon online controller-selection experiment (TAR-UCB)

Infrastructure to run the paper's **Section 7 (Online controller selection)** bandit
comparison over **months-scale horizons**, comparing **TAR-UCB** against fixed
controllers, a rule-based (operator-like) policy, and two common bandit baselines
(**Discounted-UCB** and **Discounted Thompson sampling**), plus an **Oracle**.

## Why this design

A literal 6-month closed loop at `f_s = 256 Hz` is ~`4e9` samples *per policy*. Holding
the injected noise in RAM (three regime streams) is ~100+ GB — impossible. So:

1. **Generate the environment noise once, store it, stream it in chunks.** The
   environment (regime schedule `W(t)`, injected test-mass noise, input power) is the
   *same for every policy* (common random numbers) — only the arm choices differ. We
   pre-blend `n(t)=Σ_i W_i(t) n_i(t)` and `P_in(t)=Σ_i W_i(t) P_i(t)` **once** and store
   the two test-mass streams + input power as `float32` memmaps on disk. The experiment
   memory-maps them and reads one RAM chunk at a time.
2. **Faithful physics.** Noise is synthesised block-by-block with the *exact* frequency-
   domain construction of `Lightsaber.Plant` (white × root-PSD × suspension TF → irfft);
   the closed loop is the validated Numba `fast_engine` kernel. Streaming (block split +
   cross-window state carry) was verified **bit-identical** to a single-shot run
   (`max|Δ| = 0`).
3. **Sensor noise** is white and cheap → regenerated deterministically per window,
   identical across policies.

## Pipeline (three steps)

```bash
PY="C:/Users/marco/anaconda3/python.exe"   # base env has numpy/scipy/numba/matplotlib

# 1. Generate + store the noise ONCE (6 months, ~48 GB float32, ~70 min).
$PY bandit_noise_cache.py --horizon 15552000 --block 4096 --period 86400 \
    --out bandit_runs/cache_6mo

# 2. Calibrate the reward (Oracle table + TAR-UCB sigma) from the current fast engine.
#    (Auto-invoked by step 3; run standalone to inspect.)  Cached to bandit_runs/calibration.npz
$PY bandit_calibrate.py --dur 600 --hold 20

# 3. Run all policies over the cached horizon (checkpointed per policy).
$PY bandit_long_experiment.py --cache bandit_runs/cache_6mo --hold 20
```

Horizons: `6 months = 15552000 s`, `1 month = 2592000`, `1 week = 604800`, `1 day = 86400`.

## Outputs (`<cache>/experiment/`)

- `policies/<name>.npz` — per-policy checkpoint: `rewards`, `arms`, `ctx`, `raw` per
  decision (+ TAR-UCB `tar_J`/`tar_mode`/`tar_Z` diagnostics). Reruns **skip** finished
  policies; an interrupted run only loses the in-progress policy. Force recompute with
  `--force`; run a subset with `--only TAR-UCB,D-UCB`.
- `summary.csv` / `summary.json` — cum reward, mean reward, regret vs Oracle, switches,
  fraction-optimal per policy.
- `fig_cumreward_regret.png`, `fig_timelines.png`, `fig_tarucb_diag.png`.

## Key parameters

- `--hold` (default 20 s): bandit decision interval `T_hold`. Reward is the mean over the
  window's continuous multi-band logistic score.
- `--block` (cache, default 4096 s): FFT synthesis block. Larger = fewer block-edge
  transients (each edge costs ~60 s of reward-filter settling; negligible vs the horizon).
- `--period` (default `horizon/2.5`): diurnal cycle. Use `86400` for a real 1-day cycle.
- Policy hyperparameters (forgetting rates `gamma`, TAR-UCB thresholds) are set in
  `make_policies()` for the **1-day diurnal cadence** (~4320 decisions/cycle at
  `hold=20 s`). They are scaled to the measured per-window reward noise `sigma≈0.10`.
  Re-tune there if `--hold`, `--period`, or the reward calibration change materially.

## Reward map (logistic — the default, and it matters)

The scalar reward is the smooth logistic of the raw multi-band score
(`sd.logistic`, r0=167.5, sigma0=2.5): non-saturating *within* the settled band
(rewards ~0.31–0.60, matches paper Table 2) but saturating *outside* it.

That saturation is deliberate. TAR-UCB identifies the regime from the **absolute reward
vector**, so it is level-sensitive. Transition/spike periods drive the raw score far below
the settled band; the logistic compresses those level swings, leaving the settled-regime
*shape* (which-arm-is-best) as the dominant classification signal. A controlled study on a
fixed cache (`bandit_tune_tarucb.py` / reward-map sweep) showed TAR-UCB regime ID degrades
sharply with a linear min–max reward instead:

    reward map        library J (true=3)   arm-accuracy
    logistic                  5                0.47      <- default
    min-max (steady floor)    5                0.32
    min-max (dynamic floor)   8                0.23

A linear min–max *spreads* the level swings across [0,1], so level dominates the fingerprint
and one physical regime fragments into several library entries. Additionally, a hard min–max
clip pins transient windows at exactly 0, which the change-detector mistakes for a settled
regime and turns into spurious "[0,0,0]" regimes. The logistic avoids both (smooth, no clip).

`--reward {logistic,norm,linear}` selects the map (default logistic). `norm` is the linear
min–max; if you use it, `bandit_calibrate.py` sets its floor below the DRIFTING-environment
minimum (a short fixed-controller run over transitions/spikes), not just the steady-cell min,
so transients don't clip — but logistic remains preferred for the TAR-UCB comparison.

## Calibrated reward landscape (logistic; matches paper Table 2)

Per-regime optimum is the designed diagonal (R0→C0, R1→C1, R2→C2). Logistic rewards:
R0: 0.60/0.52/0.49, R1: 0.32/0.34/0.31, R2: 0.49/0.57/0.57 (span ~0.31–0.60, non-saturating
within the band). Arm separations in R1/R2 are comparable to the per-window noise (~0.08) —
a genuinely hard non-stationary bandit whose achievable advantage over a fixed controller is
modest and accrues over long horizons and regime recurrences (TAR-UCB's target regime).

## LI-TAR-UCB (level-invariant variant, `paper/li_tar_ucb_draft.tex`)

`bandit_policies.LITARUCB` implements the draft: regimes stored/matched by **centered
shape** `phi(v)=v-mean(v)` (level-invariant recognition + persistent new-regime rule +
library merge). First tests exposed a **detection blind spot**: with only the level-adjusted
played-arm drop CUSUM, the constant-level diurnal R0<->R2 *shape* transitions are invisible
(the level tracker absorbs drift; the played arm says nothing about unplayed coordinates).
1-week result: played C0 80% of R2, regret 883 (worse than Fixed-C0's 777).

The draft was **revised** (Sec. "Scheduled balanced probing and shape-change detection"):
STABLE mode probes all arms every `L_probe` windows (`m_blk` samples/arm); the block's
centered shape both feeds the library and drives a level-invariant detector (`q_det`
consecutive blocks farther than `r_det` from the current shape -> alarm). Detectability of
direct switches follows from the existing shape-separation assumption — no new assumption;
regret gains a probing term `~ Delta_max*K*m_blk*T/L` and the delay term `q_det*(L+K*m_blk)`
(optimal `L* ~ sqrt(K*m_blk*T/(q_det*G_T))`, Remark "Choice of the probing period").

1-week head-to-head (logistic, `L_probe=60, m_blk=6, q_det=2`, **hold=20s**; sweep in
`bandit_tune_litarucb.py`): regret **883 -> 597**, C0-in-R2 0.80 -> 0.09, now ~tied with
TAR-UCB (562) and clearly ahead of Thompson (805) / Fixed-C0 (777) / D-UCB (1230), with a
consolidated library (J=2 vs TAR-UCB's J=14). Residual gap to TAR-UCB is probing overhead
plus settling on C1 instead of C2 inside R2, where the C1/C2 gap (0.005) is far below noise
(near-degenerate arms — negligible reward cost). Old no-probe checkpoint kept as
`policies/LI-TAR-UCB-noprobe.npz`.

### hold=100s + longer/rarer R1 spikes (2026-07-02)

Switched the default decision window to `--hold 100` (was 20s) and exposed the R1
micro-seism spike shape as CLI flags on `bandit_noise_cache.py`:
`--spike-rate` (Poisson rate per diurnal period, default 4.0/day) and
`--spike-attack`/`--spike-hold`/`--spike-decay` (mean ramp-up/plateau/ramp-down [s],
defaults 30/150/30). These were previously hardcoded in `RegimeEnv.__init__`
(`bandit_experiment.py:72`) and are now forwarded through `generate()`/`build_env_and_W()`
and folded into the manifest cache key, so changing them correctly triggers regeneration.
Test cache: `bandit_runs/cache_3d_r1long` (`--spike-rate 1 --spike-hold 1500`, 3 days) —
R1 episodes now last 561–1388s (mean ~975s dominant-regime duration), up from ~170s mean
at the old defaults, giving the TRANS→DIAG pipeline realistic room to actually classify R1
(previously it never could — see "Also diagnosed" note in
`memory/bandit-long-experiment.md`).

Recalibrating at hold=100 dropped `sigma_hat` 0.084 -> **0.041** (less than the naive
sqrt(5) from averaging 5x more sub-blocks per window, since the physics noise is
correlated, but still a real SNR gain).

**Re-swept `L_probe`/`m_blk`/`q_det`** for the new hold+noise (`bandit_tune_litarucb_v2.py`,
first on `cache_3d_r1long`, then a clean false-alarm check on a stationary synthetic
control). Counterintuitive finding: shrinking `L_probe` (probing *more often*) monotonically
**hurt** regret (53.7 -> 92.6 as L_probe: 60->6 windows) — the forced-exploration cost of
frequent probing dominates any detection-latency benefit at these timescales, since R1 is
collapse-mediated and already caught by the original drop-CUSUM regardless of probing
settings (probing only matters for the constant-level R0<->R2 direct switches, which persist
for hours — an L_probe of even 60-120 windows resolves them with room to spare). The
old-scaled config (`L=60, m_blk=6`, a window-count carryover from the hold=20s tuning)
remained the single best regret (53.7) and had the lowest clean false-alarm rate
(0.108/day) of everything tried.

Per the user's explicit ask to trade a bit of diagnosis speed for more false-positive risk,
the production default is now **`L_probe=90, m_blk=3, q_det=2`**: halving `m_blk` (6->3)
cuts the time to resolve a triggered probe roughly in half, at a real cost — clean
false-alarm rate rises ~5x (0.108 -> 0.540/day, i.e. about 1 spurious alarm every 2 days
instead of every 9) and 3-day regret rises modestly (53.7 -> 63.6). `q_det=2` is kept
(persistence against single-block noise). This is a deliberate, measured trade, not a free
win — see `bandit_tune_litarucb_v2.py` for the full sweep and the stationary-control
false-alarm test.

### 1-week run on the new noise, and the transition+diagnostic budget fix (2026-07-02)

Ran the full policy comparison on `bandit_runs/cache_1w_r1long` (spike_rate=1/day,
spike_hold=1500s, hold=100s, 13 R1 episodes over the week, durations 115-2666s). With the
`L_probe=90, m_blk=3` shape detector in place, **LI-TAR-UCB became the best bandit-family
method**: regret 138.6, ahead of TAR-UCB (154.8), Fixed-C0 (155.9), D-UCB (159.6), Thompson
(176.8). But R1 still had no dedicated library entry (J=2, not 3) — its per-regime arm
fractions stayed nearly flat (C0=.32/C1=.40/C2=.28).

Root cause: the shape detector (`L_probe`/`m_blk`) only controls how fast a regime *change*
is flagged. Recognising the *new* regime still requires completing TRANSITION-exit
(`m_batch` samples/arm x `q_stab` consecutive stable+non-collapsed batches) and DIAGNOSTIC
classification (`m_cls` samples/arm) -- total budget `(m_batch*q_stab + m_cls)*K*hold`
seconds. At the old defaults (`q_stab=2, m_cls=8`) that's **3000s**, longer than every
single observed R1 episode this week (max 2666s) — so R1 could never finish being
classified regardless of how fast it was detected.

Swept `(m_batch, q_stab, m_cls)` on `cache_1w_r1long` (`bandit_tune_diag.py`):

| config | budget | regret | frac_opt | J | R1 arm split |
|---|---|---|---|---|---|
| base (q_stab=2, m_cls=8) | 3000s | 138.6 | 0.331 | 2 | C0=.32 C1=.40 C2=.28 |
| **q_stab=1, m_cls=6** | **2100s** | **115.3** | **0.345** | **3** | C0=.32 C1=.43 C2=.25 |
| q_stab=1, m_cls=4 | 1500s | 116.8 | 0.315 | 4 (over-frag) | C0=.39 (wrong arm) |
| q_stab=1, m_cls=3 | 1200s | 137.6 | 0.328 | 4 (over-frag) | C0=.31 C1=.42 C2=.27 |
| q_stab=1, m_cls=2 | 900s | 159.3 | 0.236 | 3 | degraded across the board |

`q_stab=1, m_cls=6` is a genuine improvement, not just a faster/riskier point on a
monotone tradeoff curve: regret improves ~17% (138.6->115.3), the library correctly
settles at J=3 for the first time, and switches drop 2574->1810. Shrinking further
(`m_cls<=4`) over-fragments the library (J=4, a spurious extra entry) and actively flips
R1's preferred arm to the wrong controller — there's a real sweet spot, not a free lunch
in one direction. Set as the new production default in `make_policies()`.

Regime-schedule plots for the new noise: `bandit_plot_regime.py --cache <cache>
--periods 2 --zoom-spikes` — the `--zoom-spikes` flag (new) plots each individual R1
episode on a real-seconds scale (`r1_episode_NN_durXXXXs.png`), since on the multi-day
overview plots every spike looks like a thin line regardless of duration.
