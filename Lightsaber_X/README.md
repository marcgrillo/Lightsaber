# Lightsaber_X — online controller selection under non-stationary noise

A fork of **Lightsaber** (the Virgo/A+ angular-control simulator) extended to study
**online switching between fixed linear controllers** with a **bandit** policy, in a
**non-stationary** noise environment.

The physics simulator is unchanged in spirit from the trusted reference (`old_ref`); on
top of it we add: (1) a regime-switching environment, (2) a fast Numba closed-loop engine
that runs three controller banks in parallel with bumpless handover, (3) a calibrated
multi-band reward, and (4) a family of bandit policies that pick the controller online.
The headline method is **RA-TS-F** (Recurrence-Aware Thompson Sampling with a bounded
coverage probe); it is compared against fixed controllers, an operator-like rule-based
baseline, and standard bandits (D-UCB, discounted Thompson), plus earlier in-house methods
(TAR-UCB, LI-TAR-UCB, CG-ICLB family).

> **Run everything from this `Lightsaber_X/` directory** using the project Python
> (`C:/Users/marco/anaconda3/python.exe` on the dev machine — plain `python` on PATH is
> broken there). Scripts resolve data and output paths relative to this directory.

---

## Quick start — reproduce the benchmark

```bash
# 0) one-time: sanity-check the fast engine against the reference physics
python fast_engine.py                 # prints settled scores for R0/C0, R1/C1, R2/C2

# 1) calibrate the reward map (oracle table, reward noise sigma, normalisation endpoints)
python bandit_calibrate.py --hold 100 --reward logistic

# 2) generate a disk-backed noise cache (environment shared by ALL policies).
#    Small demo (1 h) first; the paper uses a 6-month horizon (~48 GB of memmaps).
python bandit_noise_cache.py --horizon 3600     --out bandit_runs/cache_demo
python bandit_noise_cache.py --horizon 15552000 --out bandit_runs/cache_6mo --block 4096

# 3) run the full policy comparison over the cache (checkpoints per policy)
python bandit_long_experiment.py --cache bandit_runs/cache_6mo --hold 100 --reward logistic
```

Outputs (summary CSV/JSON + figures) land in `<cache>/experiment/`. Each policy is
checkpointed to `<cache>/experiment/policies/<name>.npz`, so a re-run skips finished
policies and an interruption only loses the in-progress one. Use
`--only RA-TS-F` to (re)run a single policy, `--force` to recompute.

For a shorter self-contained run, `week_simulation.py --days 7 --hold 300` generates its
own week-scale environment and runs the same comparison.

---

## Repository layout

```
Lightsaber_X/
├── README.md                     ← you are here
├── VALIDATION_vs_old_ref.md      ← proof the Plant-class base matches the trusted old_ref
│
│  ── core physics + engine ───────────────────────────────────────────────
├── Lightsaber.py                 Plant / Sensors / SS_compensation / Controller classes
├── simulate.py                   base single-run driver (original Lightsaber_X entry)
├── fast_engine.py                Numba JIT closed-loop kernel (3 parallel controller banks,
│                                 bumpless 2 s ramp); `validate()` vs the slow engine
│
│  ── reward ───────────────────────────────────────────────────────────────
├── bandit_rewards.py             multi-band log-RMS features + weights loading
├── reward_stream.py              continuous (stateful) reward filtering across windows
├── bandit_reward_weights_rel2base.npz   calibrated band weights (used everywhere)
│
│  ── environment + policies ────────────────────────────────────────────────
├── bandit_simulation.py          regime definitions, scales, closed-loop simulation
├── bandit_experiment.py          RegimeEnv (non-stationary schedule + R1 spikes)
├── bandit_policies.py            ALL bandit policies (see list below)
├── switching_dynamics.py         hot/cold controller-switch transient study (paper Sec. 6)
│
│  ── experiment drivers ────────────────────────────────────────────────────
├── bandit_calibrate.py           reward calibration → oracle table, sigma, lo/hi
├── bandit_noise_cache.py         generate the disk-backed shared noise cache
├── bandit_long_experiment.py     MAIN months-scale policy comparison (streams the cache)
├── week_simulation.py            week-scale self-contained comparison
│
├── noise_inputs/                 measured ASDs, OSEM/readout noise, power PSDs
├── transfer_functions/           suspension transfer functions
│
├── paper/                        LaTeX sources (see below)
│   ├── main.tex                  ← the paper being written (canonical; builds with pdflatex)
│   ├── ra_ts_*.tex               RA-TS algorithm / theory / skeleton write-ups (the paper's method)
│   ├── figs/                     figures main.tex includes (tracked, committed as PNGs)
│   └── archive/                  superseded drafts + earlier algorithms no longer pursued
│                                 (lightsaber_draft, tar_ucb_draft, li_tar_ucb_draft, cg_iclb_*)
│
└── dev/                          exploratory / tuning / audit scripts + notes (see below)
```

### The bandit policies (`bandit_policies.py`)

| Policy         | What it is |
|----------------|------------|
| **RA-TS-F**    | Recurrence-Aware Thompson Sampling + drift detector + warm-start library + bounded coverage probe. **Best performer.** |
| RA-TS / -FQ / -FR | RA-TS ablations: no probe (stalls), delayed-commit queue, rollback-on-alarm. |
| CG-ICLB / -TS / -TT / -R / -R2 | Confidence-Gated Identify-and-Commit Library Bandit family. |
| TAR-UCB, LI-TAR-UCB | Transition-Aware Recurrent UCB and its level-invariant successor. |
| D-UCB, Thompson | Standard discounted-UCB / discounted-Thompson baselines. |
| Rule-based     | Operator-like baseline: reward-only EWMA alarm + periodic re-check + round-robin probe. **No access to the true regime** (only Oracle sees it). |
| Fixed-C0/C1/C2 | Each fixed controller. |
| Oracle         | Plays `argmax` from the calibrated reward table (upper bound). |

---

## `dev/` — exploratory scripts and notes

Hyperparameter tuning (`bandit_tune_*`), sweeps (`bandit_sweep_lprobe`), diagnostics/audits
(`bandit_audit_litarucb`, `bandit_log_litarucb`, `bandit_plot_regime`), figure generation
(`bandit_paper_figs`, `rats_demo_figs/`, `week_sim_figs/`), reward characterization
(`characterize_*`), alternative switch studies (`switching_dynamics_fast/realistic`), and
the working notes (`TARUCB_ISSUES.txt`, `LITARUCB_TRANSITION_AUDIT.txt`,
`BANDIT_LONG_README.md`). These are the experimental record — not needed to run the core
benchmark, but they document how the design decisions were reached.

Run them from `Lightsaber_X/` as `python dev/<name>.py`; each has a small path-bootstrap
header that makes the core modules importable and anchors the working directory here.

---

## Large / regenerable outputs (not tracked)

`.gitignore` excludes everything regenerable, in particular:

- `bandit_runs/` — noise caches and experiment checkpoints (up to ~48 GB for 6 months);
- `switching_dynamics/`, `paper_characterization/` — scratch figure-output directories
  (generated by `dev/switching_dynamics_fast.py` and `dev/characterize_for_paper_fast.py`);
  the specific PNGs `paper/main.tex` includes are copied into the tracked `paper/figs/`, so
  the paper builds without regenerating anything;
- `__pycache__/`, Numba caches, LaTeX build artifacts.

To regenerate a paper figure from scratch (e.g. after a reward/engine change), rerun the
relevant `dev/` script and re-copy the updated PNG into `paper/figs/`.

---

## Provenance

`Lightsaber_X` is a fork of Lightsaber (author: T. Andric). The Plant-class base is
validated against the original component-graph reference in `old_ref` — see
[`VALIDATION_vs_old_ref.md`](VALIDATION_vs_old_ref.md). Only the switching environment,
fast engine, reward calibration, and bandit policies are new here.
