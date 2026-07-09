# --- repo path bootstrap (this script was moved into dev/; run it as
#     `python dev/<name>.py` from the Lightsaber_X/ directory). Makes the
#     Lightsaber_X package root, the bandit/ package, and this dev/ folder
#     importable, and anchors the working dir to the package root so the
#     relative data/output paths (noise_inputs/, *.npz, bandit_runs/) resolve. ---
import os as _os, sys as _sys
_DEV = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_DEV)
for _p in (_ROOT, _os.path.join(_ROOT, 'bandit'), _DEV):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)
# --- end bootstrap ---

"""
Fast TAR-UCB hyperparameter sweep on a pre-generated noise cache.

Reuses the exact streaming runner from bandit_long_experiment (so results match a
real run), but only runs TAR-UCB with different (r_cls, m_cls, h, nu, ...) settings.
Reports, per config: cumulative reward, regret vs the cached Oracle, fraction of
decisions on the oracle-optimal arm, number of switches, final library size J, and
the number of distinct regime labels actually used.

Run the baseline experiment first (to cache Oracle):
    python bandit_long_experiment.py --cache bandit_runs/cache_1w
    python bandit_tune_tarucb.py --cache bandit_runs/cache_1w
"""
import os, sys, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import bandit_rewards
import bandit_policies as bp
from bandit_noise_cache import load_cache
from bandit_calibrate import calibrate
from bandit_long_experiment import run_policy_stream

FS = 256


def make_accessors(cache):
    Wsec = np.asarray(cache['Wsec']); tsec = np.asarray(cache['tsec']); sens_sec = np.asarray(cache['sens_sec'])
    def W_of_sample(a0, a1):
        ts = np.arange(a0, a1) / FS
        return np.vstack([np.interp(ts, tsec, Wsec[i]) for i in range(3)])
    def sens_of_sample(a0, a1):
        ts = np.arange(a0, a1) / FS
        return np.interp(ts, tsec, sens_sec)
    return W_of_sample, sens_of_sample, Wsec, tsec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hold", type=int, default=100)
    args = ap.parse_args()
    cache = load_cache(args.cache); N = int(cache['N'])
    calib = calibrate(hold=args.hold); sigma = float(calib['sigma_hat'])
    lo, hi = float(calib['lo']), float(calib['hi'])
    oracle_table = np.asarray(calib['oracle_table'], float)
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    W_of_sample, sens_of_sample, Wsec, tsec = make_accessors(cache)

    # oracle-optimal arm per decision (for frac_opt / regret reference)
    hold_n = args.hold * FS; ndec = (N - 1) // hold_n
    dec_t = (np.arange(ndec) * hold_n) / FS
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])
    opt_arm = np.argmax(oracle_table @ Wdec, axis=0)
    ck = os.path.join(args.cache, "experiment", "policies", "Oracle.npz")
    oracle_cum = float(np.load(ck)['rewards'][:ndec].sum()) if os.path.exists(ck) else np.nan

    s = sigma
    # Levers against library bloat under ADIABATIC drift:
    #   q_stab/r_stab -- how settled the reward must be before we classify (avoid mid-drift diagnosis)
    #   h             -- detection threshold (higher = fewer, only-large-change detections)
    #   r_cls / m_cls -- classification tolerance / samples averaged before matching
    # Batch 1 showed: stricter change-detection -> more exploration -> worse. So push the
    # other way -- explore LESS: less UCB optimism (larger delta), fast transition exit
    # (q_stab=1, small m_cls), fewer false alarms (higher nu). Goal: beat default's regret 727.
    configs = {
        "default":         dict(m_cls=4, r_cls=2.0*s, m0=3, nu=0.6*s, h=7.0*s, r_stab=1.2*s, q_stab=2, delta=0.1),
        "fast_exit":       dict(m_cls=3, r_cls=2.0*s, m0=2, nu=0.7*s, h=7.0*s, r_stab=1.5*s, q_stab=1, delta=0.1),
        "less_ucb(d=0.4)": dict(m_cls=4, r_cls=2.0*s, m0=3, nu=0.6*s, h=7.0*s, r_stab=1.2*s, q_stab=2, delta=0.4),
        "lazy+fast":       dict(m_cls=3, r_cls=2.0*s, m0=2, nu=0.8*s, h=8.0*s, r_stab=1.5*s, q_stab=1, delta=0.3),
        "lean":            dict(m_cls=3, r_cls=2.2*s, m0=2, nu=0.7*s, h=7.0*s, r_stab=1.5*s, q_stab=1, delta=0.5),
    }
    print(f"sigma={s:.3f}  oracle_cum={oracle_cum:.1f}  ndec={ndec}  (true regimes=3)")
    print(f"{'config':30s} {'cum':>8s} {'regret':>8s} {'frac_opt':>8s} {'switch':>7s} {'J':>3s} {'labels':>7s}")
    for name, kw in configs.items():
        pol = bp.TARUCB(3, sigma=s, m_batch=1, add_diag=True, seed=0, **kw)
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, reward_mode='norm', progress_every=0, name=name)
        r = h['rewards'][:ndec]; a = h['arms'][:ndec]
        frac = float(np.mean(a == opt_arm)); sw = int((np.diff(h['arms']) != 0).sum())
        J = int(h['tar_J'][-1]); labels = len(set(int(z) for z in h['tar_Z']))
        print(f"{name:30s} {r.sum():8.1f} {oracle_cum-r.sum():8.1f} {frac:8.3f} {sw:7d} {J:3d} {labels:7d}", flush=True)


if __name__ == "__main__":
    main()
