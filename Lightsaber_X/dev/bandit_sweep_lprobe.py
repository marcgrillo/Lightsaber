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
Sweep L_probe on 1-week cache to find the optimal probing frequency.

Fixes q_stab=1, m_cls=6 (the sweet spot from bandit_tune_diag.py) and sweeps L_probe
(probing frequency in decision windows). Smaller L_probe = more forced exploration but
faster shape-change detection. Larger L_probe = less overhead but slower detection.

The direct R0<->R2 switches persist for hours, so we may not need to probe as frequently
as every 2.5 hours (L=90). Try L in [60, 90, 120, 180, 240, 300] and pick the best.

    python bandit_sweep_lprobe.py --cache bandit_runs/cache_1w_r1long --hold 100
"""
import os, sys, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import bandit_rewards
import bandit_policies as bp
from bandit_noise_cache import load_cache
from bandit_calibrate import calibrate
from bandit_long_experiment import run_policy_stream, shape_separation
from bandit_tune_tarucb import make_accessors

FS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--reward", default="logistic", choices=["norm", "logistic"])
    args = ap.parse_args()
    cache = load_cache(args.cache); N = int(cache['N'])
    calib = calibrate(hold=args.hold, reward_mode=args.reward)
    s = float(calib['sigma_hat']); lo, hi = float(calib['lo']), float(calib['hi'])
    oracle_table = np.asarray(calib['oracle_table'], float)
    dsh = shape_separation(oracle_table)
    r_sh = float(np.clip(0.45 * dsh, 0.6 * s, 1.2 * s))
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    W_of_sample, sens_of_sample, Wsec, tsec = make_accessors(cache)

    hold_n = args.hold * FS; ndec = (N - 1) // hold_n
    dec_t = (np.arange(ndec) * hold_n) / FS
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])
    opt_arm = np.argmax(oracle_table @ Wdec, axis=0)
    ck = os.path.join(args.cache, "experiment", "policies", "Oracle.npz")
    oracle_cum = float(np.load(ck)['rewards'][:ndec].sum()) if os.path.exists(ck) else np.nan

    base = dict(sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                m_batch=1, r_stab=1.0*s, q_stab=1, d_exit=1.5*s, r_coll=r_sh,
                m_cls=6, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
                level_window=30, m_blk=3, q_det=2, seed=0)

    L_probes = [60, 90, 120, 180, 240, 300]

    print(f"sigma={s:.4f} Delta_sh={dsh:.4f} r_sh={r_sh:.4f} oracle_cum={oracle_cum:.1f} ndec={ndec}")
    print(f"{'L_probe':>7s} {'probe_interval_h':>15s} {'cum':>8s} {'regret':>8s} {'frac_opt':>8s} "
          f"{'switch':>7s} {'J':>3s} {'probes':>7s}")
    for L in L_probes:
        probe_interval_h = L * args.hold / 3600.0
        pol = bp.LITARUCB(3, L_probe=L, **base)
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, reward_mode=args.reward,
                              progress_every=0, name=f"L{L}")
        r = h['rewards'][:ndec]; a = h['arms'][:ndec]
        frac = float(np.mean(a == opt_arm)); sw = int((np.diff(h['arms']) != 0).sum())
        J = len(pol.th_sum); n_probes = len(pol.probe_log)
        print(f"{L:7d} {probe_interval_h:15.2f} {r.sum():8.1f} {oracle_cum-r.sum():8.1f} {frac:8.3f} "
              f"{sw:7d} {J:3d} {n_probes:7d}", flush=True)


if __name__ == "__main__":
    main()
