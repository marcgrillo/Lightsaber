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
LI-TAR-UCB probing-detector sweep on a pre-generated noise cache.

Tests the revised algorithm (paper/li_tar_ucb_draft.tex, Sec. "Scheduled balanced
probing and shape-change detection"): STABLE mode interleaves balanced probing blocks
(every L_probe decision windows, m_blk samples/arm) whose centered shape feeds both the
library and a level-invariant shape-change detector (q_det consecutive deviations >
r_det -> alarm). This removes the drop-detector blind spot for the constant-level
R0<->R2 diurnal shape transitions.

Run after the baseline experiment (Oracle must be cached):
    python bandit_tune_litarucb.py --cache bandit_runs/cache_1w --reward logistic
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
                m_batch=1, r_stab=1.0*s, q_stab=2, d_exit=1.5*s, r_coll=r_sh,
                m_cls=8, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh, level_window=30, seed=0)
    # L_probe / m_blk in decision windows (hold=20 s -> L=60 is a probe every 20 min,
    # block cost K*m_blk windows). q_det consecutive deviating blocks -> shape alarm.
    configs = {
        "no-probe(old)": dict(L_probe=10**9),
        "L60_m6_q2":     dict(L_probe=60,  m_blk=6, q_det=2),
        "L100_m8_q2":    dict(L_probe=100, m_blk=8, q_det=2),
        "L60_m6_q3":     dict(L_probe=60,  m_blk=6, q_det=3),
        "L150_m8_q2":    dict(L_probe=150, m_blk=8, q_det=2),
    }
    print(f"sigma={s:.3f} Delta_sh={dsh:.3f} r_sh=r_det={r_sh:.3f} oracle_cum={oracle_cum:.1f} ndec={ndec}")
    print(f"{'config':16s} {'cum':>8s} {'regret':>8s} {'frac_opt':>8s} {'switch':>7s} {'J':>3s} "
          f"{'probes':>7s} {'alarms':>7s}")
    for name, kw in configs.items():
        pol = bp.LITARUCB(3, **{**base, **kw})
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, reward_mode=args.reward,
                              progress_every=0, name=name)
        r = h['rewards'][:ndec]; a = h['arms'][:ndec]
        frac = float(np.mean(a == opt_arm)); sw = int((np.diff(h['arms']) != 0).sum())
        J = len(pol.th_sum); alarms = sum(1 for e in pol.probe_log if e[3])
        print(f"{name:16s} {r.sum():8.1f} {oracle_cum-r.sum():8.1f} {frac:8.3f} {sw:7d} {J:3d} "
              f"{len(pol.probe_log):7d} {alarms:7d}", flush=True)


if __name__ == "__main__":
    main()
