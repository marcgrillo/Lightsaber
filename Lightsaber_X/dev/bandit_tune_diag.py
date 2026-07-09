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
LI-TAR-UCB transition-exit / diagnostic-budget sweep.

The scheduled-probing shape detector (L_probe/m_blk/q_det, tuned in
bandit_tune_litarucb_v2.py) only controls how fast a regime CHANGE is flagged. Once
flagged, the algorithm still has to (1) exit TRANSITION mode (m_batch samples/arm per
batch, q_stab consecutive stable+non-collapsed batches) and (2) run DIAGNOSTIC classification
(m_cls samples/arm). Total real-time budget before a regime can be recognised/reused is

    (m_batch*q_stab + m_cls) * K * hold   seconds

At the current defaults (m_batch=1, q_stab=2, m_cls=8, K=3, hold=100s) that's 3000s -- longer
than even the longest R1 episode observed on cache_1w_r1long this week (2666s), which is why
R1 never gets its own library entry (J stays at 2, not 3) despite spike_hold=1500 already
being a 10x lengthening over the original noise params. This sweep shrinks that budget and
checks: does R1 get recognised (J=3, clear per-regime arm fraction), and at what cost to
overall regret / R0-R2 library fragmentation (a real risk: fewer diagnostic samples -> noisier
classification -> more spurious library entries there too, not just faster R1 recognition).

    python bandit_tune_diag.py --cache bandit_runs/cache_1w_r1long --hold 100
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
    reg_dom = np.argmax(Wdec, axis=0)
    ck = os.path.join(args.cache, "experiment", "policies", "Oracle.npz")
    oracle_cum = float(np.load(ck)['rewards'][:ndec].sum()) if os.path.exists(ck) else np.nan

    K = 3
    base = dict(sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                d_exit=1.5*s, r_coll=r_sh, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
                level_window=30, L_probe=90, m_blk=3, q_det=2, seed=0)
    # (m_batch, q_stab, m_cls) -> total transition+diagnostic budget in seconds
    configs = {
        "base(m_batch1,q2,m_cls8)":  dict(m_batch=1, r_stab=1.0*s, q_stab=2, m_cls=8),
        "q1,m_cls6":                 dict(m_batch=1, r_stab=1.0*s, q_stab=1, m_cls=6),
        "q1,m_cls4":                 dict(m_batch=1, r_stab=1.0*s, q_stab=1, m_cls=4),
        "q1,m_cls3":                 dict(m_batch=1, r_stab=1.0*s, q_stab=1, m_cls=3),
        "q1,m_cls2":                 dict(m_batch=1, r_stab=1.0*s, q_stab=1, m_cls=2),
    }
    print(f"sigma={s:.4f} Delta_sh={dsh:.4f} r_sh={r_sh:.4f} oracle_cum={oracle_cum:.1f} ndec={ndec}")
    print(f"{'config':26s} {'budget_s':>8s} {'cum':>8s} {'regret':>8s} {'frac_opt':>8s} "
          f"{'switch':>7s} {'J':>3s} {'R1_n':>5s} {'R1_C0':>6s} {'R1_C1':>6s} {'R1_C2':>6s}")
    r1_mask = reg_dom == 1
    for name, kw in configs.items():
        budget_s = (kw['m_batch'] * kw['q_stab'] + kw['m_cls']) * K * args.hold
        pol = bp.LITARUCB(3, **{**base, **kw})
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, reward_mode=args.reward,
                              progress_every=0, name=name)
        r = h['rewards'][:ndec]; a = h['arms'][:ndec]
        frac = float(np.mean(a == opt_arm)); sw = int((np.diff(h['arms']) != 0).sum())
        J = len(pol.th_sum)
        r1_a = a[r1_mask]; n1 = len(r1_a)
        fr = [np.mean(r1_a == k) if n1 else float('nan') for k in range(3)]
        print(f"{name:26s} {budget_s:8d} {r.sum():8.1f} {oracle_cum-r.sum():8.1f} {frac:8.3f} "
              f"{sw:7d} {J:3d} {n1:5d} {fr[0]:6.2f} {fr[1]:6.2f} {fr[2]:6.2f}", flush=True)


if __name__ == "__main__":
    main()
