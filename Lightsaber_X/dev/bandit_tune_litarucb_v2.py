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
LI-TAR-UCB probing-detector sweep, v2: hold=100s + lengthened/rarer R1
(spike_rate=1/day, spike_hold=1500s -- see bandit_noise_cache.py --spike-* flags).

Reports, per config: cum reward, regret vs Oracle, frac-optimal, switches, library size J,
number of probing blocks / shape alarms, AND a false-alarm rate measured only inside R0/R2
regime interiors (windows far from any true transition) -- the direct trade-off the user
asked to lean into (faster diagnosis for a bit more false-positive risk).

    python bandit_tune_litarucb_v2.py --cache bandit_runs/cache_3d_r1long --hold 100
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
    reg_dom = np.argmax(Wdec, axis=0)                  # dominant regime per decision (0/1/2)
    ck = os.path.join(args.cache, "experiment", "policies", "Oracle.npz")
    oracle_cum = float(np.load(ck)['rewards'][:ndec].sum()) if os.path.exists(ck) else np.nan

    # windows "deep inside" a stable regime: dominant regime unchanged for the last 20 windows
    # and unchanged for the next 5 -- used to measure the probing detector's false-alarm rate
    stable_mask = np.ones(ndec, bool)
    for k in range(1, 21):
        stable_mask[k:] &= (reg_dom[k:] == reg_dom[:-k])
    stable_mask[:20] = False; stable_mask[-5:] = False

    base = dict(sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                m_batch=1, r_stab=1.0*s, q_stab=2, d_exit=1.5*s, r_coll=r_sh,
                m_cls=8, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh, level_window=30, seed=0)
    configs = {
        "L60_m6_q2":   dict(L_probe=60,  m_blk=6, q_det=2),   # old, window-count carryover from hold=20
        "L60_m3_q2":   dict(L_probe=60,  m_blk=3, q_det=2),   # same cadence, cheaper block (lower sigma)
        "L90_m3_q2":   dict(L_probe=90,  m_blk=3, q_det=2),
        "L120_m3_q2":  dict(L_probe=120, m_blk=3, q_det=2),
        "L90_m2_q2":   dict(L_probe=90,  m_blk=2, q_det=2),
        "L20_m4_q2":   dict(L_probe=20,  m_blk=4, q_det=2),
        "L10_m2_q2":   dict(L_probe=10,  m_blk=2, q_det=2),
    }
    print(f"sigma={s:.4f} Delta_sh={dsh:.4f} r_sh=r_det={r_sh:.4f} oracle_cum={oracle_cum:.1f} ndec={ndec}")
    print(f"{'config':14s} {'cum':>8s} {'regret':>8s} {'frac_opt':>8s} {'switch':>7s} {'J':>3s} "
          f"{'probes':>7s} {'alarms':>7s} {'false_al':>9s} {'R1_frac_opt':>11s}")
    r1_mask = reg_dom == 1
    for name, kw in configs.items():
        pol = bp.LITARUCB(3, **{**base, **kw})
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, reward_mode=args.reward,
                              progress_every=0, name=name)
        r = h['rewards'][:ndec]; a = h['arms'][:ndec]
        frac = float(np.mean(a == opt_arm)); sw = int((np.diff(h['arms']) != 0).sum())
        J = len(pol.th_sum); alarms = [e for e in pol.probe_log if e[3]]
        # pol.probe_log stores self.t, which is already a DECISION-WINDOW count (policy.update()
        # increments it once per hold-window), not raw seconds -- do not divide by hold_n again.
        false_alarms = sum(1 for (t, D, c, al) in pol.probe_log
                            if al and int(t) < ndec and stable_mask[int(t)])
        r1_frac = float(np.mean(a[r1_mask] == opt_arm[r1_mask])) if r1_mask.sum() else float('nan')
        print(f"{name:14s} {r.sum():8.1f} {oracle_cum-r.sum():8.1f} {frac:8.3f} {sw:7d} {J:3d} "
              f"{len(pol.probe_log):7d} {len(alarms):7d} {false_alarms:9d} {r1_frac:11.3f}", flush=True)


if __name__ == "__main__":
    main()
