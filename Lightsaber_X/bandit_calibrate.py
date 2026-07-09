"""
Calibrate the online-selection reward from the CURRENT validated fast engine.

Defines a linear min-max reward normalisation with margins:

    lo = min_raw - delta,   hi = max_raw + delta,   delta = frac * (max_raw - min_raw)
    r  = clip( (raw - lo) / (hi - lo), 0, 1 )

CRUCIAL: min_raw / max_raw are taken over BOTH the 9 steady (regime, controller) cells AND
a short run of a fixed controller over the *drifting* environment (regime transitions + R1
micro-seism spikes). The steady cells alone bottom out around raw~163, but the real dynamic
environment dips to raw~150 during transitions/spikes -- so a steady-only floor would clip
~10% of decisions to exactly 0. A clipped, FLAT-at-zero transient is then mistaken by a
change-detection policy (TAR-UCB) for a settled regime, producing a corrupt [0,0,0] regime
fingerprint and library fragmentation. Setting the floor below the true dynamic minimum
keeps a recovering transient a *varying ramp* (150->165), which the policy's unmodified
stabilisation test correctly refuses to treat as settled -- fixing fragmentation without
touching the algorithm. The linear map is scale-invariant for the bandits (sigma and the
oracle table rescale together), so lowering the floor is harmless apart from removing the clip.

Products (cached to bandit_runs/calibration.npz):
  * lo, hi            -- the reward-normalisation endpoints (used everywhere downstream)
  * oracle_table      -- normalised r_c(R_i), (arm, regime), for the Oracle policy
  * sigma_hat         -- median within-cell per-window reward std, sets TAR-UCB sigma

    python bandit_calibrate.py --dur 1200 --hold 20 --frac 0.10
"""
import os, sys, time, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import bandit_rewards
import fast_engine as fe
import switching_dynamics as sd
from reward_stream import StreamReward

FS = 256
OUT = "bandit_runs"
SENS = {0: 120.0, 1: 12.0, 2: 0.25}


def dynamic_env_raws(feat_names, w, hold_n, dur=14400, period=1800, seed=7, controller=1, warm_n=None):
    """Per-window raw scores for a FIXED controller over a short DRIFTING environment
    (regime transitions + R1 spikes). Captures the transition/spike low-reward transients
    that the steady 9-cell characterisation never sees. In-RAM (short)."""
    from bandit_noise_cache import RegimeNoiseGen
    from bandit_experiment import RegimeEnv
    fs = FS; N = int(dur * fs); block_n = 2048 * fs
    if warm_n is None:
        warm_n = 60 * fs
    env = RegimeEnv(dur, period=period, seed=seed)
    S = int(np.ceil(dur)) + 1; tsec = np.arange(S)
    Wsec = np.array([env.get_weights(float(t)) for t in tsec]).T
    sens_sec = np.array([SENS[0], SENS[1], SENS[2]]) @ Wsec
    gens = {R: RegimeNoiseGen(R, block_n, fs) for R in (0, 1, 2)}
    sus_itm = np.empty(N); sus_etm = np.empty(N); power = np.empty(N)
    tall = np.arange(block_n) / fs
    for b in range(int(np.ceil(N / block_n))):
        a0 = b * block_n; a1 = min(a0 + block_n, N); nn = a1 - a0
        tsamp = (a0 / fs) + tall[:nn]
        W = np.vstack([np.interp(tsamp, tsec, Wsec[i]) for i in range(3)])
        bi = np.zeros(nn); be = np.zeros(nn); bp_ = np.zeros(nn)
        for R in (0, 1, 2):
            itm, etm, pw = gens[R].block(seed + 1000 * b)
            bi += W[R] * itm[:nn]; be += W[R] * etm[:nn]; bp_ += W[R] * pw[:nn]
        sus_itm[a0:a1] = bi; sus_etm[a0:a1] = be; power[a0:a1] = bp_
    plant, sensors, ssc, data, phys = sd.setup(0, 8, seed=seed)
    K = fe.extract_plant_consts(plant, sensors, ssc, fs); csoft, chard, ce = fe.build_ctrl_stack(data, phys)
    st = fe.new_state()
    hp = np.zeros((K['hp_sos'].shape[0], 2)); rz0 = np.zeros((K['rad_sos'].shape[0], 2)); rz1 = rz0.copy()
    az0 = np.zeros((K['act_sos'].shape[0], 2)); az1 = az0.copy()
    szs = np.zeros((K['ss_sos'].shape[0], 2)); szh = szs.copy()
    czs = np.zeros((3, csoft.shape[1], 2)); czh = np.zeros((3, chard.shape[1], 2))
    rew = StreamReward(feat_names, w); srng = np.random.RandomState(12345); scl = (fs / 2.0) ** 0.5
    raws = []; nwin = (N - 1) // hold_n
    for j in range(nwin):
        a0 = j * hold_n
        sens_win = np.interp(np.arange(a0, a0 + hold_n) / fs, tsec, sens_sec)
        sn_s = srng.normal(0, scl * 1e-13, hold_n) * sens_win
        sn_h = srng.normal(0, scl * 3e-14, hold_n) * sens_win
        pitch, ctl, _ = fe.run_kernel(
            hold_n, 0, np.ascontiguousarray(sus_itm[a0:a0+hold_n]), np.ascontiguousarray(sus_etm[a0:a0+hold_n]),
            np.ascontiguousarray(power[a0:a0+hold_n]), sn_s, sn_h,
            K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
            K['hp_sos'], hp, K['rad_sos'], rz0, rz1, K['act_sos'], az0, az1,
            csoft, czs, chard, czh, ce, controller, controller, 0,
            K['loc2eig'], K['ss_sos'], szs, szh, K['ss_e2l'],
            K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], st)
        sc = rew.score(pitch, ctl); raws.append(sc.mean() if len(sc) else 167.5)
    raws = np.array(raws)
    return raws[warm_n // hold_n:]


def normalize_reward(raw, lo, hi):
    """Linear min-max map with margins -> [0,1] (see module docstring)."""
    return np.clip((np.asarray(raw) - lo) / (hi - lo), 0.0, 1.0)


def apply_reward(raw, reward_mode, lo, hi):
    """Scalarise a raw multi-band score. 'logistic' is the default (smooth saturation
    outside the settled band -> level swings from transitions/spikes are compressed, so a
    change-detection policy classifies regimes by arm-SHAPE not level; see the reward-map
    study). 'norm' is the linear min-max map; 'linear' a fixed affine fallback."""
    if reward_mode == 'logistic':
        return sd.logistic(np.asarray(raw))
    if reward_mode == 'norm':
        return normalize_reward(raw, lo, hi)
    return np.clip((np.asarray(raw) - 148.0) / 22.0, 0.0, 1.0)


def window_raws(pitch, ctl, feat_names, w, hold_n):
    """Per-window RAW scores using the continuous stream reward (state carried across windows)."""
    rew = StreamReward(feat_names, w)
    N = len(pitch); raws = []; ti = 0
    while ti + hold_n <= N:
        sc = rew.score(pitch[ti:ti+hold_n], ctl[ti:ti+hold_n])
        raws.append(sc.mean() if len(sc) else 167.5)
        ti += hold_n
    return np.array(raws)


def calibrate(dur=1200, hold=20, warmup=60, seed=1, reward_mode='logistic', frac=0.10,
              dyn=True, force=False):
    path = os.path.join(OUT, "calibration.npz")
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(path) and not force:
        m = np.load(path, allow_pickle=True)
        if int(m['hold']) == hold and int(m['dur']) == dur \
                and str(m.get('reward_mode', 'logistic')) == reward_mode \
                and (reward_mode != 'norm' or (float(m['frac']) == float(frac) and bool(m.get('dyn', True)) == dyn)):
            print(f"[calib] cached at {path} ({reward_mode}) -- skipping.")
            return {k: m[k] for k in m.files}
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    hold_n = int(hold * FS); warm_n = int(warmup * FS)
    # pass 1: steady per-window raw scores for all 9 cells (-> oracle table + sigma)
    raws = {}
    print(f"[calib] dur={dur}s hold={hold}s reward={reward_mode} -- 9 steady cells")
    for R in (0, 1, 2):
        for C in (0, 1, 2):
            t0 = time.time()
            pitch, ctl = fe.run_fast(R, C, dur, seed=seed, sens_scale=SENS[R])
            rw = window_raws(pitch[warm_n:], ctl[warm_n:], feat_names, w, hold_n)
            raws[(R, C)] = rw
            print(f"  R{R} C{C}: raw={rw.mean():7.2f} [{rw.min():7.2f},{rw.max():7.2f}]  "
                  f"(n={len(rw)}, {time.time()-t0:.1f}s)", flush=True)
    steady = np.concatenate([raws[k] for k in raws])
    # min-max endpoints (only needed for reward_mode='norm'). The linear map clips transients,
    # so its floor must sit below the DRIFTING-environment minimum, not just the steady min.
    lo, hi, min_raw, max_raw, delta = np.nan, np.nan, float(steady.min()), float(steady.max()), np.nan
    if reward_mode == 'norm':
        pool = steady
        if dyn:
            t0 = time.time()
            dyn_raw = dynamic_env_raws(feat_names, w, hold_n, seed=seed + 6)
            print(f"[calib] dynamic-env raw range [{dyn_raw.min():.2f},{dyn_raw.max():.2f}] "
                  f"(steady min was {steady.min():.2f}); n={len(dyn_raw)}, {time.time()-t0:.1f}s")
            pool = np.concatenate([steady, dyn_raw])
        min_raw, max_raw = float(pool.min()), float(pool.max())
        delta = frac * (max_raw - min_raw)
        lo, hi = min_raw - delta, max_raw + delta
        print(f"[calib] raw range [{min_raw:.2f},{max_raw:.2f}] delta={delta:.2f} -> lo={lo:.2f} hi={hi:.2f} "
              f"(clip frac {float(np.mean(pool <= lo)):.4f})")
    # pass 2: reward statistics under the chosen map (-> oracle table + sigma)
    raw_mean = np.zeros((3, 3)); rew_mean = np.zeros((3, 3)); rew_std = np.zeros((3, 3))
    for (R, C), rw in raws.items():
        r = apply_reward(rw, reward_mode, lo, hi)
        raw_mean[R, C] = rw.mean(); rew_mean[R, C] = r.mean(); rew_std[R, C] = r.std()
    oracle_table = rew_mean.T.copy()               # (arm, regime)
    diag = [int(np.argmax(rew_mean[R])) for R in (0, 1, 2)]
    sigma_hat = float(np.median(rew_std))
    print(f"[calib] reward span: min cell={rew_mean.min():.3f}  max cell={rew_mean.max():.3f}")
    print(f"[calib] per-regime argmax controller = {diag} (designed 0,1,2)")
    print(f"[calib] median within-cell per-window reward std = {sigma_hat:.3f} -> TAR-UCB sigma")
    np.savez(path, raw_mean=raw_mean, rew_mean=rew_mean, rew_std=rew_std,
             oracle_table=oracle_table, sigma_hat=sigma_hat, diag=np.array(diag),
             lo=lo, hi=hi, min_raw=min_raw, max_raw=max_raw, delta=delta, frac=frac,
             reward_mode=reward_mode, dyn=dyn, hold=hold, dur=dur)
    return dict(raw_mean=raw_mean, rew_mean=rew_mean, rew_std=rew_std, oracle_table=oracle_table,
                sigma_hat=sigma_hat, diag=np.array(diag), lo=lo, hi=hi,
                min_raw=min_raw, max_raw=max_raw, delta=delta, frac=frac, reward_mode=reward_mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=int, default=1200)
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--reward", default="logistic", choices=["logistic", "norm", "linear"])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    calibrate(dur=a.dur, hold=a.hold, reward_mode=a.reward, frac=a.frac, force=a.force)
