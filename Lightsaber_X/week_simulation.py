"""
One-week reward-in-time simulation on the adiabatic non-stationary environment.

A genuine closed-loop run over a full 7 days (604800 s, 2016 decision windows at
T_hold=300 s) on the validated fast (Numba) engine.  A continuous week at 256 Hz
would need ~13 GB of pre-built IFFT noise, so the exogenous noise is generated in
1-hour CHUNKS: within a chunk the three regime noises share a seed (coherent blend),
the closed-loop filter/plant/controller states persist across chunks (only the
injected noise has a negligible once-an-hour seam).  All policies advance in lockstep
on the IDENTICAL per-window noise (common random numbers); each keeps its own plant +
controller-bank + reward-filter state.

Schedule: RegimeEnv with a 1-day diurnal R0<->R2 drift (hard-thresholded) plus Poisson
micro-seismic R1 spikes, over 7 cycles.  Each T_hold the segment-mean reward (logistic
of the multi-band log-RMS score, continuous StreamReward filtering) updates each policy.

Outputs (./bandit_runs/week/): histories.npz, summary.csv, and figures.
"""
import os, sys, time, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import matplotlib.pyplot as plt
import bandit_rewards
import switching_dynamics as sd
import bandit_policies as bp
import fast_engine as fe
from reward_stream import StreamReward
from bandit_experiment import RegimeEnv, RAW_TABLE, ORACLE_TABLE, SENS

FS = 256
OUT = os.path.join("bandit_runs", "week")
DAY = 86400


class PolicyState:
    """Persistent closed-loop state for one policy (its own plant/SS/controller-bank
    delay lines, reward-filter state, and bandit), advanced window by window."""
    def __init__(self, name, pol, K, Ns, Nh, feat_names, w):
        self.name, self.pol = name, pol
        self.st = fe.new_state()
        self.hp_zi = np.zeros((K['hp_sos'].shape[0], 2))
        self.rz0 = np.zeros((K['rad_sos'].shape[0], 2)); self.rz1 = np.zeros((K['rad_sos'].shape[0], 2))
        self.az0 = np.zeros((K['act_sos'].shape[0], 2)); self.az1 = np.zeros((K['act_sos'].shape[0], 2))
        self.szs = np.zeros((K['ss_sos'].shape[0], 2)); self.szh = np.zeros((K['ss_sos'].shape[0], 2))
        self.csoft_zi = np.zeros((3, Ns, 2)); self.chard_zi = np.zeros((3, Nh, 2))
        self.rew = StreamReward(feat_names, w)
        self.active = None; self.prev = None; self.ramp_this = 0
        self.rewards, self.arms, self.times, self.ctx = [], [], [], []

    def window(self, K, csoft, chard, ce, ramp_n, sus_w, suse_w, pow_w, sns_w, snh_w,
               t_end, ctx_end, reward_mode):
        """Run one decision window; update the bandit; record reward/arm. ctx_end = W at
        window end (the context revealed to context-aware policies for the next choice)."""
        if self.active is None:                       # first call: initial choice
            self.active = self.pol.choose(0, ctx_end); self.prev = self.active; self.ramp_this = 0
        nb = len(sus_w)
        pitch, ctl, _ = fe.run_kernel(
            nb, 0, sus_w, suse_w, pow_w, sns_w, snh_w,
            K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
            K['hp_sos'], self.hp_zi, K['rad_sos'], self.rz0, self.rz1, K['act_sos'], self.az0, self.az1,
            csoft, self.csoft_zi, chard, self.chard_zi, ce, self.active, self.prev, self.ramp_this,
            K['loc2eig'], K['ss_sos'], self.szs, self.szh, K['ss_e2l'],
            K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], self.st)
        sc = self.rew.score(pitch, ctl)
        mean_raw = sc.mean() if len(sc) else 167.5
        reward = (float(np.clip((mean_raw - 148.0)/22.0, 0, 1)) if reward_mode == 'linear'
                  else sd.logistic(mean_raw))
        self.pol.update(self.active, reward)
        self.rewards.append(reward); self.arms.append(self.active); self.times.append(t_end)
        self.ctx.append(ctx_end.copy())
        new = self.pol.choose(len(self.rewards), ctx_end)
        if new != self.active:
            self.prev = self.active; self.active = new; self.ramp_this = ramp_n
        else:
            self.prev = self.active; self.ramp_this = 0


def regime_noises_chunk(seed, dur_s):
    """Three coherent (same-seed) regime noises for a chunk: R -> (tst(2,N) contiguous, power(N,))."""
    out = {}
    for R in (0, 1, 2):
        plant, _, _, _, _ = sd.setup(R, dur_s, seed)
        out[R] = (np.ascontiguousarray(plant.tst_noise_t[0]),
                  np.ascontiguousarray(plant.tst_noise_t[1]),
                  np.ascontiguousarray(plant.input_power))
    return out


def make_policies():
    return {
        "Oracle":        bp.Oracle(ORACLE_TABLE),
        "Contextual-TS": bp.ContextualTS(3, sigma=0.05, ctx_noise=0.2, seed=7),
        "TAR-UCB":       bp.TARUCB(3, sigma=0.05, delta=0.1, m0=3, nu=0.05, h=0.35,
                                   m_batch=2, r_stab=0.06, q_stab=3, m_cls=3, r_cls=0.08, seed=0),
        "Fixed-C0":      bp.FixedArm(0),
        "Fixed-C1":      bp.FixedArm(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0, help="horizon [days]")
    ap.add_argument("--period", type=float, default=DAY, help="diurnal period [s] (default 1 day)")
    ap.add_argument("--hold", type=int, default=300, help="decision interval T_hold [s]")
    ap.add_argument("--chunk", type=int, default=3600, help="noise-generation chunk [s]")
    ap.add_argument("--reward", choices=["logistic", "linear"], default="linear")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")

    week_s = int(round(args.days*DAY))
    hold_n = args.hold*FS
    chunk_s = args.chunk
    assert chunk_s % args.hold == 0, "chunk must be a whole number of hold windows"
    n_chunks = int(np.ceil(week_s/chunk_s))
    ramp_n = int(2.0*FS)
    scl = (FS/2.0)**0.5

    # --- environment schedule: per-second regime weights for the whole week ---
    env = RegimeEnv(week_s, period=args.period, seed=args.seed)
    print(f"RegimeEnv: {args.days:g} d, period={env.period:.0f}s, "
          f"{len(env.drift)} drift transitions, {len(env.intervals)} R1 spikes", flush=True)
    secs = np.arange(0, week_s + 1)
    t0 = time.time()
    Wsec = np.array([env.get_weights(float(t)) for t in secs]).T   # (3, week_s+1)
    print(f"  weight schedule built ({time.time()-t0:.0f}s)", flush=True)

    # --- shared plant constants + controller banks (regime-independent) ---
    plant, sensors, ssc, data, phys = sd.setup(0, args.chunk, args.seed)
    K = fe.extract_plant_consts(plant, sensors, ssc, FS)
    csoft, chard, ce = fe.build_ctrl_stack(data, phys)
    Ns, Nh = csoft.shape[1], chard.shape[1]

    pols = make_policies()
    states = {name: PolicyState(name, pol, K, Ns, Nh, feat_names, w) for name, pol in pols.items()}
    print(f"Policies: {', '.join(states)}", flush=True)

    t_start = time.time()
    for c in range(n_chunks):
        c0 = c*chunk_s; c1 = min(c0 + chunk_s, week_s); cn = c1 - c0
        N = cn*FS
        # coherent regime noise for this chunk (fresh seed per chunk -> independent segment)
        noises = regime_noises_chunk(args.seed + c, cn)
        # per-sample regime weights over the chunk (interp the per-second schedule)
        ts = np.arange(c0, c1 + 1)
        tsamp = c0 + np.arange(N)/FS
        Wc = np.vstack([np.interp(tsamp, ts, Wsec[i, c0:c1 + 1]) for i in range(3)])  # (3,N)
        # blended exogenous noise (coherent), power, and sensing scale
        sus = np.ascontiguousarray(Wc[0]*noises[0][0] + Wc[1]*noises[1][0] + Wc[2]*noises[2][0])
        suse = np.ascontiguousarray(Wc[0]*noises[0][1] + Wc[1]*noises[1][1] + Wc[2]*noises[2][1])
        powr = np.ascontiguousarray(Wc[0]*noises[0][2] + Wc[1]*noises[1][2] + Wc[2]*noises[2][2])
        sens_t = SENS[0]*Wc[0] + SENS[1]*Wc[1] + SENS[2]*Wc[2]
        rng = np.random.RandomState(args.seed + 10000 + c)   # sensor noise, shared across policies
        sns = np.ascontiguousarray(rng.normal(0, scl*1e-13, N)*sens_t)
        snh = np.ascontiguousarray(rng.normal(0, scl*3e-14, N)*sens_t)

        # decision windows within the chunk (lockstep over policies = common random numbers)
        w0 = 0
        while w0 < N - 1:
            nb = min(hold_n, N - 1 - w0)
            w1 = w0 + nb
            t_end = (c0*FS + w1)/FS                       # absolute time at window end [s]
            ie = min(int(round(t_end)), week_s)
            ctx_end = Wsec[:, ie].copy()
            sus_w = sus[w0:w1]; suse_w = suse[w0:w1]; pow_w = powr[w0:w1]
            sns_w = sns[w0:w1]; snh_w = snh[w0:w1]
            for stt in states.values():
                stt.window(K, csoft, chard, ce, ramp_n, sus_w, suse_w, pow_w, sns_w, snh_w,
                           t_end, ctx_end, args.reward)
            w0 = w1
        done = len(next(iter(states.values())).rewards)
        el = time.time() - t_start
        print(f"  chunk {c+1}/{n_chunks}  day {c1/DAY:5.2f}  windows={done}  ({el:.0f}s)", flush=True)

    # ---- aggregate + save ----
    times = np.array(states['Oracle'].times)/DAY          # decision times [days]
    ctxW = np.array(states['Oracle'].ctx).T               # (3, n_dec)
    sev = 1.0*ctxW[1] + 2.0*ctxW[2]
    opt_arm = np.argmax(ORACLE_TABLE @ ctxW, axis=0)

    np.savez_compressed(os.path.join(OUT, "histories.npz"),
                        times=times, ctxW=ctxW, sev=sev, opt_arm=opt_arm,
                        **{f"{n}_rewards": np.array(s.rewards) for n, s in states.items()},
                        **{f"{n}_arms": np.array(s.arms) for n, s in states.items()})

    oracle_cum = np.sum(states['Oracle'].rewards)
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("policy,cum_reward,mean_reward,regret_vs_oracle,switches,frac_optimal\n")
        for n, s in states.items():
            arms = np.array(s.arms); rw = np.array(s.rewards)
            m = min(len(arms), len(opt_arm))
            frac = float(np.mean(arms[:m] == opt_arm[:m]))
            sw = int((np.diff(arms) != 0).sum())
            f.write(f"{n},{rw.sum():.3f},{rw.mean():.4f},{oracle_cum-rw.sum():.3f},{sw},{frac:.3f}\n")
            print(f"{n:14s} cum={rw.sum():8.2f}  mean={rw.mean():.3f}  switches={sw:4d}  frac_opt={frac:.3f}", flush=True)

    fixed_names = [n for n in states if n.startswith("Fixed")]
    best_fixed = max(fixed_names, key=lambda n: np.sum(states[n].rewards))

    # TAR-UCB regime-discovery diagnostics
    if "TAR-UCB" in states:
        tar = states["TAR-UCB"].pol
        print(f"\nTAR-UCB: discovered J={len(tar.N)} regimes; library mean vectors:")
        for j in range(len(tar.N)):
            print(f"    regime {j}: mu_hat=[{', '.join(f'{x:.3f}' for x in tar._mu(j))}]  "
                  f"N={tar.N[j].astype(int)}", flush=True)
        np.savez_compressed(os.path.join(OUT, "tarucb_diag.npz"),
                            Z_hist=np.array(tar.Z_hist), J_hist=np.array(tar.J_hist),
                            mode_hist=np.array(tar.mode_hist),
                            lib_mu=np.array([tar._mu(j) for j in range(len(tar.N))]))

    def roll(x, k=12):
        if len(x) < k: return x
        return np.convolve(x, np.ones(k)/k, mode='same')

    # ---- fig: reward in time (the requested view) ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 1]})
    axes[0].fill_between(times, 0, ctxW[0], color='tab:green', alpha=0.4, label='R0 (calm)')
    axes[0].fill_between(times, 0, ctxW[2], color='tab:red', alpha=0.35, label='R2 (severe)')
    axes[0].plot(times, ctxW[1], color='tab:orange', lw=0.8, label='R1 (spikes)')
    for d in range(1, int(round(times[-1]))):
        axes[0].axvline(d, color='k', ls=':', alpha=0.25)
    axes[0].set_ylabel("regime weight"); axes[0].legend(fontsize=8, ncol=3, loc='upper right')
    axes[0].set_title(f"One-week non-stationary run (period={env.period/DAY:.2f} d, "
                      f"T_hold={args.hold}s, {len(times)} windows)")
    cols = {"Oracle": 'k', "Contextual-TS": 'tab:blue', "TAR-UCB": 'tab:red', best_fixed: 'tab:gray'}
    panel = ["Oracle", "Contextual-TS", "TAR-UCB", best_fixed]
    for n in dict.fromkeys(panel):                      # de-dup if best_fixed collides
        rw = np.array(states[n].rewards)
        axes[1].plot(times, rw, color=cols[n], alpha=0.15, lw=0.5)
        axes[1].plot(times, roll(rw), color=cols[n], lw=1.8,
                     label=f"{n} (mean {rw.mean():.3f})")
    axes[1].set_ylabel("window reward"); axes[1].legend(fontsize=9, loc='lower left'); axes[1].grid(alpha=0.3)
    axes[2].step(times, opt_arm, where='post', color='k', lw=1.0, label='optimal')
    axes[2].step(times, np.array(states['Contextual-TS'].arms), where='post',
                 color='tab:blue', lw=1.0, alpha=0.8, label='Contextual-TS')
    axes[2].set_yticks([0, 1, 2]); axes[2].set_yticklabels(["C0", "C1", "C2"])
    axes[2].set_ylabel("controller"); axes[2].set_xlabel("time [days]")
    axes[2].legend(fontsize=8, loc='upper right'); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_week_reward.png"), dpi=130); plt.close()

    # ---- fig: cumulative reward / regret ----
    plt.figure(figsize=(11, 6))
    for n, s in states.items():
        rw = np.array(s.rewards)
        plt.plot(times, np.cumsum(rw), label=f"{n} ({rw.sum():.1f})",
                 lw=(2.2 if n == "Oracle" else 1.4), ls=('--' if n.startswith("Fixed") else '-'))
    plt.xlabel("time [days]"); plt.ylabel("cumulative reward")
    plt.title("One-week cumulative reward")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_week_cumreward.png"), dpi=130); plt.close()

    print(f"\nbest fixed = {best_fixed}.  Saved figures + summary to {OUT}  "
          f"(total {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
