"""
Adiabatic online controller-selection experiment.

Environment: the operating regime drifts slowly (adiabatically) along a severity
path R0 <-> R1 <-> R2.  The three regime noises are generated from the SAME
white-noise seed (so they are coherent) and blended, n(t)=sum_i W_i(t) n_i(t),
which is exactly noise drawn from the time-varying blended ASD --- continuous,
no block edges.  Optical power and sensing-noise scales are blended likewise.

Each policy runs its own closed loop on the IDENTICAL environment (common random
numbers).  Controllers are run as parallel banks with a short bumpless ramp on
each switch (the realistic handover; see switching_dynamics).  Every T_hold the
segment-mean reward (logistic of the multi-band log-RMS score) updates the policy.

Outputs (./bandit_runs/): per-policy history (npz), summary.csv, and figures.
"""
import os, sys, time, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import matplotlib.pyplot as plt
import bandit_rewards
import switching_dynamics as sd
from Lightsaber import Controller
from bandit_simulation import CONTROLLER_CHOICES
import bandit_policies as bp
import fast_engine as fe

FS = 256
OUT = "bandit_runs"

# Characterised mean RAW reward r_c(R_i): rows = regime, cols = controller (from sec.5).
RAW_TABLE = np.array([[168.57, 167.65, 166.63],
                      [159.34, 160.00, 160.01],
                      [151.22, 152.48, 152.90]])
# Oracle reward table[c] = [r_c(R0), r_c(R1), r_c(R2)]
ORACLE_TABLE = RAW_TABLE.T
SENS = {0: 120.0, 1: 12.0, 2: 0.25}


def generate_regime_noises(seed, dur):
    """Three coherent (same-seed) regime noises: R -> (tst_noise(2,N), input_power(N,))."""
    out = {}
    for R in (0, 1, 2):
        plant, _, _, _, _ = sd.setup(R, dur, seed)
        out[R] = (plant.tst_noise_t.copy(), plant.input_power.copy())
    return out


def severity_schedule(N, fs=FS):
    """Adiabatic severity s(t) in [0,2] via keyframes (fraction of horizon, severity)."""
    H = N/fs
    kf_frac = np.array([0.00, 0.12, 0.42, 0.55, 0.85, 1.00])
    kf_sev  = np.array([0.00, 0.00, 2.00, 2.00, 0.00, 0.00])
    t = np.arange(N)/fs
    s = np.interp(t, kf_frac*H, kf_sev)
    return s


def weights_from_severity(s):
    """Map severity s in [0,2] to regime weights (w0,w1,w2), piecewise linear on R0,R1,R2."""
    s = np.asarray(s)
    w0 = np.clip(1.0 - s, 0, 1)
    w2 = np.clip(s - 1.0, 0, 1)
    w1 = 1.0 - w0 - w2
    return np.vstack([w0, w1, w2])  # (3,N)


class RegimeEnv:
    """Diurnal (R0<->R2) drift via stochastic ramped transitions, with Poisson
    micro-seismic (R1) spike events [attack,hold,decay]. Compressed-timescale,
    parametrised port of the SinusoidalPoissonEnv (badai backup)."""
    def __init__(self, total_duration, period=None, exponent=0.33, mu=20, timing_std=None,
                 drift_ramp=40.0, spike_rate=4.0, spike_attack=30.0, spike_hold=150.0,
                 spike_decay=30.0, hard_threshold=True, seed=42):
        self.T = total_duration
        self.period = period if period else total_duration/2.5
        self.hard = hard_threshold
        rng = np.random.RandomState(seed)
        timing_std = timing_std if timing_std is not None else self.period/8.0
        # diurnal "peaky" sine, sampled at stochastic (Poisson) transition times
        def sine(t):
            raw = np.cos(2*np.pi*t/self.period)
            return 0.5 + 0.5*np.sign(raw)*np.abs(raw)**exponent
        self.sine = sine
        num_cycles = int(np.ceil(total_duration/self.period))
        drift_starts = []
        for c in range(num_cycles):
            k = rng.poisson(mu)
            centers = [c*self.period + self.period/4.0, c*self.period + 3*self.period/4.0]
            for _ in range(k):
                drift_starts.append(np.clip(rng.normal(rng.choice(centers), timing_std), 0, total_duration))
        drift_starts.sort()
        self.drift = []; curr = sine(0.0); last = 0.0
        for ts in drift_starts:
            ts = max(ts, last); te = ts + rng.exponential(drift_ramp); tv = sine(te)
            self.drift.append((ts, te, curr, tv)); curr, last = tv, te
        self.start_val = sine(0.0)
        # Poisson R1 spikes
        lam = spike_rate/self.period
        ev = np.cumsum(rng.exponential(1.0/lam, max(2, int(lam*total_duration*3))))
        self.intervals = []
        for ts in ev[ev < total_duration]:
            ta, th, td = rng.exponential(spike_attack), rng.exponential(spike_hold), rng.exponential(spike_decay)
            self.intervals.append((ts, ts+ta, ts+ta+th, ts+ta+th+td))

    def get_weights(self, t):
        val = self.start_val
        for ts, te, vs, ve in self.drift:
            if ts <= t <= te:
                val = vs + (t-ts)/(te-ts+1e-9)*(ve-vs); break
            elif t > te:
                val = ve
        if self.hard:
            if val > 0.75: val = 1.0
            elif val < 0.25: val = 0.0
        w0b, w2b, w1 = val, 1.0-val, 0.0
        for t0, t1, t2, t3 in self.intervals:
            if t0 <= t <= t3:
                if t < t1: w1 = (t-t0)/(t1-t0+1e-9)
                elif t <= t2: w1 = 1.0
                else: w1 = 1.0 - (t-t2)/(t3-t2+1e-9)
                break
        w1 = float(np.clip(w1, 0, 1))
        return (w0b*(1-w1), w1, w2b*(1-w1))

    def weight_array(self, N, fs):
        ts = np.arange(0, int(np.ceil(N/fs))+1)
        Wsec = np.array([self.get_weights(float(t)) for t in ts]).T  # (3, len)
        tsamp = np.arange(N)/fs
        return np.vstack([np.interp(tsamp, ts, Wsec[i]) for i in range(3)])


def reward_map(mean_raw, mode):
    if mode == 'linear':
        return float(np.clip((mean_raw - 148.0)/22.0, 0.0, 1.0))  # ~[148,170] -> [0,1], non-saturating
    return sd.logistic(mean_raw)                                   # logistic, saturates outside R0


def run_policy(policy, noises, W, sens_t, seed, T_hold, feat_names, w_weights, t_ramp=2.0, reward_mode='logistic'):
    N = W.shape[1]
    tst = W[0]*noises[0][0] + W[1]*noises[1][0] + W[2]*noises[2][0]
    power = W[0]*noises[0][1] + W[1]*noises[1][1] + W[2]*noises[2][1]
    plant, sensors, ssc, data, phys = sd.setup(0, int(round(N/FS)), seed)
    tst = tst[:, :N]; power = power[:N]
    plant.tst_noise_t = tst; plant.input_power = power; plant.reset_counters()
    ctrls = {a: Controller(data, phys, plot_dir=None, controller_name=CONTROLLER_CHOICES[a]) for a in (0, 1, 2)}
    hold_n = int(T_hold*FS); ramp_n = int(t_ramp*FS)
    active = policy.choose(0, W[:, 0]); prev = active; switch_k = -1
    tstP = np.zeros((N, 2)); ctl = np.zeros((N, 2)); rd = np.zeros((N, 2)); ssn = np.zeros((N, 2)); cav = np.zeros((N, 1))
    rewards, arms_log, ctx_log, raw_log = [], [], [], []
    for k in range(N-1):
        sensors.n_hard = 3e-14*sens_t[k]; sensors.n_soft = 1e-13*sens_t[k]
        tstP[k+1], _, cav[k+1], _, _, _ = plant.propagate(pum_input_signal=-ctl[k], SS_comp=ssn[k])
        if not np.isfinite(tstP[k+1]).all():
            print("  NaN at k=%d (arm history len %d)" % (k, len(rewards))); break
        rd[k+1] = sensors.sample_readout(input_signal_s=tstP[k+1])
        ssn[k+1] = ssc.sample_compensation(cavity_power=cav[k+1], input_signal=rd[k+1])
        outs = {a: ctrls[a].sample_feedback(input_signal=rd[k+1]) for a in (0, 1, 2)}
        if switch_k >= 0 and (k - switch_k) < ramp_n:
            al = (k - switch_k)/ramp_n
            ctl[k+1] = (1-al)*outs[prev] + al*outs[active]
        else:
            ctl[k+1] = outs[active]
        if (k+1) % hold_n == 0:
            a0, b0 = k+1-hold_n, k+1
            _, sc = sd.block_scores(tstP[a0:b0], ctl[a0:b0], feat_names, w_weights)
            mean_raw = sc.mean() if len(sc) else 167.5
            reward = reward_map(mean_raw, reward_mode)
            policy.update(active, reward)
            rewards.append(reward); arms_log.append(active); ctx_log.append(W[:, k].copy()); raw_log.append(mean_raw)
            new = policy.choose(len(rewards), W[:, k])
            if new != active:
                prev = active; active = new; switch_k = k
    return dict(rewards=np.array(rewards), arms=np.array(arms_log),
                ctx=np.array(ctx_log), raw=np.array(raw_log))


def make_policies(horizon_steps):
    return {
        "AdSwitch":   bp.AdSwitchPolicy(3, horizon=horizon_steps, C1=4.0),
        "SW-UCB":     bp.SlidingWindowUCB(3, window=25, xi=0.3),
        "D-UCB":      bp.DiscountedUCB(3, gamma=0.97, xi=0.3),
        "D-TS":       bp.DiscountedTS(3, gamma=0.92, sigma=0.08),
        "TS-stat":    bp.GaussianTS(3, sigma=0.08),
        "Contextual-TS": bp.ContextualTS(3, sigma=0.05, ctx_noise=0.2, seed=7),
        "Fixed-C0":   bp.FixedArm(0),
        "Fixed-C1":   bp.FixedArm(1),
        "Fixed-C2":   bp.FixedArm(2),
        "Rule-based": bp.RuleBased(seed=7),
        "Oracle":     bp.Oracle(ORACLE_TABLE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1500, help="simulation horizon [s]")
    ap.add_argument("--hold", type=int, default=20, help="bandit decision interval T_hold [s]")
    ap.add_argument("--period", type=float, default=0.0, help="diurnal period [s] (0 = horizon/2.5)")
    ap.add_argument("--reward", choices=["logistic", "linear"], default="linear")
    ap.add_argument("--fast", action="store_true", help="use the validated Numba fast engine")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--test", action="store_true", help="vertex sanity check only")
    args = ap.parse_args()
    outdir = os.path.join(OUT, ("fast_" if args.fast else "") + args.reward)
    os.makedirs(outdir, exist_ok=True)
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")

    if args.test:
        # vertex reproduction: fixed regime R, controller C; compare mean logistic reward
        N = int(200*FS)
        noises = generate_regime_noises(args.seed, 200)
        for R in (0, 1, 2):
            W = np.zeros((3, N)); W[R] = 1.0
            sens_t = np.full(N, SENS[R])
            pol = bp.FixedArm(R)  # play the diagonal controller
            h = run_policy(pol, noises, W, sens_t, args.seed, args.hold, feat_names, w, reward_mode=args.reward)
            print(f"R{R} C{R}: mean_raw={h['raw'].mean():.2f}  mean_logistic={h['rewards'].mean():.3f}  (n={len(h['rewards'])})", flush=True)
        return

    N = int(args.horizon*FS)
    print(f"Generating coherent regime noises ({args.horizon}s)...", flush=True)
    noises = generate_regime_noises(args.seed, args.horizon)
    env = RegimeEnv(args.horizon, period=(args.period or None), seed=args.seed)
    W = env.weight_array(N, FS)                      # diurnal R0<->R2 drift + Poisson R1 spikes
    s = 1.0*W[1] + 2.0*W[2]                           # severity proxy for plots
    sens_t = SENS[0]*W[0] + SENS[1]*W[1] + SENS[2]*W[2]
    print(f"Regime env: period={env.period:.0f}s, {len(env.intervals)} R1 spikes, {len(env.drift)} drift transitions", flush=True)
    horizon_steps = N // int(args.hold*FS)
    optimal = np.argmax(ORACLE_TABLE @ W, axis=0)  # per-sample optimal arm

    runner = fe.run_policy_fast if args.fast else run_policy
    results = {}
    for name, pol in make_policies(horizon_steps).items():
        np.random.seed(1234)  # reproducible exploration for stochastic policies
        t0 = time.time()
        h = runner(pol, noises, W, sens_t, args.seed, args.hold, feat_names, w, reward_mode=args.reward)
        results[name] = h
        print(f"{name:11s}: cum_reward={h['rewards'].sum():.2f}  switches={int((np.diff(h['arms'])!=0).sum())}  ({time.time()-t0:.0f}s)", flush=True)

    # decision-step optimal arm (context sampled at each decision)
    hold_n = int(args.hold*FS)
    dec_k = np.arange(1, len(results['Oracle']['rewards'])+1)*hold_n - hold_n
    opt_dec = optimal[np.clip(dec_k, 0, N-1)]

    np.savez_compressed(os.path.join(outdir, "histories.npz"),
                        W=W[:, ::FS], sev=s[::FS], optimal=optimal[::FS],
                        **{f"{k}_rewards": v['rewards'] for k, v in results.items()},
                        **{f"{k}_arms": v['arms'] for k, v in results.items()})

    oracle_cum = results['Oracle']['rewards'].sum()
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write("policy,cum_reward,mean_reward,regret_vs_oracle,switches,frac_optimal\n")
        for name, v in results.items():
            n = min(len(v['arms']), len(opt_dec))
            frac = float(np.mean(v['arms'][:n] == opt_dec[:n]))
            sw = int((np.diff(v['arms']) != 0).sum())
            f.write(f"{name},{v['rewards'].sum():.3f},{v['rewards'].mean():.4f},"
                    f"{oracle_cum - v['rewards'].sum():.3f},{sw},{frac:.3f}\n")

    # ---- fig 1: cumulative reward ----
    plt.figure(figsize=(10, 6))
    order = ["Oracle", "Contextual-TS", "Rule-based", "AdSwitch", "SW-UCB", "D-UCB", "D-TS", "TS-stat",
             "Fixed-C0", "Fixed-C1", "Fixed-C2"]
    for name in order:
        v = results[name]
        plt.plot(np.arange(len(v['rewards'])), np.cumsum(v['rewards']),
                 label=name, lw=(2.2 if name in ("Oracle", "AdSwitch") else 1.3),
                 ls=('--' if name.startswith("Fixed") else '-'))
    plt.xlabel("decision step"); plt.ylabel("cumulative reward")
    plt.title(f"Adiabatic regime drift: cumulative reward (H={args.horizon}s, T_hold={args.hold}s)")
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_cumreward.png"), dpi=130); plt.close()

    # ---- fig 2: severity + arm timelines for key policies ----
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    td = dec_k/FS
    axes[0].plot(np.arange(N)/FS, s, 'k'); axes[0].set_ylabel("severity"); axes[0].grid(alpha=0.3)
    axes[0].set_title("regime severity and controller choices")
    for ax, name in zip(axes[1:], ["Oracle", "Contextual-TS", "AdSwitch"]):
        v = results[name]
        ax.step(td[:len(v['arms'])], v['arms'], where='post')
        ax.set_ylabel(name); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["C0", "C1", "C2"]); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_timelines.png"), dpi=130); plt.close()

    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
