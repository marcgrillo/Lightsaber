"""
Realistic controller handover: PARALLEL-running banks (+ ramped/bumpless variant).

All candidate controllers run continuously on the same readout, each keeping its own state;
only the active output drives the suspensions. A switch = change which output is applied
('parallel' = instantaneous) or cross-fade over t_ramp seconds ('ramp' = bumpless).

Compares the realistic modes against the cold/naive-hot bounds (loaded from the earlier run),
using the same common-random-numbers method (dR vs always-Cj reference). Saves to ./switching_dynamics/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
import switching_dynamics as sd
from Lightsaber import Controller
from bandit_simulation import CONTROLLER_CHOICES

OUT = "switching_dynamics"
FS = sd.FS
DC_GAIN = sd.DC_GAIN


def run_parallel(regime_key, seg_ctrls, seg_durs, noise, power, seed, mode='parallel', t_ramp=2.0):
    """Parallel-running controllers; active output applied. mode='parallel' (instant) or 'ramp' (cross-fade t_ramp s)."""
    total = sum(seg_durs)
    plant, sensors, ssc, data, phys = sd.setup(regime_key, total, seed)
    N = int(total * FS)
    noise = noise[:, :N] if noise.shape[1] >= N else np.hstack([noise, np.zeros((2, N - noise.shape[1]))])
    power = power[:N] if power.shape[0] >= N else np.hstack([power, np.zeros(N - power.shape[0])])
    plant.tst_noise_t = noise; plant.input_power = power
    arms = sorted(set(int(a) for a in seg_ctrls))
    ctrls = {a: Controller(data, phys, plot_dir=None, controller_name=CONTROLLER_CHOICES[a]) for a in arms}
    ends_k = (np.cumsum(seg_durs) * FS).astype(int)
    seg = 0
    active = int(seg_ctrls[0]); prev = active; switch_k = -1
    ramp_n = int(t_ramp * FS)
    tstP = np.zeros((N, 2)); ctl = np.zeros((N, 2)); rd = np.zeros((N, 2)); ssn = np.zeros((N, 2)); cav = np.zeros((N, 1))
    for k in range(N - 1):
        if seg < len(ends_k) - 1 and k >= ends_k[seg]:
            seg += 1; prev = active; active = int(seg_ctrls[seg]); switch_k = k
        tstP[k+1], _, cav[k+1], _, _, _ = plant.propagate(pum_input_signal=-ctl[k], SS_comp=ssn[k])
        if not np.isfinite(tstP[k+1]).all():
            break
        rd[k+1] = sensors.sample_readout(input_signal_s=tstP[k+1])
        ssn[k+1] = ssc.sample_compensation(cavity_power=cav[k+1], input_signal=rd[k+1])
        outs = {a: ctrls[a].sample_feedback(input_signal=rd[k+1]) for a in arms}  # ALL update state
        if mode == 'ramp' and switch_k >= 0 and ramp_n > 0 and (k - switch_k) < ramp_n:
            al = (k - switch_k) / ramp_n
            ctl[k+1] = (1 - al) * outs[prev] + al * outs[active]
        else:
            ctl[k+1] = outs[active]
    return tstP, ctl


def metrics(dR_raw, dR_log, t_since, nseeds, bs):
    m = dR_raw.mean(0); depth = m.min(); idx = int(m.argmin())
    sem = dR_raw.std(0) / np.sqrt(nseeds); rec = np.nan
    for j in range(idx, len(m)):
        if m[j] >= -max(sem[j], 1e-9):
            rec = t_since[j] if j < len(t_since) else np.nan; break
    return depth, rec, float(np.sum(m) * bs), float(np.sum(dR_log.mean(0)) * bs)


def main():
    fn, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    T_WARM, T_TEST = 15, 30
    SEEDS = list(range(5))
    R = 0
    transitions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    bs = int(sd.BLOCK_S * FS)
    nb = int((T_WARM + T_TEST) * FS) // bs
    tfull = (np.arange(nb) * sd.BLOCK_S) + sd.BLOCK_S / 2
    post = tfull > T_WARM
    t_since = tfull[post] - T_WARM

    # modes to compute: parallel (all transitions), ramp2/ramp8 (extreme transitions only)
    runs = [('parallel', 0.0, transitions)]
    runs += [('ramp', 2.0, [(2, 0), (0, 2)]), ('ramp', 8.0, [(2, 0), (0, 2)])]

    ref_cache = {}
    results = {}  # (ci,cj,modename) -> (dR_raw, dR_log)
    for (mode, tr, trans_list) in runs:
        mname = mode if mode == 'parallel' else f'ramp{int(tr)}'
        for (ci, cj) in trans_list:
            dRr, dRl = [], []
            for s in SEEDS:
                noise, power = sd.gen_noise(R, T_WARM + T_TEST, seed=100 + s)
                key = (cj, s)
                if key not in ref_cache:
                    tP, cl = run_parallel(R, [cj], [T_WARM + T_TEST], noise, power, seed=100 + s, mode='parallel')
                    _, sc = sd.block_scores(tP, cl, fn, w); ref_cache[key] = sc
                ref = ref_cache[key]
                tP, cl = run_parallel(R, [ci, cj], [T_WARM, T_TEST], noise, power, seed=100 + s, mode=mode, t_ramp=tr)
                _, sw = sd.block_scores(tP, cl, fn, w)
                n = min(len(ref), len(sw), len(post)); m = post[:n]
                dRr.append(sw[:n][m] - ref[:n][m])
                dRl.append(sd.logistic(sw[:n][m]) - sd.logistic(ref[:n][m]))
            dRr = np.array(dRr); dRl = np.array(dRl)
            results[(ci, cj, mname)] = (dRr, dRl)
            d, rec, ilr, ill = metrics(dRr, dRl, t_since, len(SEEDS), sd.BLOCK_S)
            print(f"R0 C{ci}->C{cj} {mname:8s}: gain={DC_GAIN[cj]/DC_GAIN[ci]:.3f} depth={d:+.3f} "
                  f"recovery={rec if rec==rec else float('nan'):.1f}s iloss_raw={ilr:+.2f} iloss_log={ill:+.4f}", flush=True)

    np.savez_compressed(os.path.join(OUT, "transients_realistic.npz"), t_since=t_since,
                        **{f"C{ci}C{cj}_{mn}_raw": v[0] for (ci, cj, mn), v in results.items()})

    # summary csv (parallel for all transitions)
    with open(os.path.join(OUT, "summary_realistic.csv"), "w") as f:
        f.write("regime,from,to,mode,gain_ratio,depth_raw,recovery_s,iloss_raw,iloss_log\n")
        for (ci, cj, mn), (dRr, dRl) in results.items():
            d, rec, ilr, ill = metrics(dRr, dRl, t_since, len(SEEDS), sd.BLOCK_S)
            f.write(f"R0,C{ci},C{cj},{mn},{DC_GAIN[cj]/DC_GAIN[ci]:.3f},{d:.4f},{rec:.1f},{ilr:.3f},{ill:.5f}\n")

    # ---- fig A: parallel transient by transition (realistic fig1) ----
    plt.figure(figsize=(10, 6))
    for (ci, cj) in transitions:
        m = results[(ci, cj, 'parallel')][0].mean(0); e = results[(ci, cj, 'parallel')][0].std(0) / np.sqrt(len(SEEDS))
        plt.plot(t_since, m, '-o', ms=3, label=f"C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f})")
        plt.fill_between(t_since, m - e, m + e, alpha=0.15)
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("time since switch [s]"); plt.ylabel("raw reward diff (switch - always-Cj)")
    plt.title("R0: switching transient, PARALLEL-running (realistic)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_parallel_transitions.png"), dpi=130); plt.close()

    # ---- fig B: 4-mode comparison for the two extreme transitions ----
    old = np.load(os.path.join(OUT, "transients.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (ci, cj) in zip(axes, [(2, 0), (0, 2)]):
        series = [
            ('cold', old[f"R0_C{ci}C{cj}_cold_raw"], 'tab:red'),
            ('hot', old[f"R0_C{ci}C{cj}_hot_raw"], 'tab:blue'),
            ('parallel', results[(ci, cj, 'parallel')][0], 'tab:green'),
            ('ramp2s', results[(ci, cj, 'ramp2')][0], 'tab:orange'),
            ('ramp8s', results[(ci, cj, 'ramp8')][0], 'tab:purple'),
        ]
        for nm, arr, c in series:
            m = arr.mean(0); ax.plot(t_since[:len(m)], m, '-o', ms=3, color=c, label=nm)
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f"R0 C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f}): handover modes")
        ax.set_xlabel("time since switch [s]"); ax.set_ylabel("raw reward diff"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_handover_modes.png"), dpi=130); plt.close()

    print("\nSaved realistic results to", OUT)


if __name__ == "__main__":
    main()
