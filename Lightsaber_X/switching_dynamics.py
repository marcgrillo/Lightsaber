"""
Switching dynamics: how a controller switch affects the reward (transient analysis).

Method: common-random-numbers. For each regime R, seed s, and transition Ci->Cj we run
  - SWITCH run: Ci for T_warm, then switch to Cj for T_test
  - REFERENCE run: always Cj for (T_warm+T_test)
on the SAME injected noise + same sensor-noise seed. The post-switch reward difference
  dR(t) = score_switch(t) - score_reference(t)
isolates the switching transient (reference = fully-settled Cj). Averaged over seeds.

Switch modes:
  - hot : controller filter state preserved across the switch (realistic; default behaviour)
  - cold: controller hard-filter state reset to zero on the switch (naive)

Explores dependence on: transition (gain jump), hot/cold, regime, and averaging window.
Saves data (npz), summary (csv), figures, and FINDINGS to ./switching_dynamics/.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
from Lightsaber import Plant, Sensors, SS_compensation, Controller
from bandit_simulation import (DEFAULT_FLAGS, REGIME_POWER_SCALE, REGIME_SENSING_SCALE,
                               REGIME_OSEM_SCALE, REGIME_SEISMIC_SCALE, REGIME_FILES,
                               CONTROLLER_CHOICES, T_FFT, FS)

OUT = "switching_dynamics"; os.makedirs(OUT, exist_ok=True)
FS = 256
BLOCK_S = 2
DC_GAIN = {0: 30.0, 1: 44.2, 2: 50.0}
LOGIT_C, LOGIT_S = 167.5, 2.5

# eigen transform (hard mode)
_L, _Ri, _Re = 3994.5, 1934, 2245
_g1, _g2 = 1 - _L/_Ri, 1 - _L/_Re
_r = 0.5*(_g1-_g2+np.sqrt((_g1-_g2)**2+4))
LOCAL2EIG = np.array([[1, _r], [-_r, 1]])/(1+_r**2)


def logistic(x):
    return 1.0/(1.0+np.exp(-(x-LOGIT_C)/LOGIT_S))


def setup(regime_key, dur_s, seed):
    rn, rf = REGIME_FILES[str(regime_key)]
    f = DEFAULT_FLAGS.copy()
    f['scale_OSEM_L'] = f['scale_OSEM_P'] = REGIME_OSEM_SCALE.get(rn, 1.0)
    f['P'] = f['P']*REGIME_POWER_SCALE.get(rn, 1.0)
    ss = REGIME_SEISMIC_SCALE.get(rn, 1.0); f['scale_ITM_ISI_L'] = f['scale_ETM_ISI_L'] = ss
    sn = REGIME_SENSING_SCALE.get(rn, 1.0); f['n_hard'] *= sn; f['n_soft'] *= sn
    data = {'sampling_frequency': f['fs'], 'duration_batch': dur_s, 'duration_fft': T_FFT}
    phys = {'P': f['P'], 'L': f['L'], 'R_ITM': f['R_ITM'], 'R_ETM': f['R_ETM'], 't_ITM': f['t_ITM']}
    sens = {'noise_hard_mode': f['n_hard'], 'noise_soft_mode': f['n_soft']}
    pars = {'scale_ITM_ISI_L': f['scale_ITM_ISI_L'], 'scale_ETM_ISI_L': f['scale_ETM_ISI_L'],
            'scale_OSEM_L': f['scale_OSEM_L'], 'scale_OSEM_P': f['scale_OSEM_P'], 'scale_RIN': f['scale_RIN']}
    nf = [rf, rf, 'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt', 'noise_inputs/O3_power_psd.csv']
    tf = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt',
          'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    ref = 'noise_inputs/aicReferenceData_Aplus.txt'
    plant = Plant(phys, data, pars, plot_dir=None, noise_files=nf, reference_data_file=ref, transfer_files=tf, seed=seed)
    sensors = Sensors(sens, phys, data, seed=seed)
    ssc = SS_compensation(data, phys, plant, plot_dir=None)
    return plant, sensors, ssc, data, phys


def gen_noise(regime_key, dur_s, seed):
    p, _, _, _, _ = setup(regime_key, dur_s, seed)
    return p.tst_noise_t.copy(), p.input_power.copy()


def run_segmented(regime_key, seg_controllers, seg_durs, noise, power, seed, mode='hot'):
    """seg_controllers: list of controller keys; seg_durs: list of seconds. On boundary, switch (hot/cold)."""
    total = sum(seg_durs)
    plant, sensors, ssc, data, phys = setup(regime_key, total, seed)
    N = int(total*FS)
    noise = noise[:, :N] if noise.shape[1] >= N else np.hstack([noise, np.zeros((2, N-noise.shape[1]))])
    power = power[:N] if power.shape[0] >= N else np.hstack([power, np.zeros(N-power.shape[0])])
    plant.tst_noise_t = noise; plant.input_power = power
    ctrl = Controller(data, phys, plot_dir=None, controller_name=CONTROLLER_CHOICES[int(seg_controllers[0])])
    ends_k = (np.cumsum(seg_durs)*FS).astype(int)
    ci = 0
    tstP = np.zeros((N, 2)); ctl = np.zeros((N, 2)); rd = np.zeros((N, 2)); ssn = np.zeros((N, 2)); cav = np.zeros((N, 1))
    for k in range(N-1):
        if ci < len(ends_k)-1 and k >= ends_k[ci]:
            ci += 1
            # capture the LIVE controller state (it lives in global_control_sos_state, not the stale attrs)
            live_soft = ctrl.global_control_sos_state[0].copy()
            live_hard = ctrl.global_control_sos_state[1].copy()
            ctrl.controller_name = CONTROLLER_CHOICES[int(seg_controllers[ci])]
            ctrl.set_feedback_filter_hard(plot_dir=None)  # rebuilds hard SOS, clobbers global state -> restore below
            # soft filter is unchanged across C0/C1/C2, so its state is ALWAYS preserved.
            # hot: preserve the hard-filter live state (only the output gain changes).
            # cold: reset the hard-filter state to zero (naive handover).
            ctrl.global_control_sos_state = [live_soft,
                                             live_hard if mode == 'hot' else np.zeros_like(live_hard)]
        tstP[k+1], _, cav[k+1], _, _, _ = plant.propagate(pum_input_signal=-ctl[k], SS_comp=ssn[k])
        if not np.isfinite(tstP[k+1]).all():
            break
        rd[k+1] = sensors.sample_readout(input_signal_s=tstP[k+1])
        ssn[k+1] = ssc.sample_compensation(cavity_power=cav[k+1], input_signal=rd[k+1])
        ctl[k+1] = ctrl.sample_feedback(input_signal=rd[k+1])
    return tstP, ctl


def block_scores(tstP, ctl, feat_names, w):
    """Per-2s-block raw reward scores over a window. The band filters are applied to the WHOLE
    window once (so low-frequency bands settle and are meaningful), and RMS is taken per 2s
    block of the filtered signal -- the convention of reward_utils/bandit_simulation, for which
    the npz weights were calibrated. Returns (block_center_times, per_block_raw_scores);
    the window reward is the mean of the returned scores."""
    err_h = (LOCAL2EIG @ tstP.T).T[:, 1]
    u_h = (LOCAL2EIG @ ctl.T).T[:, 1]
    feats = {}
    feats.update(bandit_rewards.band_feats_blocks(err_h, bandit_rewards.ERR_BANDS))
    feats.update(bandit_rewards.band_feats_blocks(u_h, bandit_rewards.U_BANDS))
    nb = len(next(iter(feats.values())))
    Z = np.zeros(nb)
    for wi, fn in zip(w, feat_names):
        if fn in feats:
            Z += wi*np.log(feats[fn] + 1e-30)
    bs = int(BLOCK_S*FS)
    times = (np.arange(nb)*bs + bs/2)/FS
    return times, -Z


def main():
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    T_WARM, T_TEST = 15, 30
    SEEDS = list(range(5))
    regimes = [0, 1, 2]
    transitions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    # hot for all regimes; cold only R0 (to contrast)
    cases = []
    for R in regimes:
        for (ci, cj) in transitions:
            cases.append((R, ci, cj, 'hot'))
    for (ci, cj) in transitions:
        cases.append((0, ci, cj, 'cold'))

    bs = int(BLOCK_S*FS)
    nb_test = (int((T_WARM+T_TEST)*FS)) // bs
    # post-switch block mask (center time > T_WARM)
    times_full = (np.arange(nb_test)*BLOCK_S)+BLOCK_S/2
    post = times_full > T_WARM
    t_since = times_full[post]-T_WARM

    ref_cache = {}  # (R, Cj, seed) -> scores
    results = {}    # (R,ci,cj,mode) -> dict of arrays over seeds

    for (R, ci, cj, mode) in cases:
        dR_raw, dR_log = [], []
        for s in SEEDS:
            noise, power = gen_noise(R, T_WARM+T_TEST, seed=100+s)
            key = (R, cj, s)
            if key not in ref_cache:
                tP, cl = run_segmented(R, [cj], [T_WARM+T_TEST], noise, power, seed=100+s, mode='hot')
                _, sc = block_scores(tP, cl, feat_names, w)
                ref_cache[key] = sc
            ref = ref_cache[key]
            tP, cl = run_segmented(R, [ci, cj], [T_WARM, T_TEST], noise, power, seed=100+s, mode=mode)
            _, sw = block_scores(tP, cl, feat_names, w)
            n = min(len(ref), len(sw), len(post))
            m = post[:n]
            dR_raw.append(sw[:n][m]-ref[:n][m])
            dR_log.append(logistic(sw[:n][m])-logistic(ref[:n][m]))
        dR_raw = np.array(dR_raw); dR_log = np.array(dR_log)
        results[(R, ci, cj, mode)] = {'raw': dR_raw, 'log': dR_log}
        mraw = dR_raw.mean(0)
        depth = mraw.min(); idx = int(mraw.argmin())
        # recovery: first block after the dip where mean returns within 1 sem of 0
        sem = dR_raw.std(0)/np.sqrt(len(SEEDS))
        rec = np.nan
        for j in range(idx, len(mraw)):
            if mraw[j] >= -max(sem[j], 1e-9):
                rec = t_since[j] if j < len(t_since) else np.nan; break
        iloss_raw = float(np.sum(mraw)*BLOCK_S)
        iloss_log = float(np.sum(dR_log.mean(0))*BLOCK_S)
        gr = DC_GAIN[cj]/DC_GAIN[ci]
        print(f"R{R} C{ci}->C{cj} {mode:4s}: gain_ratio={gr:.3f} depth={depth:+.3f} "
              f"recovery={rec if rec==rec else float('nan'):.1f}s iloss_raw={iloss_raw:+.2f} iloss_log={iloss_log:+.4f}", flush=True)

    # ---- save data ----
    np.savez_compressed(os.path.join(OUT, "transients.npz"),
                        t_since=t_since, seeds=np.array(SEEDS), T_WARM=T_WARM, T_TEST=T_TEST,
                        **{f"R{R}_C{ci}C{cj}_{mode}_raw": v['raw'] for (R, ci, cj, mode), v in results.items()})

    # ---- summary csv ----
    with open(os.path.join(OUT, "summary.csv"), "w") as fcsv:
        fcsv.write("regime,from,to,mode,gain_ratio,depth_raw,recovery_s,iloss_raw,iloss_log\n")
        for (R, ci, cj, mode), v in results.items():
            mraw = v['raw'].mean(0); depth = mraw.min(); idx = int(mraw.argmin())
            sem = v['raw'].std(0)/np.sqrt(len(SEEDS)); rec = np.nan
            for j in range(idx, len(mraw)):
                if mraw[j] >= -max(sem[j], 1e-9):
                    rec = t_since[j] if j < len(t_since) else np.nan; break
            fcsv.write(f"R{R},C{ci},C{cj},{mode},{DC_GAIN[cj]/DC_GAIN[ci]:.3f},{depth:.4f},"
                       f"{rec:.1f},{np.sum(mraw)*BLOCK_S:.3f},{np.sum(v['log'].mean(0))*BLOCK_S:.5f}\n")

    # ---- fig1: R0 all transitions (hot) dR(t) ----
    plt.figure(figsize=(10, 6))
    for (ci, cj) in transitions:
        v = results[(0, ci, cj, 'hot')]['raw']; m = v.mean(0); e = v.std(0)/np.sqrt(len(SEEDS))
        plt.plot(t_since, m, '-o', ms=3, label=f"C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f})")
        plt.fill_between(t_since, m-e, m+e, alpha=0.15)
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("time since switch [s]"); plt.ylabel("raw reward difference (switch - always-Cj)")
    plt.title("R0: switching transient by transition (hot)"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_R0_transitions_hot.png"), dpi=130); plt.close()

    # ---- fig2: hot vs cold (R0) for the two extreme gain jumps ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (ci, cj) in zip(axes, [(0, 2), (2, 0)]):
        for mode, c in [('hot', 'C0'), ('cold', 'C3')]:
            v = results[(0, ci, cj, mode)]['raw']; m = v.mean(0); e = v.std(0)/np.sqrt(len(SEEDS))
            ax.plot(t_since, m, '-o', ms=3, color=c, label=mode)
            ax.fill_between(t_since, m-e, m+e, alpha=0.15, color=c)
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f"R0 C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f})")
        ax.set_xlabel("time since switch [s]"); ax.set_ylabel("raw reward diff"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_hot_vs_cold_R0.png"), dpi=130); plt.close()

    # ---- fig3: C0->C2 across regimes (hot) ----
    plt.figure(figsize=(10, 6))
    for R in regimes:
        v = results[(R, 0, 2, 'hot')]['raw']; m = v.mean(0); e = v.std(0)/np.sqrt(len(SEEDS))
        plt.plot(t_since, m, '-o', ms=3, label=f"R{R}")
        plt.fill_between(t_since, m-e, m+e, alpha=0.15)
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("time since switch [s]"); plt.ylabel("raw reward diff"); plt.title("C0->C2 transient by regime (hot)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_C0C2_by_regime.png"), dpi=130); plt.close()

    # ---- fig4: bias vs averaging window (R0, hot) ----
    plt.figure(figsize=(10, 6))
    for (ci, cj) in transitions:
        m = results[(0, ci, cj, 'hot')]['raw'].mean(0)
        cum = np.cumsum(m)/np.arange(1, len(m)+1)
        plt.plot(t_since, cum, '-', label=f"C{ci}->C{cj}")
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("averaging window after switch [s]"); plt.ylabel("mean reward bias over window")
    plt.title("R0: how the averaging (hold) window washes out the switch transient")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_bias_vs_window_R0.png"), dpi=130); plt.close()

    print("\nSaved data + figures to", OUT)


if __name__ == "__main__":
    main()
