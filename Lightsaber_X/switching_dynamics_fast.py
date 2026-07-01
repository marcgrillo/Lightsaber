"""
Switching-transient study on the FAST engine, with the corrected continuous reward.

Common-random-numbers: for a transition Ci->Cj in regime R, a SWITCH run (Ci for T_warm,
then Cj for T_test) is compared to an ALWAYS-Cj reference on the identical noise; the
post-switch reward difference dr(t) isolates the transient. The reward filters run
continuously (StreamReward). Four handover implementations, realised by manipulating the
parallel controller-bank states at the switch:
  parallel : Cj keeps its own continuously-maintained (warm) state    [realistic, instant]
  ramp     : parallel + a t_ramp cross-fade of the applied actuation   [realistic, bumpless]
  hot      : Ci's bank state is transplanted into Cj                    [naive carry-over]
  cold     : Cj's bank state is reset to zero                           [naive reset]
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
import switching_dynamics as sd
import fast_engine as fe
from reward_stream import StreamReward

FS = 256
OUT = "switching_dynamics"
SENS = {0: 120.0, 1: 12.0, 2: 0.25}
DC_GAIN = {0: 30.0, 1: 44.2, 2: 50.0}


def run_switch_fast(R, ci, cj, T_warm, T_test, seed, mode='parallel', t_ramp=2.0, fs=FS):
    """Fixed-regime closed loop: Ci for T_warm, switch to Cj for T_test. Returns pitch, ctl."""
    plant, sensors, ssc, data, phys = sd.setup(R, T_warm + T_test, seed)
    K = fe.extract_plant_consts(plant, sensors, ssc, fs)
    csoft, chard, ce = fe.build_ctrl_stack(data, phys)
    csoft_zi = np.zeros((3, csoft.shape[1], 2)); chard_zi = np.zeros((3, chard.shape[1], 2))
    N = plant.tst_noise_t.shape[1]
    sus_itm = np.ascontiguousarray(plant.tst_noise_t[0]); sus_etm = np.ascontiguousarray(plant.tst_noise_t[1])
    power = np.ascontiguousarray(plant.input_power)
    rng = np.random.RandomState(seed); scl = (fs/2.0)**0.5 * SENS[R]
    sn_s = rng.normal(0, scl*1e-13, N); sn_h = rng.normal(0, scl*3e-14, N)
    st = fe.new_state()
    hp_zi = np.zeros((K['hp_sos'].shape[0], 2))
    rz0 = np.zeros((K['rad_sos'].shape[0], 2)); rz1 = np.zeros((K['rad_sos'].shape[0], 2))
    az0 = np.zeros((K['act_sos'].shape[0], 2)); az1 = np.zeros((K['act_sos'].shape[0], 2))
    szs = np.zeros((K['ss_sos'].shape[0], 2)); szh = np.zeros((K['ss_sos'].shape[0], 2))

    def seg(n, ti0, active, prev, rn):
        return fe.run_kernel(
            n, ti0, sus_itm, sus_etm, power, sn_s[ti0:ti0+n], sn_h[ti0:ti0+n],
            K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
            K['hp_sos'], hp_zi, K['rad_sos'], rz0, rz1, K['act_sos'], az0, az1,
            csoft, csoft_zi, chard, chard_zi, ce, active, prev, rn,
            K['loc2eig'], K['ss_sos'], szs, szh, K['ss_e2l'],
            K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], st)

    Nw = int(T_warm*fs); Nt = int(T_test*fs)
    p1, c1, _ = seg(Nw, 0, ci, ci, 0)                      # warm-up on Ci
    if mode == 'cold':
        csoft_zi[cj][:] = 0.0; chard_zi[cj][:] = 0.0; a, pr, rn = cj, cj, 0
    elif mode == 'hot':
        csoft_zi[cj] = csoft_zi[ci].copy(); chard_zi[cj] = chard_zi[ci].copy(); a, pr, rn = cj, cj, 0
    elif mode == 'ramp':
        a, pr, rn = cj, ci, int(t_ramp*fs)
    else:  # parallel
        a, pr, rn = cj, cj, 0
    p2, c2, _ = seg(Nt-1, Nw, a, pr, rn)                   # test on Cj
    return np.vstack([p1, p2]), np.vstack([c1, c2])


def main():
    os.makedirs(OUT, exist_ok=True)
    fn, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    T_WARM, T_TEST = 64, 40
    SEEDS = list(range(6))
    transitions = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    bs = sd.BLOCK_S
    post0 = int(T_WARM/bs)            # first post-switch block index
    t_since = (np.arange(0, int(T_TEST/bs))*bs)

    def dr_curve(R, ci, cj, mode, tr=2.0):
        rr = []
        for s in SEEDS:
            ps, cs = run_switch_fast(R, ci, cj, T_WARM, T_TEST, 100+s, mode=mode, t_ramp=tr)
            pr, cr = run_switch_fast(R, cj, cj, T_WARM, T_TEST, 100+s, mode='parallel')  # always-Cj ref
            ssw = StreamReward(fn, w).score(ps, cs)
            sre = StreamReward(fn, w).score(pr, cr)
            n = min(len(ssw), len(sre))
            rr.append((ssw[:n] - sre[:n])[post0:post0+len(t_since)])
        return np.array(rr)

    # ---- fig 1: R0, all transitions, realistic (ramp) handover ----
    print("R0 transitions (ramp handover)...", flush=True)
    results = {}
    plt.figure(figsize=(10, 6))
    for (ci, cj) in transitions:
        v = dr_curve(0, ci, cj, 'ramp'); results[(ci, cj, 'ramp')] = v
        m = v.mean(0); e = v.std(0)/np.sqrt(len(SEEDS)); tt = t_since[:len(m)]
        plt.plot(tt, m, '-o', ms=3, label=f"C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f})")
        plt.fill_between(tt, m-e, m+e, alpha=0.15)
        print(f"  C{ci}->C{cj}: depth={m.min():+.3f} iloss={m.sum()*bs:+.2f}", flush=True)
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("time since switch [s]"); plt.ylabel("reward difference (switch - always-Cj)")
    plt.title("R0: switching transient by transition, bumpless ramp handover (corrected reward)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_R0_transitions_ramp.png"), dpi=130); plt.close()

    # ---- fig 6: handover-mode comparison for the two extreme transitions ----
    print("handover-mode comparison...", flush=True)
    modes = [('cold', 0), ('hot', 0), ('parallel', 0), ('ramp', 2.0), ('ramp', 8.0)]
    labels = ['cold', 'hot', 'parallel', 'ramp2s', 'ramp8s']
    cols = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rows = []
    for ax, (ci, cj) in zip(axes, [(2, 0), (0, 2)]):
        for (mode, tr), lab, col in zip(modes, labels, cols):
            v = dr_curve(0, ci, cj, mode, tr=tr)
            m = v.mean(0)
            ax.plot(t_since[:len(m)], m, '-o', ms=3, color=col, label=lab)
            rows.append((f"C{ci}->C{cj}", lab, DC_GAIN[cj]/DC_GAIN[ci], float(m.min()), float(m.sum()*bs)))
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.set_title(f"R0 C{ci}->C{cj} (g={DC_GAIN[cj]/DC_GAIN[ci]:.2f}): handover modes")
        ax.set_xlabel("time since switch [s]"); ax.set_ylabel("reward diff"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_handover_modes.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "summary_fast.csv"), "w") as f:
        f.write("transition,mode,gain_ratio,depth,iloss\n")
        for tr_, lab, g, d, il in rows:
            f.write(f"{tr_},{lab},{g:.3f},{d:.4f},{il:.3f}\n")
    print("\nSaved corrected sec.6 figures + summary_fast.csv to", OUT)


if __name__ == "__main__":
    main()
