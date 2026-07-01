"""
Numba JIT closed-loop kernel, mirroring the validated Plant-class slow engine.

It reproduces the exact per-sample sequence of Plant.propagate / Sensors.sample_readout /
SS_compensation.sample_compensation / Controller.sample_feedback, including the one-sample
control and SS-compensation delays. The Sidles-Sigg compensator (whose gain depends on the
running cavity power P_av) is implemented as a unit-gain SOS filter with a dynamic output
scale -- exact in steady state, and validated against the slow engine.

Constants/SOS are extracted from the validated Plant/Sensors/Controller/SS objects, so the
physics is identical by construction; `validate()` confirms it reproduces the sec.5 numbers.
"""
import numpy as np
from numba import njit
from scipy import signal
import switching_dynamics as sd
from Lightsaber import Controller
from bandit_simulation import CONTROLLER_CHOICES
from reward_stream import StreamReward

C_LIGHT = 299792458.0
SS_K = 2.567652  # rad_to_angle gain (radiation-torque coupling), as in SS_compensation


@njit(cache=True)
def _sos_step(sos, x, zi):
    """One-sample SOS cascade (Direct Form II transposed); mutates zi in place."""
    y = x
    for s in range(sos.shape[0]):
        b0, b1, b2, a1, a2 = sos[s, 0], sos[s, 1], sos[s, 2], sos[s, 4], sos[s, 5]
        w = y
        y = b0*w + zi[s, 0]
        zi[s, 0] = b1*w - a1*y + zi[s, 1]
        zi[s, 1] = b2*w - a2*y
    return y


@njit(cache=True)
def run_kernel(n, ti0,
               sus_itm, sus_etm, input_power, sn_soft, sn_hard,
               bs_matrix, rho_itm, t_itm, lam, dc_offset, p_dc,
               hp_sos, hp_zi,
               rad_sos, rad_zi0, rad_zi1,
               act_sos, act_zi0, act_zi1,
               csoft, csoft_zi, chard, chard_zi, ctrl_e2l, active, prev, ramp_n,
               loc2eig,
               ss_sos, ss_zi_s, ss_zi_h, ss_e2l,
               dydth_soft, dydth_hard, kk_lp, p_const,
               state):
    """state = [pitch0,pitch1, ctrl0,ctrl1, ss0,ss1, Pav, Ncount] (mutated)."""
    pitch_out = np.zeros((n, 2)); ctrl_out = np.zeros((n, 2)); read_out = np.zeros((n, 2))
    p0, p1 = state[0], state[1]
    c0, c1 = state[2], state[3]
    s0, s1 = state[4], state[5]
    Pav, Nc = state[6], state[7]
    for k in range(n):
        ti = ti0 + k
        # beam spots & cavity length change (uses previous pitch)
        bs0 = bs_matrix[0, 0]*p0 + bs_matrix[0, 1]*p1
        bs1 = bs_matrix[1, 0]*p0 + bs_matrix[1, 1]*p1
        dL = bs0*p0 + bs1*p1
        dL_hp = _sos_step(hp_sos, dL, hp_zi)
        phase = 4.0*np.pi*dL_hp/lam
        dre = 1.0 - rho_itm*np.cos(phase)
        dim = -rho_itm*np.sin(phase)
        cav = input_power[ti]*t_itm/(dre*dre + dim*dim)
        # radiation-pressure torques and propagation to angle
        tau0 = (2.0/C_LIGHT)*(cav*bs0 + dc_offset[0]*(cav - p_dc))
        tau1 = (2.0/C_LIGHT)*(cav*bs1 + dc_offset[1]*(cav - p_dc))
        tn0 = _sos_step(rad_sos, tau0, rad_zi0)
        tn1 = _sos_step(rad_sos, tau1, rad_zi1)
        as0 = _sos_step(act_sos, -c0, act_zi0)
        as1 = _sos_step(act_sos, -c1, act_zi1)
        np0 = sus_itm[ti] + tn0 + as0 - s0
        np1 = sus_etm[ti] + tn1 + as1 - s1
        # sensing (eigenbasis) with noise
        re_s = loc2eig[0, 0]*np0 + loc2eig[0, 1]*np1 + sn_soft[k]
        re_h = loc2eig[1, 0]*np0 + loc2eig[1, 1]*np1 + sn_hard[k]
        # Sidles-Sigg compensation (dynamic gain from running P_av)
        Pav = Pav + (1.0/Nc)*(cav - Pav)
        if Nc < 1000.0:
            Nc += 1.0
        g_s = -SS_K*2.0*Pav/C_LIGHT*dydth_soft*kk_lp
        g_h = -(1.0 - p_const/Pav)*SS_K*2.0*Pav/C_LIGHT*dydth_hard*kk_lp
        ss_s = _sos_step(ss_sos, re_s, ss_zi_s)*g_s
        ss_h = _sos_step(ss_sos, re_h, ss_zi_h)*g_h
        ns0 = ss_e2l[0, 0]*ss_s + ss_e2l[0, 1]*ss_h
        ns1 = ss_e2l[1, 0]*ss_s + ss_e2l[1, 1]*ss_h
        # controller -> local actuation (one-sample delay: applied next iteration).
        # All three controller banks run every sample (parallel banks, each keeps its state).
        # The applied actuation is the active bank, cross-faded from the previous bank over the
        # first ramp_n samples after a switch (bumpless handover). No negation here; the minus
        # is applied at the actuator above as act(-c).
        os0 = _sos_step(csoft[0], re_s, csoft_zi[0]); oh0 = _sos_step(chard[0], re_h, chard_zi[0])
        os1 = _sos_step(csoft[1], re_s, csoft_zi[1]); oh1 = _sos_step(chard[1], re_h, chard_zi[1])
        os2 = _sos_step(csoft[2], re_s, csoft_zi[2]); oh2 = _sos_step(chard[2], re_h, chard_zi[2])
        os_a = os0 if active == 0 else (os1 if active == 1 else os2)
        oh_a = oh0 if active == 0 else (oh1 if active == 1 else oh2)
        if ramp_n > 0 and k < ramp_n:
            os_p = os0 if prev == 0 else (os1 if prev == 1 else os2)
            oh_p = oh0 if prev == 0 else (oh1 if prev == 1 else oh2)
            al = (k + 1.0)/ramp_n
            out_s = (1.0 - al)*os_p + al*os_a
            out_h = (1.0 - al)*oh_p + al*oh_a
        else:
            out_s = os_a; out_h = oh_a
        nc0 = ctrl_e2l[0, 0]*out_s + ctrl_e2l[0, 1]*out_h
        nc1 = ctrl_e2l[1, 0]*out_s + ctrl_e2l[1, 1]*out_h
        pitch_out[k, 0] = np0; pitch_out[k, 1] = np1
        ctrl_out[k, 0] = nc0; ctrl_out[k, 1] = nc1
        read_out[k, 0] = re_s; read_out[k, 1] = re_h
        p0, p1, c0, c1, s0, s1 = np0, np1, nc0, nc1, ns0, ns1
    state[0], state[1], state[2], state[3] = p0, p1, c0, c1
    state[4], state[5], state[6], state[7] = s0, s1, Pav, Nc
    return pitch_out, ctrl_out, read_out


def extract_plant_consts(plant, sensors, ssc, fs):
    z_ss, p_ss = ssc.z_ss, ssc.p_ss
    unit_ss = signal.zpk2sos(*signal.bilinear_zpk(z_ss, p_ss, 1.0, fs))
    return dict(
        bs_matrix=np.ascontiguousarray(plant.bs_matrix, np.float64),
        rho_itm=float(plant.rho_ITM), t_itm=float(plant.t_ITM), lam=float(plant.lambda0),
        dc_offset=np.ascontiguousarray(plant.dc_offset, np.float64), p_dc=float(plant.P_dc),
        hp_sos=np.ascontiguousarray(plant.high_pass_sos, np.float64),
        rad_sos=np.ascontiguousarray(plant.tstP2P_sos, np.float64),
        act_sos=np.ascontiguousarray(plant.pumP_2_tstP_sos, np.float64),
        loc2eig=np.ascontiguousarray(sensors.local2eigen, np.float64),
        ss_sos=np.ascontiguousarray(unit_ss, np.float64), ss_e2l=np.ascontiguousarray(ssc.eigen2local, np.float64),
        dydth_soft=float(ssc.dydth_soft), dydth_hard=float(ssc.dydth_hard),
        kk_lp=float(ssc.kk_lp), p_const=float(ssc.P_const),
    )


def ctrl_sos(ctrl):
    return (np.ascontiguousarray(ctrl.global_control_sos[0], np.float64),
            np.ascontiguousarray(ctrl.global_control_sos[1], np.float64),
            np.ascontiguousarray(ctrl.eigen2local, np.float64))


def new_state():
    return np.zeros(8) + np.array([0, 0, 0, 0, 0, 0, 0.0, 1.0])  # Nc starts at 1


def build_ctrl_stack(data, phys):
    """Stacked SOS for the 3 controller banks: csoft (3,Ns,6), chard (3,Nh,6), e2l (2,2)."""
    ctrls = [Controller(data, phys, plot_dir=None, controller_name=CONTROLLER_CHOICES[c]) for c in (0, 1, 2)]
    csoft = np.ascontiguousarray(np.stack([np.asarray(c.global_control_sos[0], np.float64) for c in ctrls]))
    chard = np.ascontiguousarray(np.stack([np.asarray(c.global_control_sos[1], np.float64) for c in ctrls]))
    ce = np.ascontiguousarray(ctrls[0].eigen2local, np.float64)
    return csoft, chard, ce


def run_fast(regime, controller, dur, seed, fs=256, n_hard=3e-14, n_soft=1e-13, sens_scale=None):
    """Single fixed (regime, controller) fast run; returns pitch (N,2), control (N,2)."""
    plant, sensors, ssc, data, phys = sd.setup(regime, dur, seed)
    if sens_scale is not None:
        n_hard *= sens_scale; n_soft *= sens_scale
    K = extract_plant_consts(plant, sensors, ssc, fs)
    csoft, chard, ce = build_ctrl_stack(data, phys)
    N = plant.tst_noise_t.shape[1]
    sus_itm = np.ascontiguousarray(plant.tst_noise_t[0]); sus_etm = np.ascontiguousarray(plant.tst_noise_t[1])
    power = np.ascontiguousarray(plant.input_power)
    rng = np.random.RandomState(seed)
    scale = (fs/2.0)**0.5
    sn_s = rng.normal(0, scale*n_soft, N); sn_h = rng.normal(0, scale*n_hard, N)
    st = new_state()
    pitch, ctl, _ = run_kernel(
        N-1, 0, sus_itm, sus_etm, power, sn_s, sn_h,
        K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
        K['hp_sos'], np.zeros((K['hp_sos'].shape[0], 2)),
        K['rad_sos'], np.zeros((K['rad_sos'].shape[0], 2)), np.zeros((K['rad_sos'].shape[0], 2)),
        K['act_sos'], np.zeros((K['act_sos'].shape[0], 2)), np.zeros((K['act_sos'].shape[0], 2)),
        csoft, np.zeros((3, csoft.shape[1], 2)), chard, np.zeros((3, chard.shape[1], 2)), ce, controller, controller, 0,
        K['loc2eig'],
        K['ss_sos'], np.zeros((K['ss_sos'].shape[0], 2)), np.zeros((K['ss_sos'].shape[0], 2)), K['ss_e2l'],
        K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], st)
    return pitch, ctl


def run_policy_fast(policy, noises, W, sens_t, seed, T_hold, feat_names, w_weights,
                    fs=256, reward_mode='linear'):
    """Fast bandit closed loop: kernel runs the active controller; inactive controllers are
    kept warm (parallel banks) by advancing their state on the window's readout. 300s windows."""
    N = W.shape[1]
    sus_itm = np.ascontiguousarray(W[0]*noises[0][0][0] + W[1]*noises[1][0][0] + W[2]*noises[2][0][0])
    sus_etm = np.ascontiguousarray(W[0]*noises[0][0][1] + W[1]*noises[1][0][1] + W[2]*noises[2][0][1])
    power = np.ascontiguousarray(W[0]*noises[0][1] + W[1]*noises[1][1] + W[2]*noises[2][1])
    plant, sensors, ssc, data, phys = sd.setup(0, int(round(N/fs)), seed)
    K = extract_plant_consts(plant, sensors, ssc, fs)
    csoft, chard, ce = build_ctrl_stack(data, phys)
    csoft_zi = np.zeros((3, csoft.shape[1], 2)); chard_zi = np.zeros((3, chard.shape[1], 2))
    rng = np.random.RandomState(seed); scl = (fs/2.0)**0.5
    sn_s = rng.normal(0, scl*1e-13, N)*sens_t
    sn_h = rng.normal(0, scl*3e-14, N)*sens_t
    # persistent plant/SS state + SOS delay lines
    st = new_state()
    hp_zi = np.zeros((K['hp_sos'].shape[0], 2))
    rz0 = np.zeros((K['rad_sos'].shape[0], 2)); rz1 = np.zeros((K['rad_sos'].shape[0], 2))
    az0 = np.zeros((K['act_sos'].shape[0], 2)); az1 = np.zeros((K['act_sos'].shape[0], 2))
    szs = np.zeros((K['ss_sos'].shape[0], 2)); szh = np.zeros((K['ss_sos'].shape[0], 2))
    hold_n = int(T_hold*fs); ramp_n = int(2.0*fs)   # 2s bumpless cross-fade on each switch
    rew = StreamReward(feat_names, w_weights)        # continuous (stateful) reward filtering
    active = policy.choose(0, W[:, 0]); prev = active; ramp_this = 0
    rewards, arms_log, ctx_log = [], [], []
    ti = 0
    while ti < N-1:
        nb = min(hold_n, N-1-ti)
        # all 3 controller banks run in the kernel (parallel); active is applied, cross-faded
        # from prev over ramp_this samples after a switch (bumpless handover)
        pitch, ctl, read = run_kernel(
            nb, ti, sus_itm, sus_etm, power, sn_s[ti:ti+nb], sn_h[ti:ti+nb],
            K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
            K['hp_sos'], hp_zi, K['rad_sos'], rz0, rz1, K['act_sos'], az0, az1,
            csoft, csoft_zi, chard, chard_zi, ce, active, prev, ramp_this,
            K['loc2eig'], K['ss_sos'], szs, szh, K['ss_e2l'],
            K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], st)
        sc = rew.score(pitch, ctl)          # filters carry state across windows
        mean_raw = sc.mean() if len(sc) else 167.5
        reward = float(np.clip((mean_raw-148.0)/22.0, 0, 1)) if reward_mode == 'linear' else sd.logistic(mean_raw)
        policy.update(active, reward)
        rewards.append(reward); arms_log.append(active); ctx_log.append(W[:, ti].copy())
        new = policy.choose(len(rewards), W[:, min(ti+nb, N-1)])
        if new != active:
            prev = active; active = new; ramp_this = ramp_n
        else:
            prev = active; ramp_this = 0
        ti += nb
    return dict(rewards=np.array(rewards), arms=np.array(arms_log), ctx=np.array(ctx_log))


def validate():
    import time, bandit_rewards
    fn, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    SENS = {0: 120.0, 1: 12.0, 2: 0.25}
    print("Validating fast kernel vs sec.5 (200s/case)...")
    for R, Cdiag in [(0, 0), (1, 1), (2, 2)]:
        t0 = time.time()
        pitch, ctl = run_fast(R, Cdiag, 200, seed=1, sens_scale=SENS[R])
        dt = time.time() - t0
        _, sc = sd.block_scores(pitch, ctl, fn, w)
        print(f"  R{R} C{Cdiag}: mean_raw={sc.mean():.2f}  (fast {dt:.1f}s, {len(pitch)/dt/1000:.0f}k steps/s)", flush=True)


if __name__ == "__main__":
    validate()
