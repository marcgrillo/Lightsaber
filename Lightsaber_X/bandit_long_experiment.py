"""
Long-horizon online controller-selection experiment (months-scale).

Streams the pre-generated, disk-backed environment noise (bandit_noise_cache.py)
through the validated fast closed-loop kernel, once per policy, carrying full
plant + reward-filter state across the whole run.  Compares:

    TAR-UCB            -- Transition-Aware Recurrent UCB (the method under study)
    D-UCB             -- Discounted UCB           }  the two common bandit baselines
    Thompson (D-TS)   -- Discounted Thompson sampling
    Rule-based        -- naive operator practice: reward-only alarm + periodic re-check
                        (NO access to the true regime/context, see bandit_policies.RuleBased)
    Fixed-C0/C1/C2    -- each fixed controller
    Oracle            -- plays argmax_c r_c(W) from the calibrated reward table

Design choices that make months-scale feasible and faithful:
  * The environment (regime schedule W(t), injected noise, input power) is common
    to all policies; the expensive noise is generated ONCE and streamed from disk.
  * Sensor (white) noise is regenerated deterministically per window, identical
    across policies (common random numbers).
  * Controllers run as parallel banks with a 2 s bumpless ramp on each switch
    (the realistic near-free handover; see switching_dynamics / paper Sec. 6).
  * Reward is the continuous multi-band logistic score (state carried across
    windows), calibrated by bandit_calibrate.py.

Results are checkpointed PER POLICY to <out>/policies/<name>.npz, so a rerun skips
finished policies (and an interrupted run only loses the in-progress policy).

    # 1) generate the noise once (see bandit_noise_cache.py), then:
    python bandit_long_experiment.py --cache bandit_runs/cache_6mo --hold 20
"""
import os, sys, time, json, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bandit_rewards
import switching_dynamics as sd
import fast_engine as fe
import bandit_policies as bp
from reward_stream import StreamReward
from bandit_noise_cache import load_cache
from bandit_calibrate import calibrate

FS = 256
SENSOR_SEED = 20260701      # fixed -> identical sensor noise across all policies
# CG-ICLB-TS commit-baseline source: 0 = theta_cert (faithful to draft), 1 = theta_sh
# (clean balanced-block shape; ablation for the boundary-pollution false-alarm diagnosis).
CGTS_BASELINE_SHAPE = os.environ.get("CGTS_BASELINE_SHAPE", "0") == "1"


def shape_separation(oracle_table):
    """Min pairwise sup-norm distance between CENTERED per-regime reward vectors
    (the shape separation Delta_sh that governs LI-TAR-UCB recognition)."""
    T = np.asarray(oracle_table, float)              # (arm, regime)
    shapes = [T[:, z] - T[:, z].mean() for z in range(T.shape[1])]
    d = [float(np.max(np.abs(shapes[a] - shapes[b])))
         for a in range(len(shapes)) for b in range(a + 1, len(shapes))]
    return min(d) if d else 0.0


def make_policies(horizon_steps, oracle_table, sigma):
    """TAR-UCB and its level-invariant successor LI-TAR-UCB + the two common baselines
    (D-UCB, Thompson) + reference policies. Detection/stability thresholds scale with the
    measured per-window reward noise sigma; LI-TAR-UCB shape thresholds also respect the
    measured shape separation Delta_sh (bandit_calibrate)."""
    s = max(sigma, 1e-3)
    dsh = shape_separation(oracle_table)             # ~0.05-0.12 here
    r_sh = float(np.clip(0.45 * dsh, 0.6 * s, 1.2 * s))   # in the theory band, floored by noise
    rb_floor = bp.rule_based_floor(oracle_table)     # naive reward-only alarm level (see bandit_policies.RuleBased)
    # CG-ICLB radius constant: anchor the BLOCK-level shape resolution b(m) to the same
    # calibrated band as LI-TAR-UCB's r_sh, so matching/certification discriminate the
    # measured shape separation Delta_sh rather than the very conservative literal
    # 8KT/delta constant (the draft notes the exact constant is unimportant). The sqrt(1/n)
    # shrinkage is preserved, so a well-visited regime still certifies its best arm.
    cg_m = 6; cg_delta = 0.1
    cg_L = 2.0 * np.log(8.0 * 3 * max(int(horizon_steps), 2) / cg_delta)
    cg_crad = float(r_sh / (2.0 * s * np.sqrt(cg_L / cg_m)))
    return {
        "TAR-UCB":    bp.TARUCB(3, sigma=s, delta=0.1, m0=3, nu=0.6*s, h=7.0*s,
                                m_batch=1, r_stab=1.2*s, q_stab=2, m_cls=4, r_cls=2.0*s,
                                add_diag=True, seed=0),
        "LI-TAR-UCB": bp.LITARUCB(3, sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                                  m_batch=1, r_stab=1.0*s, q_stab=1, d_exit=1.5*s, r_coll=r_sh,
                                  m_cls=6, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
                                  level_window=30,
                                  # scheduled probing shape detector (draft Sec. 5.3), tuned
                                  # for hold=100s (bandit_tune_litarucb_v2.py sweep on
                                  # cache_3d_r1long): probe every 90 windows (2.5h), 3
                                  # samples/arm/block (~15min to resolve once triggered).
                                  # Deliberately halves m_blk vs the hold=20s-era default
                                  # (6->3) to cut per-probe diagnosis latency, at the cost of
                                  # a ~5x higher clean false-alarm rate (still <1/2 days,
                                  # measured on a stationary synthetic control) -- the
                                  # user-requested "faster diagnosis for a bit more
                                  # false-positive risk" trade. q_det=2 keeps a persistence
                                  # check against single-block noise blips.
                                  #
                                  # q_stab/m_cls (bandit_tune_diag.py sweep on
                                  # cache_1w_r1long): the TRANSITION-exit + DIAGNOSTIC
                                  # classification budget is (m_batch*q_stab + m_cls)*K*hold
                                  # seconds; at the old defaults (q_stab=2, m_cls=8) that's
                                  # 3000s, longer than every observed R1 episode (max 2666s
                                  # at spike_hold=1500), so R1 never got its own library
                                  # entry (J stuck at 2). q_stab=1, m_cls=6 (budget 2100s) is
                                  # a genuine win, not just a speed/risk trade: 1-week regret
                                  # improved 138.6->115.3, library correctly settles at J=3,
                                  # and switches dropped 2574->1810. Shrinking m_cls further
                                  # (<=4) over-fragments the library (J=4) and actively
                                  # degrades R1's arm choice (flips to preferring the wrong
                                  # controller) -- q_stab=1/m_cls=6 is the sweet spot, not an
                                  # arbitrary point on a monotone tradeoff curve.
                                  L_probe=90, m_blk=3, q_det=2, seed=0,
                                  # guarantee-preserving extensions (bandit_audit_litarucb.py
                                  # A/B on both 1-week caches): the one-sided drop-CUSUM
                                  # cannot see the END of a deep transition collapse (a
                                  # reward RISE), and the logistic saturates every arm to
                                  # ~0 mid-collapse ("blackout"), flattening the shape so
                                  # TRANSITION would exit and DIAGNOSTIC would commit a
                                  # garbage flat regime. h_up adds the symmetric rise
                                  # CUSUM; eps_floor makes blackout batches uninformative
                                  # (never certify stability / never reach the library).
                                  # 1-week regret: 115.3->96.0 (r1long), 142.9->84.0
                                  # (r1freq); dark settles 3->0 / 3->1; every regime's
                                  # near-optimal play rate >= 0.88 / 0.97.
                                  h_up=5.0*s,
                                  eps_floor=0.5*float(oracle_table.max(axis=0).min())),
        # Confidence-Gated Identify-and-Commit Library Bandit (bandit_policies.CGICLB,
        # paper/cg_iclb_detailed_algorithm.tex). Balanced blocks identify+certify; a
        # certified regime is played deterministically (no UCB/TS in stable operation) and
        # reused on recurrence via the shape library. Radius anchored to r_sh (c_rad above);
        # commit-drift CUSUM on |Y - (ell+theta_a)| with nu>E|noise|=0.8s to avoid noise
        # false-alarms, h in the LI-TAR-UCB band. gamma=0: commit whenever an arm is
        # statistically separated (R2's arm2~arm1 and the flat R1 then never certify -> the
        # method keeps exploring there, which is near-free since the arms are nearly tied).
        "CG-ICLB":    bp.CGICLB(3, sigma=s, delta=cg_delta, T=int(horizon_steps), m=cg_m,
                                c_rad=cg_crad, gamma=0.0, nu=1.3*s, h=8.0*s, seed=0),
        # CG-ICLB-TS (bandit_policies.CGICLBTS, paper/cg_iclb_ts_certification_algorithm.tex):
        # same identify+commit skeleton but CERTIFY is Thompson-guided with eps_opt-optimal
        # stopping. TS concentrates certification pulls on plausibly-best arms (near-optimal
        # reward while certifying), and eps_opt lets the flat R1 (gap 0.014) and R2's arm2~arm1
        # (gap 0.005) certify a practically-optimal arm instead of stalling in balanced
        # exploration -- the two things that hurt plain CG-ICLB on the frequent-switch cache.
        # Shape radius keeps c_rad (matching); cert radius uses the literal bound (c_rad_cert=1)
        # so commitment needs genuine per-arm evidence; eps_opt in the calibrated tie band.
        "CG-ICLB-TS": bp.CGICLBTS(3, sigma=s, delta=cg_delta, T=int(horizon_steps), m=cg_m,
                                  c_rad=cg_crad, c_rad_cert=1.0,
                                  eps_opt=float(np.clip(0.5*dsh, 0.6*s, 1.2*s)),
                                  nu=1.3*s, h=8.0*s, seed=0,
                                  baseline_shape=CGTS_BASELINE_SHAPE),
        # CG-ICLB-TT: same as CG-ICLB-TS but certification uses TOP-TWO Thompson sampling
        # (tt_beta=0.5). Plain TS starves the suboptimal arms the eps_opt stop must bound, so
        # certification stalls (often past the episode) -> commit-into-next-regime churn; top-two
        # guarantees the runner-up is pulled, so the stop fires fast. Faster/cleaner certification
        # also cuts boundary mis-IDs (hence theta_cert baseline pollution) -- aims to fix the
        # CG-ICLB-TS long-episode regression WITHOUT the baseline_shape frequent-switch trade-off.
        "CG-ICLB-TT": bp.CGICLBTS(3, sigma=s, delta=cg_delta, T=int(horizon_steps), m=cg_m,
                                  c_rad=cg_crad, c_rad_cert=1.0,
                                  eps_opt=float(np.clip(0.5*dsh, 0.6*s, 1.2*s)),
                                  nu=1.3*s, h=8.0*s, seed=0,
                                  baseline_shape=CGTS_BASELINE_SHAPE, tt_beta=0.5),
        # CG-ICLB-R (bandit_policies.CGICLBR): removes the CERTIFY/COMMIT split. Each regime
        # stores its accumulated per-arm TS evidence; on recurrence the posterior is RESUMED,
        # not re-certified, and stable operation is plain Thompson sampling (commit-like
        # determinism emerges as the posterior concentrates -- no eps_opt, no c_rad_cert, no
        # top-two). Drift uses a PER-ARM |X-base| CUSUM vs a fixed entry snapshot (RUN plays a
        # TS mixture, not one arm). Same shape-matching radius (c_rad) and same drift
        # calibration (nu=1.3s, h=8s) as CG-ICLB. Tests whether cross-visit accumulation beats
        # the deterministic-commit tail on long episodes; expected to be strongest under
        # frequent switching (resume vs re-pay certification).
        "CG-ICLB-R":  bp.CGICLBR(3, sigma=s, delta=cg_delta, T=int(horizon_steps), m=cg_m,
                                 c_rad=cg_crad, nu=1.3*s, h=8.0*s, seed=0),
        # CG-ICLB-R2: CG-ICLB-R + FAST RE-ACQUIRE. R re-pays the balanced-block round-robin ID
        # every short episode (28.6% of steps on r1freq, which is why it lost to CG-ICLB-TT
        # there). Strict unique-match waits for the aggregate radius to shrink until every other
        # regime leaves its band; instead, since a post-alarm env almost always RETURNED to a
        # known regime, resume the NEAREST library regime as soon as it is a valid match AND
        # beats the runner-up by reacq_margin=0.5*dsh (a shape-separation margin guarding a wrong
        # resume). Often fires after one block. New-regime creation stays strict. Tests whether
        # cutting the ID tax closes R's frequent-switch gap to TT without hurting long episodes.
        "CG-ICLB-R2": bp.CGICLBR(3, sigma=s, delta=cg_delta, T=int(horizon_steps), m=cg_m,
                                 c_rad=cg_crad, nu=1.3*s, h=8.0*s,
                                 fast_reacquire=True, reacq_margin=0.5*dsh, seed=0),
        # RA-TS (bandit_policies.RATS): the distilled algorithm -- JUST Thompson sampling + a
        # drift detector + a warm-start library, with regime identification EMERGENT from TS's
        # own exploration (no balanced-block ID phase, no CERT/COMMIT, no eps_opt/top-two, no
        # level estimation). ACQUIRE resets to a diffuse prior so plain TS spreads across arms;
        # once every arm is covered it matches the CENTERED shape (level-invariant) against the
        # library and TRACKs the regime's resumed posterior, with the same snapshot-baseline
        # per-arm CUSUM as R2. mu0 = mean achievable reward; tau0 diffuse; match tol/margin tied
        # to the shape separation dsh; CUSUM calib (nu=1.3s,h=8s) shared with the CG-ICLB family.
        "RA-TS":      bp.RATS(3, sigma=s, mu0=float(oracle_table.mean()), tau0=0.5,
                              n_min=3, n_new=8, match_tol=0.8*dsh, match_margin=0.4*dsh,
                              nu=1.3*s, h=8.0*s, explore_floor=False, seed=0),
        # RA-TS-F: same, with the bounded coverage probe (explore_floor). Pure TS (RA-TS above)
        # provably STALLS in ACQUIRE -- it will not sample the loser arms identification needs;
        # the floor forces the least-sampled arm until each has n_min samples, then hands back to
        # TS. This is the minimal exploration needed for regime ID; <= n_min*K pulls per re-acquire.
        "RA-TS-F":    bp.RATS(3, sigma=s, mu0=float(oracle_table.mean()), tau0=0.5,
                              n_min=3, n_new=8, match_tol=0.8*dsh, match_margin=0.4*dsh,
                              nu=1.3*s, h=8.0*s, explore_floor=True, seed=0),
        # RA-TS-FQ: RA-TS-F + DELAYED COMMIT (commit_delay=q), the purity device from
        # paper/ra_ts_paper_skeleton.tex. Every TRACK observation sits in a q-round FIFO queue
        # and is folded into the persistent library stats only if no alarm fires in that
        # window; on alarm the whole queue is discarded, so post-switch samples collected
        # during the detection delay can never contaminate the old regime's entry. This is a
        # pure A/B against RA-TS-F (q=0): it trades a q-sample-stale Thompson posterior
        # (delayed feedback) for a guaranteed-clean library. q=10 is a round-number pick in the
        # observed detection-delay range (empirical median delay ~2-10 decisions on these
        # caches; see bandit-rats memory).
        "RA-TS-FQ":   bp.RATS(3, sigma=s, mu0=float(oracle_table.mean()), tau0=0.5,
                              n_min=3, n_new=8, match_tol=0.8*dsh, match_margin=0.4*dsh,
                              nu=1.3*s, h=8.0*s, explore_floor=True, commit_delay=10, seed=0),
        # RA-TS-FR: RA-TS-F + ROLLBACK. Commits every TRACK sample immediately (posterior never
        # stale, unlike FQ's queue), but on an alarm retroactively un-commits the contiguous run
        # of samples since the alarming arm's CUSUM last reset to zero -- the change-point segment
        # carrying the post-switch dirty tail. Removes the contamination that forces the tolerance
        # eps_tol without the delayed-feedback cost that made FQ regress. Tests whether the
        # theoretical tolerance-shrinking shows up as cleaner library + equal-or-better regret.
        "RA-TS-FR":   bp.RATS(3, sigma=s, mu0=float(oracle_table.mean()), tau0=0.5,
                              n_min=3, n_new=8, match_tol=0.8*dsh, match_margin=0.4*dsh,
                              nu=1.3*s, h=8.0*s, explore_floor=True, rollback=True, seed=0),
        "D-UCB":      bp.DiscountedUCB(3, gamma=0.999, xi=0.5),
        "Thompson":   bp.DiscountedTS(3, gamma=0.999, sigma=s, mu0=1.0),
        # naive operator practice (bandit_policies.RuleBased): NO ctx/regime access, only
        # the observed reward stream. floor=rb_floor separates a genuinely low-reward
        # regime (R1 here) from a normal one regardless of arm; window counts below are
        # tuned for hold=100s (q_bad=3 -> 300s alarm persistence, m_probe=5 -> 1500s
        # round-robin probe across K=3 arms, t_periodic=180 -> 5h routine re-check,
        # dwell=300 -> 8.3h cooldown). Re-tune the window counts if hold changes.
        "Rule-based": bp.RuleBased(3, floor=rb_floor, ewma_beta=0.15, q_bad=3, m_probe=5,
                                    dwell=300, t_periodic=180, seed=7),
        "Fixed-C0":   bp.FixedArm(0),
        "Fixed-C1":   bp.FixedArm(1),
        "Fixed-C2":   bp.FixedArm(2),
        "Oracle":     bp.Oracle(oracle_table),
    }


def run_policy_stream(policy, cache, sens_of_sample, W_of_sample, feat_names, w,
                      hold, lo, hi, t_ramp=2.0, read_block=4096, reward_mode='norm',
                      progress_every=50000, name=""):
    """Run one policy over the whole cached horizon, streaming from disk.

    sens_of_sample(a0,a1) -> sens_t[a0:a1]; W_of_sample(a0,a1) -> (3, n) weights."""
    N = int(cache['N']); fs = FS
    sus_itm_mm = cache['sus_itm']; sus_etm_mm = cache['sus_etm']; power_mm = cache['power']
    # plant constants + controller banks (physics only; independent of the noise)
    plant, sensors, ssc, data, phys = sd.setup(0, 8, seed=int(cache['seed']))
    K = fe.extract_plant_consts(plant, sensors, ssc, fs)
    csoft, chard, ce = fe.build_ctrl_stack(data, phys)
    csoft_zi = np.zeros((3, csoft.shape[1], 2)); chard_zi = np.zeros((3, chard.shape[1], 2))
    st = fe.new_state()
    hp_zi = np.zeros((K['hp_sos'].shape[0], 2))
    rz0 = np.zeros((K['rad_sos'].shape[0], 2)); rz1 = np.zeros((K['rad_sos'].shape[0], 2))
    az0 = np.zeros((K['act_sos'].shape[0], 2)); az1 = np.zeros((K['act_sos'].shape[0], 2))
    szs = np.zeros((K['ss_sos'].shape[0], 2)); szh = np.zeros((K['ss_sos'].shape[0], 2))
    hold_n = int(hold * fs); ramp_n = int(t_ramp * fs)
    rew = StreamReward(feat_names, w)
    srng = np.random.RandomState(SENSOR_SEED); scl = (fs / 2.0) ** 0.5
    W0 = W_of_sample(0, 1)[:, 0]
    active = policy.choose(0, W0); prev = active; ramp_this = 0
    rewards, arms_log, ctx_log, raw_log = [], [], [], []
    n_windows = (N - 1) // hold_n                       # drop the (<hold_n) tail, like the reference
    # RAM read-chunk aligned to a whole number of decision windows (else a window would
    # straddle two reads -> a spurious short window, breaking the continuous state carry)
    read_n = max(1, int(read_block * fs) // hold_n) * hold_n
    t0 = time.time(); w0 = 0
    while w0 < n_windows:
        wchunk = min(read_n // hold_n, n_windows - w0)
        a_lo = w0 * hold_n; a_hi = a_lo + wchunk * hold_n
        itm = np.ascontiguousarray(sus_itm_mm[a_lo:a_hi], np.float64)
        etm = np.ascontiguousarray(sus_etm_mm[a_lo:a_hi], np.float64)
        pw = np.ascontiguousarray(power_mm[a_lo:a_hi], np.float64)
        for j in range(wchunk):
            a0 = a_lo + j * hold_n; off = j * hold_n
            sens_win = sens_of_sample(a0, a0 + hold_n)
            sn_s = srng.normal(0, scl * 1e-13, hold_n) * sens_win
            sn_h = srng.normal(0, scl * 3e-14, hold_n) * sens_win
            pitch, ctl, _ = fe.run_kernel(
                hold_n, 0, itm[off:off+hold_n], etm[off:off+hold_n], pw[off:off+hold_n], sn_s, sn_h,
                K['bs_matrix'], K['rho_itm'], K['t_itm'], K['lam'], K['dc_offset'], K['p_dc'],
                K['hp_sos'], hp_zi, K['rad_sos'], rz0, rz1, K['act_sos'], az0, az1,
                csoft, csoft_zi, chard, chard_zi, ce, active, prev, ramp_this,
                K['loc2eig'], K['ss_sos'], szs, szh, K['ss_e2l'],
                K['dydth_soft'], K['dydth_hard'], K['kk_lp'], K['p_const'], st)
            sc = rew.score(pitch, ctl)
            mean_raw = sc.mean() if len(sc) else 167.5
            if reward_mode == 'norm':
                reward = float(np.clip((mean_raw - lo) / (hi - lo), 0.0, 1.0))
            elif reward_mode == 'logistic':
                reward = sd.logistic(mean_raw)
            else:
                reward = float(np.clip((mean_raw - 148.0)/22.0, 0, 1))
            policy.update(active, reward)
            Wdec = W_of_sample(a0, a0 + 1)[:, 0]
            rewards.append(reward); arms_log.append(active); ctx_log.append(Wdec); raw_log.append(mean_raw)
            new = policy.choose(len(rewards), Wdec)
            if new != active:
                prev = active; active = new; ramp_this = ramp_n
            else:
                prev = active; ramp_this = 0
            if progress_every and (w0 + j) % progress_every == 0 and (w0 + j) > 0:
                el = time.time() - t0
                print(f"    [{name}] {a0/fs/86400:.2f}d / {N/fs/86400:.2f}d  "
                      f"({el:.0f}s, {a0/el/1e3:.0f}k samp/s)", flush=True)
        w0 += wchunk
    out = dict(rewards=np.array(rewards), arms=np.array(arms_log, np.int8),
               ctx=np.array(ctx_log, np.float32), raw=np.array(raw_log, np.float32),
               wall_s=time.time() - t0)
    # structural diagnostics (TAR-UCB, LI-TAR-UCB, CG-ICLB[-TS] share the J/mode/Z interface)
    if isinstance(policy, (bp.TARUCB, bp.LITARUCB, bp.CGICLB, bp.CGICLBTS, bp.CGICLBR, bp.RATS)):
        out['tar_J'] = np.array(policy.J_hist, np.int16)
        out['tar_mode'] = np.array(policy.mode_hist, np.int8)
        out['tar_Z'] = np.array(policy.Z_hist, np.int16)
    if isinstance(policy, bp.RATS) and len(policy.N):     # final library centered shapes + counts
        out['rats_shape'] = np.array([policy._center(policy.S[g]/np.maximum(policy.N[g], 1e-9))
                                      for g in range(len(policy.N))], np.float32)
        out['rats_count'] = np.array([policy.N[g].sum() for g in range(len(policy.N))], np.float64)
        out['rats_surp'] = np.array(policy.surp_hist, np.float32)   # per-step detector surprise |Y-b|
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="noise cache dir (from bandit_noise_cache.py)")
    ap.add_argument("--out", default=None, help="output dir (default <cache>/experiment)")
    ap.add_argument("--hold", type=int, default=100, help="decision interval T_hold [s]")
    ap.add_argument("--reward", default="logistic", choices=["logistic", "norm", "linear"])
    ap.add_argument("--read-block", type=float, default=4096, help="RAM read-chunk [s]")
    ap.add_argument("--only", default=None, help="comma-list of policies to (re)run")
    ap.add_argument("--force", action="store_true", help="recompute even if checkpoint exists")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    N = int(cache['N']); horizon = float(cache['horizon'])
    outdir = args.out or os.path.join(args.cache, "experiment")
    poldir = os.path.join(outdir, "policies"); os.makedirs(poldir, exist_ok=True)
    print(f"[exp] cache={args.cache}  horizon={horizon:.0f}s ({horizon/86400:.1f}d)  N={N:,}  hold={args.hold}s")

    calib = calibrate(hold=args.hold, reward_mode=args.reward)
    oracle_table = np.asarray(calib['oracle_table'], float)      # (arm, regime)
    sigma = float(calib['sigma_hat'])
    lo, hi = float(calib['lo']), float(calib['hi'])
    print(f"[exp] reward={args.reward}  sigma={sigma:.3f}"
          + (f"  lo={lo:.2f} hi={hi:.2f}" if args.reward == 'norm' else ""))
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")

    # per-second schedules -> fast sample-accurate accessors
    Wsec = np.asarray(cache['Wsec']); tsec = np.asarray(cache['tsec']); sens_sec = np.asarray(cache['sens_sec'])
    def W_of_sample(a0, a1):
        ts = np.arange(a0, a1) / FS
        return np.vstack([np.interp(ts, tsec, Wsec[i]) for i in range(3)])
    def sens_of_sample(a0, a1):
        ts = np.arange(a0, a1) / FS
        return np.interp(ts, tsec, sens_sec)

    horizon_steps = N // int(args.hold * FS)
    only = set(args.only.split(",")) if args.only else None
    results = {}
    for name, pol in make_policies(horizon_steps, oracle_table, sigma).items():
        if only and name not in only:
            ckpt = os.path.join(poldir, f"{name}.npz")
            if os.path.exists(ckpt):
                results[name] = dict(np.load(ckpt, allow_pickle=True))
            continue
        ckpt = os.path.join(poldir, f"{name}.npz")
        if os.path.exists(ckpt) and not args.force:
            results[name] = dict(np.load(ckpt, allow_pickle=True))
            print(f"[exp] {name:11s}: loaded checkpoint "
                  f"(cum={results[name]['rewards'].sum():.1f})")
            continue
        np.random.seed(1234)                       # reproducible exploration (D-TS uses global RNG)
        print(f"[exp] {name:11s}: running...", flush=True)
        h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                              args.hold, lo, hi, read_block=args.read_block,
                              reward_mode=args.reward, name=name)
        np.savez_compressed(ckpt, **h)
        results[name] = h
        sw = int((np.diff(h['arms']) != 0).sum())
        print(f"[exp] {name:11s}: cum={h['rewards'].sum():.1f}  mean={h['rewards'].mean():.4f}  "
              f"switches={sw}  ({h['wall_s']:.0f}s)", flush=True)

    write_outputs(outdir, results, cache, oracle_table, args.hold, sigma)
    print(f"\n[exp] saved to {outdir}")


def write_outputs(outdir, results, cache, oracle_table, hold, sigma):
    N = int(cache['N']); fs = FS
    hold_n = int(hold * fs)
    Wsec = np.asarray(cache['Wsec']); tsec = np.asarray(cache['tsec'])
    ndec = min(len(v['rewards']) for v in results.values())
    # int64: ndec*hold_n exceeds int32 beyond ~97 days (np.arange defaults to int32 on Windows)
    dec_t = (np.arange(ndec, dtype=np.int64) * hold_n) / fs
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])     # (3, ndec)
    exp_r = oracle_table @ Wdec                                              # (arm, ndec) expected reward
    best_r = exp_r.max(axis=0)                                               # per-decision best achievable
    opt_arm = np.argmax(exp_r, axis=0)                                       # (ndec,)
    # NEAR-optimal: played arm's expected-reward gap to the best <= eps_near. Because two
    # controllers are near-tied in some regimes (e.g. C1 vs C2 in R2, gap ~0.005), a strict
    # optimal-arm count penalises harmless choices; the near-optimal fraction counts a window
    # as a "hit" whenever the per-window regret is below half the reward noise, so it tracks
    # regret rather than contradicting it. eps_near = 0.5*sigma (~0.021 here) sits well below
    # every costly confusion (gaps >= 0.08) and above the genuine near-ties.
    eps_near = 0.5 * float(sigma)

    def near_opt_frac(a):
        return float(np.mean(best_r - exp_r[a, np.arange(len(a))] <= eps_near))
    sev = 1.0 * Wdec[1] + 2.0 * Wdec[2]

    # summary
    oracle_cum = results['Oracle']['rewards'][:ndec].sum() if 'Oracle' in results else np.nan
    with open(os.path.join(outdir, "summary.csv"), "w") as f:
        f.write("policy,cum_reward,mean_reward,regret_vs_oracle,switches,frac_optimal,frac_near_optimal\n")
        for name, v in results.items():
            r = v['rewards'][:ndec]; a = v['arms'][:ndec]
            frac = float(np.mean(a == opt_arm)); fnear = near_opt_frac(a)
            sw = int((np.diff(v['arms']) != 0).sum())
            f.write(f"{name},{r.sum():.3f},{r.mean():.4f},{oracle_cum - r.sum():.3f},{sw},{frac:.3f},{fnear:.3f}\n")
    # console table
    print(f"\n[near-optimal threshold eps = {eps_near:.4f} = 0.5*sigma]")
    print(f"{'policy':12s} {'cum_reward':>11s} {'regret':>9s} {'frac_opt':>9s} {'near_opt':>9s} {'switches':>9s}")
    order = [n for n in ["Oracle", "RA-TS-FR", "RA-TS-FQ", "RA-TS-F", "RA-TS", "CG-ICLB-R2",
                         "CG-ICLB-R", "CG-ICLB-TT", "CG-ICLB-TS", "CG-ICLB", "LI-TAR-UCB",
                         "TAR-UCB", "Thompson", "D-UCB", "Rule-based", "Fixed-C0", "Fixed-C1",
                         "Fixed-C2"]
             if n in results]
    for name in order:
        v = results[name]; r = v['rewards'][:ndec]; a = v['arms'][:ndec]
        print(f"{name:12s} {r.sum():11.1f} {oracle_cum - r.sum():9.1f} "
              f"{np.mean(a==opt_arm):9.3f} {near_opt_frac(a):9.3f} {int((np.diff(v['arms'])!=0).sum()):9d}")

    # fig 1: cumulative reward + regret vs oracle
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name in order:
        v = results[name]; r = v['rewards'][:ndec]
        ax[0].plot(dec_t/86400, np.cumsum(r), label=name,
                   lw=(2.4 if name in ("Oracle", "LI-TAR-UCB") else 1.3),
                   ls=('--' if name.startswith("Fixed") else '-'))
    ax[0].set_xlabel("time [days]"); ax[0].set_ylabel("cumulative reward")
    ax[0].set_title("Cumulative reward"); ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)
    if 'Oracle' in results:
        oc = np.cumsum(results['Oracle']['rewards'][:ndec])
        for name in order:
            if name == 'Oracle':
                continue
            v = results[name]; r = np.cumsum(v['rewards'][:ndec])
            ax[1].plot(dec_t/86400, oc - r, label=name,
                       lw=(2.2 if name in ("LI-TAR-UCB", "TAR-UCB") else 1.2),
                       ls=('--' if name.startswith("Fixed") else '-'))
        ax[1].set_xlabel("time [days]"); ax[1].set_ylabel("cumulative regret vs Oracle")
        ax[1].set_title("Regret vs Oracle"); ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_cumreward_regret.png"), dpi=130); plt.close()

    # fig 2: severity + arm timelines
    show = [n for n in ["Oracle", "RA-TS-FR", "RA-TS-F", "CG-ICLB-TT", "LI-TAR-UCB"] if n in results]
    fig, axes = plt.subplots(len(show)+1, 1, figsize=(13, 1.6*(len(show)+1)), sharex=True)
    axes[0].plot(dec_t/86400, sev, 'k', lw=0.8); axes[0].set_ylabel("severity")
    axes[0].set_title("regime severity and controller choices"); axes[0].grid(alpha=0.3)
    for ax, name in zip(axes[1:], show):
        a = results[name]['arms'][:ndec]
        ax.step(dec_t/86400, a, where='post', lw=0.7)
        ax.set_ylabel(name); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["C0", "C1", "C2"]); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [days]")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_timelines.png"), dpi=130); plt.close()

    # fig 3: library/mode diagnostics (prefer LI-TAR-UCB)
    diag_name = 'LI-TAR-UCB' if ('LI-TAR-UCB' in results and 'tar_J' in results['LI-TAR-UCB']) \
        else ('TAR-UCB' if 'TAR-UCB' in results and 'tar_J' in results['TAR-UCB'] else None)
    if diag_name:
        v = results[diag_name]
        fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        td = (np.arange(len(v['tar_J']), dtype=np.int64) * hold_n) / fs / 86400
        ax[0].plot(td, v['tar_J'], 'tab:red'); ax[0].set_ylabel("library size J"); ax[0].grid(alpha=0.3)
        ax[0].set_title(f"{diag_name}: recurrent-regime library and mode")
        ax[1].plot(td, v['tar_mode'], 'tab:blue', lw=0.6)
        ax[1].set_ylabel("mode (0=I,1=S,2=T,3=D)"); ax[1].set_xlabel("time [days]"); ax[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_tarucb_diag.png"), dpi=130); plt.close()

    # compact machine-readable summary
    summ = {name: dict(cum=float(v['rewards'][:ndec].sum()),
                       mean=float(v['rewards'][:ndec].mean()),
                       switches=int((np.diff(v['arms']) != 0).sum()),
                       frac_opt=float(np.mean(v['arms'][:ndec] == opt_arm)),
                       frac_near_opt=near_opt_frac(v['arms'][:ndec]))
            for name, v in results.items()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(dict(horizon_s=float(cache['horizon']), ndec=int(ndec), hold=hold,
                       eps_near=float(eps_near), oracle_table=oracle_table.tolist(),
                       policies=summ), f, indent=2)


if __name__ == "__main__":
    main()
