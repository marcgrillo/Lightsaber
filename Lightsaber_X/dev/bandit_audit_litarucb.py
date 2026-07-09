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
Deep audit of LI-TAR-UCB regime-level behaviour, answering four questions:

 1) Does the algorithm play the optimal (or near-optimal) controller in each regime,
    while it believes itself settled (STABLE)?
 2) If not, why -- which library entries are contaminated, and by what?
 3) How does it handle each true regime transition -- detection lag, detection path
    (drop-CUSUM vs scheduled-probe shape alarm), what DIAGNOSTIC decided, and whether
    it settled during a "blackout" (the deep-collapse phase of a transition where the
    logistic reward saturates near 0 for EVERY arm, so the shape is unidentifiable)?
 4) A/B of the proposed guarantee-preserving fixes (--fixes):
      (a) two-sided level CUSUM: alongside the existing one-sided DROP detector, an
          identical CUSUM on upward residuals, so a reward RISE (end of a blackout the
          algorithm mistakenly settled in) also triggers re-diagnosis. Same
          Lorden/Page-type delay + false-alarm bounds as the drop side (it is the same
          test applied to -X); a union bound doubles the false-alarm probability,
          absorbed by an O(log 2) increase in h (negligible).
      (b) blackout guard: TRANSITION batches / DIAGNOSTIC blocks whose best arm-mean is
          below eps_floor are treated as uninformative -- they never count toward the
          stability exit and never reach the library. Assumption added: every genuine
          regime's best-arm mean reward exceeds eps_floor by a margin (holds by design:
          calibrated best-arm rewards are 0.60/0.34/0.57 vs floor 0.17). Under it,
          blackout windows are a subset of the transition windows already charged in
          the regret bound, and the guard restores the draft's collapse model (additive
          collapse preserving centered shape), which logistic saturation violates.

    python bandit_audit_litarucb.py --cache bandit_runs/cache_1w_r1long --hold 100
    python bandit_audit_litarucb.py --cache bandit_runs/cache_1w_r1long --hold 100 --fixes
"""
import os, sys, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bandit_rewards
import bandit_policies as bp
from bandit_noise_cache import load_cache
from bandit_calibrate import calibrate
from bandit_long_experiment import run_policy_stream, shape_separation
from bandit_tune_tarucb import make_accessors

FS = 256
MODE_NAME = {0: "INIT", 1: "STABLE", 2: "TRANS", 3: "DIAG"}
REG_NAME = {0: "R0", 1: "R1", 2: "R2"}
MODE_COL = {"INIT": "tab:gray", "STABLE": "tab:blue", "TRANS": "tab:purple", "DIAG": "tab:brown"}
REG_COL = {"R0": "tab:green", "R1": "tab:orange", "R2": "tab:red"}


class AuditLITARUCB(bp.LITARUCB):
    """Same algorithm; adds passive per-step history of detector internals."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.G_hist, self.Gup_hist, self.ell_hist = [], [], []
        self.probing_hist, self.c_det_hist = [], []

    def update(self, arm, reward):
        super().update(arm, reward)
        self.G_hist.append(self.G)
        self.Gup_hist.append(float(getattr(self, 'G_up', 0.0)))
        self.ell_hist.append(self.ell)
        self.probing_hist.append(bool(self._probing))
        self.c_det_hist.append(self.c_det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bandit_runs/cache_1w_r1long")
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--reward", default="logistic", choices=["norm", "logistic"])
    ap.add_argument("--fixes", action="store_true",
                     help="enable the two proposed fixes (rise-CUSUM + blackout guard)")
    ap.add_argument("--max-figs", type=int, default=14)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = load_cache(args.cache); N = int(cache['N'])
    tag = "fixed" if args.fixes else "baseline"
    outdir = args.out or os.path.join(args.cache, f"litarucb_audit_{tag}")
    os.makedirs(outdir, exist_ok=True)

    calib = calibrate(hold=args.hold, reward_mode=args.reward)
    s = float(calib['sigma_hat']); lo, hi = float(calib['lo']), float(calib['hi'])
    oracle_table = np.asarray(calib['oracle_table'], float)          # (arm, regime)
    dsh = shape_separation(oracle_table)
    r_sh = float(np.clip(0.45 * dsh, 0.6 * s, 1.2 * s))
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    W_of_sample, sens_of_sample, Wsec, tsec = make_accessors(cache)

    # blackout floor: half the worst regime's best-arm reward. Best-arm rewards by
    # design are ~0.60/0.34/0.57, so floor ~0.17 sits ~4 sigma above the saturated
    # collapse level (~0.02) and ~4 sigma below the weakest genuine regime.
    eps_floor = 0.5 * float(oracle_table.max(axis=0).min())

    kw = dict(sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
              m_batch=1, r_stab=1.0*s, q_stab=1, d_exit=1.5*s, r_coll=r_sh,
              m_cls=6, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
              level_window=30, L_probe=90, m_blk=3, q_det=2, seed=0)
    if args.fixes:
        kw['h_up'] = 5.0 * s          # rise-CUSUM threshold, symmetric with drop side
        kw['eps_floor'] = eps_floor   # blackout guard
    pol = AuditLITARUCB(3, **kw)

    print(f"[audit:{tag}] cache={args.cache} sigma={s:.4f} r_sh={r_sh:.4f} "
          f"eps_floor={eps_floor:.3f} fixes={'ON' if args.fixes else 'off'}")
    h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                          args.hold, lo, hi, reward_mode=args.reward,
                          progress_every=0, name=f"LI-TAR-UCB-{tag}")

    ndec = len(h['rewards'])
    hold_n = args.hold * FS
    dec_t = (np.arange(ndec) * hold_n) / FS
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])
    reg_dom = np.argmax(Wdec, axis=0)
    Er = oracle_table @ Wdec                                          # (arm, ndec)
    opt_arm = np.argmax(Er, axis=0)
    near01 = Er >= (Er.max(axis=0) - 0.01)                            # (arm, ndec) bool
    near02 = Er >= (Er.max(axis=0) - 0.02)

    ck = os.path.join(args.cache, "experiment", "policies", "Oracle.npz")
    oracle_rewards = np.load(ck)['rewards'][:ndec] if os.path.exists(ck) else Er.max(axis=0)

    arms = h['arms'][:ndec]; rewards = h['rewards'][:ndec]
    mode = np.array(pol.mode_hist[:ndec]); Z = np.array(pol.Z_hist[:ndec])
    J_hist = np.array(pol.J_hist[:ndec])
    probing = np.array(pol.probing_hist[:ndec], bool)
    J = len(pol.th_sum)

    stable = mode == 1
    pure_stable = stable & ~probing            # exclude forced probing windows

    # ---- library identity: majority true regime per entry (STABLE co-occurrence) ----
    # NB: Z_hist can reference indices above the FINAL library size if entries were
    # merged mid-run (merge renumbers). Identity is behavioural (majority true regime
    # while that index was believed), well-defined per historical index either way.
    J_idx = max(J, int(Z.max()) + 1)
    identity = np.full(J_idx, -1, int)
    cooc = np.zeros((J_idx, 3), int)
    for j in range(J_idx):
        m = pure_stable & (Z == j)
        if m.sum():
            cooc[j] = np.bincount(reg_dom[m], minlength=3)
            identity[j] = int(np.argmax(cooc[j]))

    # ---- Q1: per-regime optimal / near-optimal rates over STABLE windows ----
    lines = []
    lines.append("=" * 96)
    lines.append(f" LI-TAR-UCB AUDIT ({tag})  --  cache={args.cache}  hold={args.hold}s  "
                 f"ndec={ndec} ({dec_t[-1]/86400:.2f}d)")
    if args.fixes:
        lines.append(f" fixes ON: rise-CUSUM h_up={kw['h_up']:.4f}, blackout guard "
                     f"eps_floor={eps_floor:.3f}")
    lines.append("=" * 96)
    regret = float(oracle_rewards.sum() - rewards.sum())
    switches = int((np.diff(arms) != 0).sum())
    lines.append(f"\nregret vs Oracle: {regret:.1f}   switches: {switches}   final J: {J}   "
                 f"frac windows STABLE: {stable.mean():.3f} (pure {pure_stable.mean():.3f})")

    lines.append("\n--- Q1: controller choice quality per TRUE regime (pure-STABLE windows only) ---")
    lines.append(f"{'regime':>7s} {'n_win':>6s} {'C0':>6s} {'C1':>6s} {'C2':>6s} "
                 f"{'optimal':>8s} {'near.01':>8s} {'near.02':>8s}   (near = within 0.01/0.02 "
                 f"of best expected reward)")
    for r in range(3):
        m = pure_stable & (reg_dom == r)
        n = int(m.sum())
        if n == 0:
            lines.append(f"{REG_NAME[r]:>7s} {0:>6d}   (never dominant during STABLE)")
            continue
        fr = [float(np.mean(arms[m] == k)) for k in range(3)]
        opt = float(np.mean(arms[m] == opt_arm[m]))
        n1 = float(np.mean(near01[arms[m], np.where(m)[0]]))
        n2 = float(np.mean(near02[arms[m], np.where(m)[0]]))
        lines.append(f"{REG_NAME[r]:>7s} {n:>6d} {fr[0]:>6.3f} {fr[1]:>6.3f} {fr[2]:>6.3f} "
                     f"{opt:>8.3f} {n1:>8.3f} {n2:>8.3f}")

    # same, but split by whether the algorithm's believed regime is the right one
    lines.append("\n  ... split by belief correctness (identity(Z) vs true regime):")
    lines.append(f"{'regime':>7s} {'belief':>9s} {'n_win':>6s} {'optimal':>8s} {'near.02':>8s}")
    for r in range(3):
        for ok, lab in [(True, "correct"), (False, "WRONG")]:
            m = pure_stable & (reg_dom == r) & ((identity[Z] == reg_dom) == ok)
            n = int(m.sum())
            if n == 0:
                continue
            opt = float(np.mean(arms[m] == opt_arm[m]))
            n2 = float(np.mean(near02[arms[m], np.where(m)[0]]))
            lines.append(f"{REG_NAME[r]:>7s} {lab:>9s} {n:>6d} {opt:>8.3f} {n2:>8.3f}")

    contam = float(np.mean(identity[Z[pure_stable]] != reg_dom[pure_stable])) if pure_stable.sum() else 0.0
    dark_stable = pure_stable & (rewards < eps_floor)
    lines.append(f"\ncontamination (STABLE windows spent under a mis-mapped belief): {contam:.3f}")
    lines.append(f"dark-STABLE windows (settled while reward < eps_floor={eps_floor:.3f}, i.e. "
                 f"believed 'settled' during a blackout): {int(dark_stable.sum())} "
                 f"({dark_stable.mean()*100:.2f}% of all windows)")

    # ---- Q2: library table ----
    lines.append("\n--- Q2: learned library ---")
    lines.append(f"{'entry':>5s} {'shape phi':>28s} {'pref':>5s} {'identity':>9s} "
                 f"{'R0/R1/R2 co-occurrence':>24s} {'flat?':>6s} {'true-best for identity':>22s}")
    for j in range(J):
        th = pol._theta(j)
        pref = int(np.argmax(th))
        ident = REG_NAME.get(identity[j], "unused") if identity[j] >= 0 else "unused"
        flat = "FLAT" if float(np.max(np.abs(th))) < 0.5 * r_sh else ""
        tb = ""
        if identity[j] >= 0:
            col = oracle_table[:, identity[j]]
            best_set = [f"C{k}" for k in range(3) if col[k] >= col.max() - 0.01]
            tb = "/".join(best_set) + (" MISMATCH" if pref not in
                 [k for k in range(3) if col[k] >= col.max() - 0.01] else " ok")
        lines.append(f"{j:>5d} {np.array2string(np.round(th,4), separator=','):>28s} "
                     f"C{pref:<4d} {ident:>9s} {str(cooc[j].tolist()):>24s} {flat:>6s} {tb:>22s}")

    # ---- Q3: transition audit ----
    changes = [i for i in range(1, ndec) if reg_dom[i] != reg_dom[i - 1]]
    probe_alarm_t = {p[0] for p in pol.probe_log if p[3]}
    lines.append(f"\n--- Q3: transition audit ({len(changes)} true dominant-regime changes) ---")
    lines.append(f"{'#':>3s} {'t[h]':>7s} {'change':>8s} {'detect':>7s} {'path':>12s} "
                 f"{'settle':>7s} {'settledZ':>8s} {'mapped':>7s} {'ok?':>4s} {'dark?':>5s}")
    n_missed = n_dark_settle = n_wrong_settle = 0
    audit_rows = []
    for ci, c in enumerate(changes):
        to_r = reg_dom[c]
        nxt = changes[ci + 1] if ci + 1 < len(changes) else ndec
        if mode[c] != 1:                      # already out of STABLE at the change
            det = c; path = "(pre)"
        else:
            det = None
            for i in range(c, nxt):
                if mode[i] == 2:
                    det = i
                    path = "shape" if (i + 1) in probe_alarm_t else "drop"
                    break
        if det is None:
            lines.append(f"{ci:>3d} {dec_t[c]/3600:>7.2f} "
                         f"{REG_NAME[reg_dom[c-1]]+'->'+REG_NAME[to_r]:>8s} "
                         f"{'--':>7s} {'MISSED':>12s} {'--':>7s} {'--':>8s} {'--':>7s} "
                         f"{'--':>4s} {'--':>5s}")
            n_missed += 1
            audit_rows.append((c, None, None))
            continue
        st = None
        for i in range(det, nxt):
            if mode[i] == 1:
                st = i
                break
        if st is None:
            lines.append(f"{ci:>3d} {dec_t[c]/3600:>7.2f} "
                         f"{REG_NAME[reg_dom[c-1]]+'->'+REG_NAME[to_r]:>8s} "
                         f"{(det-c)*args.hold/60:>6.0f}m {path:>12s} {'--':>7s} {'--':>8s} "
                         f"{'--':>7s} {'--':>4s} {'--':>5s}")
            audit_rows.append((c, det, None))
            continue
        zs = int(Z[st])
        mapped = REG_NAME.get(identity[zs], "?")
        ok = identity[zs] == to_r
        dark = bool(np.mean(rewards[st:min(st + 5, ndec)]) < eps_floor)
        n_dark_settle += dark
        n_wrong_settle += (not ok)
        lines.append(f"{ci:>3d} {dec_t[c]/3600:>7.2f} "
                     f"{REG_NAME[reg_dom[c-1]]+'->'+REG_NAME[to_r]:>8s} "
                     f"{(det-c)*args.hold/60:>6.0f}m {path:>12s} "
                     f"{(st-c)*args.hold/60:>6.0f}m {'Z='+str(zs):>8s} {mapped:>7s} "
                     f"{'yes' if ok else 'NO':>4s} {'DARK' if dark else '':>5s}")
        audit_rows.append((c, det, st))
    lines.append(f"\nsummary: {n_missed} missed, {n_wrong_settle} settled on a mis-mapped entry, "
                 f"{n_dark_settle} settled DURING a blackout (the reported flaw)")

    report = "\n".join(lines)
    rp = os.path.join(outdir, "audit_report.txt")
    with open(rp, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[audit:{tag}] report -> {rp}")

    # ---- plots ----
    th_h = dec_t / 3600.0
    # contamination timeline
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1]))
    for r in range(3):
        m = reg_dom == r
        ax[0].scatter(th_h[m], np.full(m.sum(), r), s=2, color=REG_COL[REG_NAME[r]])
    ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["R0", "R1", "R2"])
    ax[0].set_ylabel("true regime"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Belief-vs-truth timeline ({tag}); red x = STABLE under mis-mapped belief")
    bel = np.where(Z >= 0, identity[Z], -1)
    okm = pure_stable & (bel == reg_dom)
    bad = pure_stable & (bel != reg_dom)
    ax[1].scatter(th_h[okm], bel[okm], s=2, color="tab:green", label="belief correct")
    ax[1].scatter(th_h[bad], bel[bad], s=6, color="red", marker="x", label="belief WRONG")
    ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["R0", "R1", "R2"])
    ax[1].set_ylabel("believed regime\n(mapped)"); ax[1].set_xlabel("time [h]")
    ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_belief_timeline.png"), dpi=130); plt.close()

    # per-transition zooms
    nfig = 0
    for ci, (c, det, st) in enumerate(audit_rows):
        if nfig >= args.max_figs:
            break
        i0 = max(0, c - 30)
        i1 = min(ndec - 1, (st if st is not None else (det if det is not None else c)) + 60)
        fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        seg = mode[i0:i1]
        bnd = np.concatenate(([0], np.where(np.diff(seg) != 0)[0] + 1, [len(seg)]))
        for x0, x1 in zip(bnd[:-1], bnd[1:]):
            ax[0].axvspan(th_h[i0 + x0], th_h[i0 + min(x1, len(seg) - 1)],
                          color=MODE_COL[MODE_NAME[seg[x0]]], alpha=0.45, lw=0)
        ax[0].plot(th_h[i0:i1], rewards[i0:i1], "k", lw=0.9)
        ax[0].axhline(eps_floor, color="red", ls=":", lw=1, label=f"eps_floor={eps_floor:.2f}")
        ax[0].axvline(th_h[c], color="k", ls="--", lw=1.2, label="true change")
        if det is not None:
            ax[0].axvline(th_h[det], color="tab:purple", ls="--", lw=1.2, label="detected")
        if st is not None:
            ax[0].axvline(th_h[st], color="tab:blue", ls="--", lw=1.2, label="settled")
        ax[0].set_ylabel("reward")
        ax[0].set_title(f"transition {ci}: {REG_NAME[reg_dom[c-1]]}->{REG_NAME[reg_dom[c]]} "
                         f"at t={th_h[c]:.2f}h")
        ax[0].legend(loc="upper right", fontsize=7, ncol=2)
        ax[0].grid(alpha=0.3)
        ax[1].step(th_h[i0:i1], arms[i0:i1], where="post", lw=1.1)
        bel_seg = bel[i0:i1]
        badm = (bel_seg != reg_dom[i0:i1]) & (mode[i0:i1] == 1)
        ax[1].scatter(th_h[i0:i1][badm], arms[i0:i1][badm], s=12, color="red", marker="x",
                      label="STABLE, wrong belief")
        ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["C0", "C1", "C2"])
        ax[1].set_ylabel("arm"); ax[1].set_xlabel("time [h]")
        ax[1].legend(loc="upper right", fontsize=7)
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fig_transition_{ci:02d}.png"), dpi=130); plt.close()
        nfig += 1
    print(f"[audit:{tag}] figures -> {outdir} ({nfig} transition zooms)")


if __name__ == "__main__":
    main()
