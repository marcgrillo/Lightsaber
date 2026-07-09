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
Detailed, human-readable walkthrough of the LI-TAR-UCB algorithm over a multi-day
horizon -- a per-window CSV log, a narrated plain-English event log, and a set of
plots -- meant so someone who has never seen the algorithm can follow, step by step,
everything it does: initial exploration, steady-state UCB play, scheduled probing,
both change-detectors (drop-CUSUM and shape-alarm), the transition/diagnostic
re-classification pipeline, and library maintenance (new-regime creation, merging).

Does NOT modify bandit_policies.LITARUCB. LoggingLITARUCB below only ADDS passive
per-step history lists after calling the real update() -- exactly the same pattern
the class already uses internally for Z_hist/J_hist/mode_hist/diag_log/probe_log.

    python bandit_log_litarucb.py --cache bandit_runs/cache_3d_r1long --hold 100

    # report only the trailing 3 days of a longer run (policy still runs the FULL
    # horizon first, so the reported tail is warmed-up, not biased by INIT/early
    # library-building churn):
    python bandit_log_litarucb.py --cache bandit_runs/cache_1w_r1long --hold 100 --tail-days 3
"""
import os, sys, csv, argparse
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
REG_COL = {"R0": "tab:green", "R1": "tab:orange", "R2": "tab:red"}
MODE_COL = {"INIT": "tab:gray", "STABLE": "tab:blue", "TRANS": "tab:purple", "DIAG": "tab:brown"}


class LoggingLITARUCB(bp.LITARUCB):
    """Same algorithm as bp.LITARUCB; only adds passive diagnostic history."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.G_hist, self.ell_hist = [], []
        self.probing_hist, self.since_probe_hist, self.c_det_hist = [], [], []

    def update(self, arm, reward):
        super().update(arm, reward)
        self.G_hist.append(self.G)
        self.ell_hist.append(self.ell)
        self.probing_hist.append(bool(self._probing))
        self.since_probe_hist.append(self._since_probe)
        self.c_det_hist.append(self.c_det)


# --------------------------------------------------------------------------------
# narrative text
# --------------------------------------------------------------------------------

def fmt_t(t_s):
    d, rem = divmod(t_s, 86400.0)
    h = rem / 3600.0
    return f"day {int(d)+1}, {h:5.2f}h (t={t_s/3600.0:6.2f}h from start)"


def build_intro(pol, s, r_sh, dsh, hold, tail_note=None):
    tail_block = f"\n{tail_note}\n" if tail_note else ""
    return f"""
================================================================================
 LI-TAR-UCB (Level-Invariant Transition-Aware Recurrent UCB) -- WALKTHROUGH LOG
================================================================================
{tail_block}
WHAT PROBLEM IS THIS SOLVING?
  A gravitational-wave detector's alignment control loop must pick one of K={pol.K}
  fixed controllers (C0, C1, C2) to run at any moment. Which controller is best
  depends on the current disturbance environment (an unobserved, time-varying
  "regime": nominal ground motion R0, a microseism spike R1, or elevated ground
  motion R2). The algorithm does NOT get told which regime is active -- it must
  infer this purely from the reward (a performance score) it observes after
  playing each controller for one decision window ({hold}s here).

THE KEY IDEA -- RECOGNISE REGIMES BY "SHAPE", NOT BY LEVEL:
  A naive approach would remember "regime X gives roughly reward vector v" and
  recognise X again when the reward vector looks like v. The problem: the
  absolute reward LEVEL for the same physical regime can drift over time (e.g.
  a slow change in the baseline seismic noise floor), while the RELATIVE
  ranking of which controller is best (its "shape", i.e. the reward vector
  minus its own mean) stays the same. LI-TAR-UCB stores and matches regimes by
  this centered shape phi(v) = v - mean(v), so it recognises a returning regime
  even if the absolute level has moved.

  It keeps a growing LIBRARY of distinct regimes it has discovered so far
  (library size "J"); each entry holds the running average shape estimate for
  one regime. "Z" is the index into this library of whichever regime the
  algorithm currently believes is active. Z and J only ever mean something to
  the algorithm's own bookkeeping -- they are NOT the same numbering as the
  environment's true R0/R1/R2 labels (which this log also reports, for the
  reader's benefit, since we -- unlike the algorithm -- have access to the
  simulator's ground truth).

FOUR OPERATING MODES:
  INIT    Runs once, at the very start. Round-robins every arm m0={pol.m0} times
          to get an initial balanced estimate of the reward vector, and creates
          the FIRST library entry (regime index 0) from it. Then moves to STABLE.

  STABLE  Normal operation. Plays the arm with the highest UCB score
          (mean shape estimate for the current regime + a confidence bonus that
          shrinks as more data accumulates) -- standard "optimism under
          uncertainty" bandit play, but restricted to whichever library entry Z
          is currently believed active.

          Two independent change-detectors run concurrently while in STABLE:

          (a) A level-adjusted drop-CUSUM (running statistic "G"): tracks
              whether the played arm's reward is persistently BELOW what the
              current regime's library entry predicts (after subtracting a
              slowly-updated nuisance-level term "ell" that absorbs harmless
              level drift). G accumulates evidence of an unexpected drop and
              decays by a "leak" nu={pol.nu:.4f} each step; it fires when
              G >= h={pol.h:.4f}. This catches regime changes that show up as
              an overall performance COLLAPSE.

          (b) Scheduled balanced probing: every L_probe={pol.L_probe} windows,
              the algorithm pauses UCB play and round-robins all arms
              m_blk={pol.m_blk} times each ("a probing block"), regardless of
              how well things seem to be going. It compares this block's
              centered shape to the current library entry's shape: if within
              r_det={pol.r_det:.4f} (sup-norm), the block is folded into the
              library (refining the estimate); if it deviates for q_det={pol.q_det}
              consecutive blocks, that's a SHAPE-CHANGE alarm. This exists
              because (a) alone is blind to a regime change that keeps the
              PLAYED arm's absolute reward roughly constant but silently
              rearranges which arm would be best (e.g. this environment's
              R0<->R2 transitions, which have similar overall reward levels).

          Either alarm moves the algorithm into TRANSITION mode.

  TRANS   Something changed; the algorithm doesn't yet know if the environment
          has settled into a new steady state or is still transiently ramping.
          It round-robins in small batches (m_batch={pol.m_batch} per arm) and
          waits for q_stab={pol.q_stab} CONSECUTIVE batches that are (i) stable
          batch-to-batch (moved by <= r_stab={pol.r_stab:.4f}) AND (ii) not
          merely a uniformly-depressed ("collapsed") copy of the shape it just
          left (guard against mistaking a mid-drop transient for a settled
          regime). Once satisfied, moves to DIAGNOSTIC.

  DIAG    Samples m_cls={pol.m_cls} per arm, computes the resulting centered
          shape, and compares it (sup-norm distance) against EVERY entry
          currently in the library.
            - If the closest entry is within r_sh={pol.r_sh:.4f}, this is a
              RECURRENCE: the algorithm reuses that library entry (folds the
              new data in, and checks whether any two library entries have
              become close enough to merge, pooling their statistics).
            - Otherwise this is a candidate for a genuinely NEW regime -- but
              only committed after q_new={pol.q_new} CONSECUTIVE diagnostic
              blocks agree it doesn't match anything (persistence, so one noisy
              block can't spuriously fragment the library). Once committed, a
              new library entry is created (J increases by one) and merge is
              checked again.
          Either way, the algorithm then returns to STABLE play under
          whichever regime (old or new) it just settled on.

READING THIS LOG:
  Below is a chronological list of every notable event over the logged horizon:
  mode transitions, probing-block outcomes, drop-alarms, diagnostic
  classifications, new library entries, and merges. Each event reports the
  algorithm's internal state PLUS (in parentheses, prefixed "ground truth:")
  what the simulator's true regime actually was at that time -- information the
  algorithm itself never sees, included purely so a reader can check whether
  its internal bookkeeping lines up with reality.

  Calibration values used this run: reward noise sigma={s:.4f}, shape
  separation between the three designed regimes Delta_sh={dsh:.4f} (this is
  how far apart the TRUE regimes' shapes are; r_sh={r_sh:.4f} is the detection
  threshold derived from it, per the algorithm's theory).
================================================================================

"""


def build_events(pol, dec_t, reg_true, opt_arm, ndec, i_start=0):
    """Chronological list of (t_hours, header, body) event strings, covering only
    windows [i_start, ndec) -- but pol's history (mode_hist/J_hist/probe_log/diag_log)
    still reflects the algorithm's FULL run, so state entering i_start (mode, library
    size) is whatever it actually was after running the earlier, unreported windows."""
    events = []
    mode_hist = np.asarray(pol.mode_hist[:ndec])
    J_hist = np.asarray(pol.J_hist[:ndec])
    probe_by_t = {p[0]: p for p in pol.probe_log}     # t -> (t, D_b, c_det, alarmed)
    diag_by_t = {}
    for d in pol.diag_log:
        diag_by_t.setdefault(d[0], []).append(d)

    def gt(i):
        return f"ground truth: {REG_NAME[reg_true[i]]} (optimal arm C{opt_arm[i]})"

    prev_mode = mode_hist[i_start - 1] if i_start > 0 else None
    prev_J = J_hist[i_start - 1] if i_start > 0 else 0
    for i in range(i_start, ndec):
        t = i + 1                      # self.t is 1-based
        m = mode_hist[i]
        # --- probing-block outcome (only relevant while STABLE/just-left-STABLE) ---
        if t in probe_by_t:
            _, D_b, c_det, alarmed = probe_by_t[t]
            if alarmed:
                events.append((dec_t[i]/3600.0,
                    f"SHAPE-CHANGE ALARM (scheduled probe)",
                    f"  A scheduled probing block finished at {fmt_t(dec_t[i])}. Its shape "
                    f"deviated from the current library entry by D_b={D_b:.4f} "
                    f"(> r_det={pol.r_det:.4f}) for {c_det} consecutive block(s) now "
                    f"(threshold q_det={pol.q_det}) -> alarm raised, entering TRANSITION.\n"
                    f"  ({gt(i)})"))
            else:
                verdict = "consistent -> folded into the library" if D_b <= pol.r_det else \
                          f"deviating (evidence {c_det}/{pol.q_det})"
                events.append((dec_t[i]/3600.0,
                    f"probe block",
                    f"  Scheduled probing block completed at {fmt_t(dec_t[i])}: D_b={D_b:.4f}, "
                    f"{verdict}. ({gt(i)})"))
        # --- diagnostic classification outcome ---
        if t in diag_by_t:
            for (_, psi, Dlist, jstar, Dmin, not_matched) in diag_by_t[t]:
                Dstr = ", ".join(f"D(j={j})={d:.4f}" for j, d in enumerate(Dlist))
                if not_matched:
                    events.append((dec_t[i]/3600.0,
                        "diagnostic block: no library match",
                        f"  Diagnostic block at {fmt_t(dec_t[i])}: nearest library entry is "
                        f"j={jstar} at distance Dmin={Dmin:.4f} (> r_sh={pol.r_sh:.4f}) -> "
                        f"does not match anything known yet ({Dstr}). ({gt(i)})"))
                else:
                    events.append((dec_t[i]/3600.0,
                        "diagnostic block: recurrence match",
                        f"  Diagnostic block at {fmt_t(dec_t[i])}: matched library entry "
                        f"j={jstar} at distance Dmin={Dmin:.4f} (<= r_sh={pol.r_sh:.4f}) -> "
                        f"RECURRENCE, reusing regime {jstar} ({Dstr}). ({gt(i)})"))
        # --- mode transitions ---
        if prev_mode is not None and m != prev_mode:
            frm, to = MODE_NAME[prev_mode], MODE_NAME[m]
            if frm == "INIT" and to == "STABLE":
                events.append((dec_t[i]/3600.0, "INIT complete",
                    f"  Initial round-robin exploration finished at {fmt_t(dec_t[i])}. "
                    f"First library entry (regime 0) created from {pol.m0} balanced "
                    f"samples/arm. Entering STABLE play. ({gt(i)})"))
            elif frm == "STABLE" and to == "TRANS":
                if t not in probe_by_t:      # not already reported as a shape-alarm above
                    events.append((dec_t[i]/3600.0, "DROP-ALARM (CUSUM)",
                        f"  Level-adjusted drop-CUSUM crossed its threshold at "
                        f"{fmt_t(dec_t[i])} (G >= h={pol.h:.4f}): observed reward has been "
                        f"persistently below the current regime's prediction -> entering "
                        f"TRANSITION. ({gt(i)})"))
            elif frm == "TRANS" and to == "DIAG":
                events.append((dec_t[i]/3600.0, "TRANSITION stabilised",
                    f"  The post-alarm reward vector held stable for {pol.q_stab} "
                    f"consecutive batch(es) and was not a mere collapsed copy of the "
                    f"previous regime -> entering DIAGNOSTIC classification at "
                    f"{fmt_t(dec_t[i])}. ({gt(i)})"))
            elif frm == "DIAG" and to == "STABLE":
                pass   # already reported by the diagnostic-block event above
        # --- library size changes (new regime / merge) ---
        if J_hist[i] > prev_J:
            events.append((dec_t[i]/3600.0, "NEW LIBRARY ENTRY",
                f"  Library grew from J={prev_J} to J={J_hist[i]} at {fmt_t(dec_t[i])}: a "
                f"persistent (q_new={pol.q_new}-block) non-match was committed as a "
                f"genuinely new regime. ({gt(i)})"))
        elif J_hist[i] < prev_J:
            events.append((dec_t[i]/3600.0, "LIBRARY MERGE",
                f"  Library shrank from J={prev_J} to J={J_hist[i]} at {fmt_t(dec_t[i])}: "
                f"two entries whose shape estimates converged within r_merge="
                f"{pol.r_merge:.4f} were pooled into one. ({gt(i)})"))
        prev_mode = m; prev_J = J_hist[i]
    return events


def build_summary(pol, dec_t, reg_true, opt_arm, arms, rewards, i_start, ndec, oracle_cum):
    """Stats below cover only the REPORTED window [i_start, ndec); the library shape
    estimates (pol._theta) reflect the algorithm's FULL run (they keep accumulating
    across the whole horizon, reported window or not)."""
    reg_true, opt_arm, arms, rewards = (reg_true[i_start:ndec], opt_arm[i_start:ndec],
                                         arms[i_start:ndec], rewards[i_start:ndec])
    J = len(pol.th_sum)
    lines = ["", "=" * 80, " END-OF-RUN SUMMARY (reported window only)", "=" * 80, ""]
    lines.append(f"Reported window: {fmt_t(dec_t[i_start])} through {fmt_t(dec_t[ndec-1])} "
                 f"({(dec_t[ndec-1]-dec_t[i_start])/3600.0:.2f} h, {ndec-i_start} decision "
                 f"windows). The policy ran from t=0 with full history carried in, so this "
                 f"is NOT a cold start -- the library below reflects everything learned so far.")
    lines.append(f"Final library size: J={J} discovered regime(s).")
    lines.append("")
    lines.append("Per-library-entry summary (what the algorithm learned about each; "
                 "'played for' counts only the reported window):")
    Z_hist = np.asarray(pol.Z_hist[i_start:ndec])
    for j in range(J):
        th = pol._theta(j)
        mask = Z_hist == j
        n_windows = int(mask.sum())
        best_arm = int(np.argmax(th))
        mean_true_regime = "n/a"
        if n_windows > 0:
            counts = np.bincount(reg_true[mask], minlength=3)
            mean_true_regime = "/".join(f"{REG_NAME[r]}={c}" for r, c in enumerate(counts) if c > 0)
        lines.append(f"  regime {j}: centered shape phi={np.round(th,4).tolist()}, "
                     f"preferred arm=C{best_arm}, played for {n_windows} windows "
                     f"while active (true regime breakdown: {mean_true_regime})")
    lines.append("")
    frac_opt = float(np.mean(arms == opt_arm))
    switches = int((np.diff(arms) != 0).sum())
    regret = oracle_cum - float(rewards.sum())
    lines.append(f"Cumulative reward: {rewards.sum():.2f}  |  Oracle: {oracle_cum:.2f}  |  "
                 f"regret: {regret:.2f}")
    lines.append(f"Fraction of windows playing the true optimal arm: {frac_opt:.3f}")
    lines.append(f"Number of controller switches: {switches}")
    lines.append("")
    # confusion: true regime vs Z (majority-vote mapped)
    lines.append("Ground-truth regime vs. algorithm's believed regime (Z), row-normalised %:")
    conf = np.zeros((3, J))
    for r in range(3):
        m = reg_true == r
        if m.sum() == 0:
            continue
        counts = np.bincount(Z_hist[m], minlength=J)
        conf[r] = 100.0 * counts / max(m.sum(), 1)
    header = "            " + "".join(f"Z={j:<7d}" for j in range(J))
    lines.append(header)
    for r in range(3):
        row = "".join(f"{conf[r,j]:6.1f}%  " for j in range(J))
        lines.append(f"  true {REG_NAME[r]}:  {row}")
    lines.append("")
    lines.append("(If R0 and R2 columns spread across multiple Z entries, or share one Z, that")
    lines.append(" reflects the documented near-degenerate/level-invariant blind spots -- see")
    lines.append(" repo memory notes on this experiment for the full context.)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------

def make_plots(outdir, dec_t, reg_true, mode_hist, Z_hist, J_hist, G_hist, ell_hist,
               c_det_hist, arms, rewards, opt_arm, pol, oracle_rewards, i_start=0):
    """dec_t/reg_true/.../oracle_rewards are the FULL-run arrays; only [i_start:] is
    plotted (so window/probe indices below still reference the full pol logs)."""
    dec_t, reg_true, mode_hist, Z_hist, J_hist, G_hist, ell_hist, c_det_hist, arms, \
        rewards, opt_arm, oracle_rewards = (
            dec_t[i_start:], reg_true[i_start:], mode_hist[i_start:], Z_hist[i_start:],
            J_hist[i_start:], G_hist[i_start:], ell_hist[i_start:], c_det_hist[i_start:],
            arms[i_start:], rewards[i_start:], opt_arm[i_start:], oracle_rewards[i_start:])
    th = dec_t / 3600.0
    horizon_h = th[-1]

    # ---- fig 1: overview (regime+reward, mode, Z) ----
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 0.8, 1.1]))
    d = reg_true
    change = np.where(np.diff(d) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [len(d)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax[0].axvspan(th[a], th[min(b, len(th)-1)], color=REG_COL[REG_NAME[d[a]]], alpha=0.15, lw=0)
    ax[0].plot(th, rewards, color="k", lw=0.5)
    ax[0].set_ylabel("reward\n(bg = true regime)")
    ax[0].set_title("LI-TAR-UCB over the logged horizon -- true regime & reward")
    ax[0].grid(alpha=0.3)

    m = mode_hist
    change = np.where(np.diff(m) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [len(m)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax[1].axvspan(th[a], th[min(b, len(th)-1)], color=MODE_COL[MODE_NAME[m[a]]], alpha=0.6, lw=0)
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=c, label=n) for n, c in MODE_COL.items()],
                 loc="upper right", ncol=4, fontsize=8)
    ax[1].set_yticks([]); ax[1].set_ylabel("algorithm\nmode")
    ax[1].grid(alpha=0.3)

    cmap = plt.get_cmap("tab10")
    for j in range(int(J_hist.max()) + 1):
        mask = Z_hist == j
        ax[2].scatter(th[mask], Z_hist[mask], s=3, color=cmap(j % 10), label=f"Z={j}")
    ax[2].set_ylabel("believed regime (Z)")
    ax[2].set_xlabel("time [h]")
    ax[2].legend(loc="upper right", ncol=6, fontsize=8, markerscale=3)
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(outdir, "fig1_overview.png")
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  saved {fp}")

    # ---- fig 2: detectors ----
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    ax[0].plot(th, G_hist, color="tab:purple", lw=0.8, label="G (drop-CUSUM)")
    ax[0].axhline(pol.h, color="red", ls="--", lw=1, label=f"h={pol.h:.3f} (alarm)")
    alarm_idx = np.where(np.diff((G_hist >= pol.h).astype(int)) == 1)[0] + 1
    for i in alarm_idx:
        ax[0].axvline(th[i], color="red", ls=":", lw=0.8, alpha=0.6)
    ax[0].set_ylabel("G"); ax[0].legend(loc="upper right", fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("Change-detector internals: drop-CUSUM (top) and scheduled-probe deviations (bottom)")

    probe_local = [p for p in pol.probe_log if p[0] - 1 - i_start >= 0]
    if probe_local:
        pt_h = np.array([dec_t[p[0]-1-i_start] for p in probe_local]) / 3600.0
        Db = np.array([p[1] for p in probe_local])
        alarmed = np.array([p[3] for p in probe_local])
        ax[1].scatter(pt_h[~alarmed], Db[~alarmed], s=14, color="tab:blue", label="probe block (consistent/evidence)")
        ax[1].scatter(pt_h[alarmed], Db[alarmed], s=30, color="red", marker="x", label="probe block (ALARM)")
    ax[1].axhline(pol.r_det, color="red", ls="--", lw=1, label=f"r_det={pol.r_det:.3f}")
    ax[1].set_ylabel("probe deviation D_b"); ax[1].set_xlabel("time [h]")
    ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(outdir, "fig2_detectors.png")
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  saved {fp}")

    # ---- fig 3: library growth ----
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.step(th, J_hist, where="post", color="tab:blue", lw=1.2)
    grow = np.where(np.diff(J_hist) > 0)[0] + 1
    shrink = np.where(np.diff(J_hist) < 0)[0] + 1
    ax.scatter(th[grow], J_hist[grow], color="green", marker="^", s=60, zorder=5, label="new regime")
    ax.scatter(th[shrink], J_hist[shrink], color="red", marker="v", s=60, zorder=5, label="merge")
    ax.set_ylabel("library size J"); ax.set_xlabel("time [h]")
    ax.set_title("Library size over time")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(outdir, "fig3_library_growth.png")
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  saved {fp}")

    # ---- fig 4: reward/regret ----
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    ax[0].plot(th, rewards, color="k", lw=0.4, alpha=0.6, label="reward")
    subopt = arms != opt_arm
    ax[0].scatter(th[subopt], rewards[subopt], s=4, color="red", alpha=0.5, label="suboptimal arm played")
    ax[0].set_ylabel("reward"); ax[0].legend(loc="lower right", fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title("Reward & cumulative regret vs. Oracle")
    cum_regret = np.cumsum(oracle_rewards - rewards)
    ax[1].plot(th, cum_regret, color="tab:red", lw=1.2)
    ax[1].set_ylabel("cumulative regret"); ax[1].set_xlabel("time [h]"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(outdir, "fig4_regret.png")
    plt.savefig(fp, dpi=130); plt.close()
    print(f"  saved {fp}")

    # ---- fig 5: zoomed episodes (each TRANS/DIAG excursion) ----
    is_settle = (mode_hist == 2) | (mode_hist == 3)     # TRANS or DIAG
    dd = np.diff(np.concatenate(([0], is_settle.astype(int), [0])))
    starts = np.where(dd == 1)[0]; ends = np.where(dd == -1)[0]
    print(f"  {len(starts)} TRANS/DIAG episode(s) -> zoomed figures")
    for k, (a, b) in enumerate(zip(starts, ends)):
        margin = 20
        i0, i1 = max(0, a - margin), min(len(th) - 1, b + margin)
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        seg_mode = mode_hist[i0:i1]
        change = np.where(np.diff(seg_mode) != 0)[0] + 1
        bounds = np.concatenate(([0], change, [len(seg_mode)]))
        for x0, x1 in zip(bounds[:-1], bounds[1:]):
            ax[0].axvspan(th[i0+x0], th[i0+min(x1, len(seg_mode)-1)],
                          color=MODE_COL[MODE_NAME[seg_mode[x0]]], alpha=0.5, lw=0)
        ax[0].plot(th[i0:i1], rewards[i0:i1], color="k", lw=1.0)
        ax[0].set_ylabel("reward")
        ax[0].set_title(f"Episode {k}: mode excursion at t={th[a]:.2f}h "
                         f"(true regime path: " +
                         "->".join(dict.fromkeys(REG_NAME[r] for r in reg_true[i0:i1])) + ")")
        ax[0].legend(handles=[Patch(color=c, label=n) for n, c in MODE_COL.items()],
                     loc="upper right", ncol=4, fontsize=7)
        ax[1].step(th[i0:i1], arms[i0:i1], where="post", color="tab:blue", lw=1.2)
        ax[1].set_yticks([0, 1, 2]); ax[1].set_yticklabels(["C0", "C1", "C2"])
        ax[1].set_ylabel("arm played"); ax[1].set_xlabel("time [h]")
        ax[0].grid(alpha=0.3); ax[1].grid(alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(outdir, f"fig5_episode_{k:02d}.png")
        plt.savefig(fp, dpi=130); plt.close()
        print(f"  saved {fp}")


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bandit_runs/cache_3d_r1long")
    ap.add_argument("--hold", type=int, default=100)
    ap.add_argument("--reward", default="logistic", choices=["norm", "logistic"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tail-days", type=float, default=None,
                     help="only report/plot the LAST N days of the cache's horizon "
                          "(the policy still runs the full horizon first, so the "
                          "reported tail is warmed-up, not a cold start). Default: "
                          "report the whole horizon.")
    args = ap.parse_args()

    cache = load_cache(args.cache); N = int(cache['N'])
    suffix = f"_last{args.tail_days:g}d" if args.tail_days else ""
    outdir = args.out or os.path.join(args.cache, f"litarucb_log{suffix}")
    os.makedirs(outdir, exist_ok=True)

    calib = calibrate(hold=args.hold, reward_mode=args.reward)
    s = float(calib['sigma_hat']); lo, hi = float(calib['lo']), float(calib['hi'])
    oracle_table = np.asarray(calib['oracle_table'], float)
    dsh = shape_separation(oracle_table)
    r_sh = float(np.clip(0.45 * dsh, 0.6 * s, 1.2 * s))
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    W_of_sample, sens_of_sample, Wsec, tsec = make_accessors(cache)

    # SAME production hyperparameters as bandit_long_experiment.make_policies()
    pol = LoggingLITARUCB(3, sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                           m_batch=1, r_stab=1.0*s, q_stab=1, d_exit=1.5*s, r_coll=r_sh,
                           m_cls=6, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
                           level_window=30, L_probe=90, m_blk=3, q_det=2, seed=0)

    print(f"[log] cache={args.cache}  sigma={s:.4f} r_sh={r_sh:.4f} h={pol.h:.4f} nu={pol.nu:.4f}")
    h = run_policy_stream(pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                          args.hold, lo, hi, reward_mode=args.reward,
                          progress_every=0, name="LI-TAR-UCB-log")

    ndec = len(h['rewards'])
    hold_n = args.hold * FS
    dec_t = (np.arange(ndec) * hold_n) / FS       # seconds at window START
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])
    reg_true = np.argmax(Wdec, axis=0)
    opt_arm = np.argmax(oracle_table @ Wdec, axis=0)
    oracle_rewards = np.max(oracle_table @ Wdec, axis=0)

    # Oracle's actual reward trace (accounts for switching cost etc.), for regret
    oracle_pol = bp.Oracle(oracle_table)
    ho = run_policy_stream(oracle_pol, cache, sens_of_sample, W_of_sample, feat_names, w,
                           args.hold, lo, hi, reward_mode=args.reward,
                           progress_every=0, name="Oracle-log")
    oracle_rewards = ho['rewards'][:ndec]

    arms = h['arms'][:ndec]; rewards = h['rewards'][:ndec]
    mode_hist = np.array(pol.mode_hist[:ndec]); Z_hist = np.array(pol.Z_hist[:ndec])
    J_hist = np.array(pol.J_hist[:ndec])
    G_hist = np.array(pol.G_hist[:ndec]); ell_hist = np.array(pol.ell_hist[:ndec])
    probing_hist = np.array(pol.probing_hist[:ndec])
    c_det_hist = np.array(pol.c_det_hist[:ndec])

    if args.tail_days:
        tail_windows = int(round(args.tail_days * 86400.0 / args.hold))
        i_start = max(0, ndec - tail_windows)
    else:
        i_start = 0
    print(f"[log] full run: {ndec} windows ({dec_t[-1]/86400.0:.2f}d); reporting windows "
          f"[{i_start}, {ndec}) = {fmt_t(dec_t[i_start])} onward")

    # ---------------- per-window CSV (reported window only) ----------------
    csv_path = os.path.join(outdir, "log_full.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_window", "time_s", "time_h", "true_regime", "opt_arm", "mode", "Z", "J",
                     "arm_played", "reward", "G", "ell", "probing", "c_det"])
        for i in range(i_start, ndec):
            wr.writerow([i, dec_t[i], f"{dec_t[i]/3600.0:.4f}", REG_NAME[reg_true[i]], opt_arm[i],
                         MODE_NAME[mode_hist[i]], Z_hist[i], J_hist[i], arms[i],
                         f"{rewards[i]:.4f}", f"{G_hist[i]:.4f}", f"{ell_hist[i]:.4f}",
                         int(probing_hist[i]), c_det_hist[i]])
    print(f"[log] wrote {csv_path} ({ndec - i_start} windows)")

    # ---------------- narrative log ----------------
    txt_path = os.path.join(outdir, "log_narrative.txt")
    tail_note = None
    if i_start > 0:
        tail_note = (
            f"NOTE: this log reports only the TAIL of a longer run. The policy actually\n"
            f"executed the full {dec_t[-1]/86400.0:.2f}-day horizon of {args.cache} starting\n"
            f"from a cold INIT at t=0; only windows from {fmt_t(dec_t[i_start])} onward\n"
            f"({ndec - i_start} of {ndec} total windows) are shown below. This avoids the\n"
            f"early-run bias of INIT exploration and first-time library construction: by\n"
            f"this point the library already reflects everything learned over the earlier,\n"
            f"unreported days."
        )
    intro = build_intro(pol, s, r_sh, dsh, args.hold, tail_note=tail_note)
    events = build_events(pol, dec_t, reg_true, opt_arm, ndec, i_start=i_start)
    events.sort(key=lambda e: e[0])
    body = ["-" * 80]
    for t_h, header, text in events:
        body.append(f"[t={t_h:7.2f}h] {header}")
        body.append(text)
        body.append("")
    oracle_cum_tail = float(oracle_rewards[i_start:ndec].sum())
    summary = build_summary(pol, dec_t, reg_true, opt_arm, arms, rewards, i_start, ndec,
                             oracle_cum_tail)
    with open(txt_path, "w") as f:
        f.write(intro)
        f.write("\n".join(body))
        f.write(summary)
    print(f"[log] wrote {txt_path} ({len(events)} events)")

    # ---------------- plots ----------------
    make_plots(outdir, dec_t, reg_true, mode_hist, Z_hist, J_hist, G_hist, ell_hist,
               c_det_hist, arms, rewards, opt_arm, pol, oracle_rewards, i_start=i_start)

    print(f"[log] done -- outputs in {outdir}")


if __name__ == "__main__":
    main()
