"""Instrumented RA-TS-F demo: show HOW/WHEN the drift alarm fires, with figures.
Run with the anaconda python. Saves PNGs next to this script."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
# Lightsaber_X package root, resolved relative to this file (dev/rats_demo_figs/rats_demo.py)
LX = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, LX); sys.path.insert(0, os.path.join(LX, "bandit"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import bandit_policies as bp

# larger fonts for a legible paper figure
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

# ---- environment: 3 recurring regimes, same level, rotated best arm, gap ~0.09 ----
RAW = {0: np.array([0.62, 0.53, 0.52]),      # best arm 0
       1: np.array([0.52, 0.62, 0.53]),      # best arm 1
       2: np.array([0.53, 0.52, 0.62])}      # best arm 2
SIG = 0.04
NU, H = 1.3 * SIG, 8.0 * SIG                  # same calibration as the experiment harness
NMIN = 3
# shape separation dsh (max-norm between centered regime shapes)
cent = {z: RAW[z] - RAW[z].mean() for z in RAW}
dsh = min(np.max(np.abs(cent[i] - cent[j])) for i in RAW for j in RAW if i < j)


def run(schedule, seed=0, extra_switch=None):
    """schedule: array of true regime per step. Returns per-step instrumentation."""
    T = len(schedule)
    rng = np.random.RandomState(seed)
    pol = bp.RATS(3, sigma=SIG, mu0=0.557, tau0=0.5, n_min=NMIN, n_new=8,
                  match_tol=0.8 * dsh, match_margin=0.4 * dsh, nu=NU, h=H,
                  explore_floor=True, seed=1)
    rec = {k: [] for k in ("t", "z", "best", "a", "r", "mode", "g", "istrk",
                           "e", "alarm", "J", "match", "new", "nbuf")}
    for t in range(T):
        z = int(schedule[t])
        pre_mode = pol.mode
        pre_base = None if pol._base is None else pol._base.copy()
        a = pol.choose(t)
        r = float(RAW[z][a] + rng.normal(0, SIG))
        nlog = len(pol.diag_log)
        pol.update(a, r)
        # new diag events this step
        evs = pol.diag_log[nlog:]
        alarm = any(e[1] == "alarm" for e in evs)
        match = any(e[1] == "match" for e in evs)
        new = any(e[1] == "new_regime" for e in evs)
        istrk = (pre_mode == pol.TRACK and pre_base is not None)
        e = (r - pre_base[a]) if istrk else np.nan
        rec["t"].append(t); rec["z"].append(z); rec["best"].append(int(np.argmax(RAW[z])))
        rec["a"].append(a); rec["r"].append(r); rec["mode"].append(pre_mode)
        rec["g"].append(pol._Zcur); rec["istrk"].append(istrk); rec["e"].append(e)
        rec["alarm"].append(alarm); rec["J"].append(len(pol.N))
        rec["match"].append(match); rec["new"].append(new)
        rec["nbuf"].append(float(pol._n.sum()))
    for k in rec:
        rec[k] = np.array(rec[k], float if k in ("r", "e", "nbuf") else int)
    return rec, pol


def reconstruct_G(rec):
    """Rebuild the played-arm CUSUM G_t for plotting (mirrors pol.G exactly)."""
    G = 0.0; out = []
    for i in range(len(rec["t"])):
        if rec["istrk"][i]:
            G = max(0.0, G + abs(rec["e"][i]) - NU)
        else:
            G = 0.0
        out.append(G)
        if rec["alarm"][i]:
            G = 0.0                      # reset after alarm
    return np.array(out)


def shade_acquire(ax, rec, label=True):
    m = rec["mode"] == bp.RATS.ACQUIRE
    t = rec["t"]; first = True
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            ax.axvspan(t[i] - 0.5, t[j - 1] + 0.5, color="orange", alpha=0.25,
                       label=("ACQUIRE (probe)" if (label and first) else None))
            first = False; i = j
        else:
            i += 1


# ============================ FIG 1 + FIG 2: normal alarms ============================
EP = 350
sched = np.concatenate([np.full(EP, z) for z in [0, 1, 2, 0, 1, 2]])   # 6 episodes
rec, pol = run(sched, seed=0)
G = reconstruct_G(rec)
t = rec["t"]
alarms = t[rec["alarm"] == 1]
print("dsh=%.4f  match_tol=%.4f  nu=%.4f  h=%.4f" % (dsh, 0.8 * dsh, NU, H))
print("regimes discovered J=%d (true 3)" % len(pol.N))
print("alarm times:", alarms.tolist())
print("true switch times:", [EP * k for k in range(1, 6)])

fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
# (a) true regime vs tracked
ax[0].step(t, rec["z"], where="post", color="k", lw=2.4, label="true regime z(t)")
gp = rec["g"].astype(float); gp[gp < 0] = np.nan
ax[0].step(t, gp, where="post", color="tab:blue", lw=1.8, ls="--", label="RA-TS tracked regime g(t)")
ax[0].set_ylabel("regime id"); ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
ax[0].legend(loc="upper right", ncol=2); ax[0].set_title(
    "RA-TS-F on a clean toy environment (3 recurring regimes, discrete switches): "
    "identification, drift alarms and recovery")
for a0 in ax:
    shade_acquire(a0, rec, label=(a0 is ax[1]))
    for s in [EP * k for k in range(1, 6)]:
        a0.axvline(s, color="green", ls=":", lw=1, alpha=0.7)
# (b) played vs best arm
ax[1].step(t, rec["best"], where="post", color="green", lw=2.6, alpha=0.6, label="best arm (truth)")
ax[1].step(t, rec["a"], where="post", color="tab:red", lw=1.4, label="arm played")
ax[1].set_ylabel("arm"); ax[1].set_yticks([0, 1, 2]); ax[1].legend(loc="upper right", ncol=2)
# (c) residual |e_t|
ax[2].plot(t, np.abs(rec["e"]), color="gray", lw=0.9, label=r"$|e_t|=|Y_t-\mathrm{base}[a]|$")
ax[2].axhline(NU, color="purple", ls="--", lw=1.6, label=r"$\nu$ (CUSUM slack)")
ax[2].set_ylabel("|residual|"); ax[2].legend(loc="upper right", ncol=2); ax[2].set_ylim(0, 0.25)
# (d) CUSUM G_t
ax[3].plot(t, G, color="tab:orange", lw=1.8, label=r"CUSUM $G_t$ (played arm)")
ax[3].axhline(H, color="red", ls="--", lw=1.6, label=r"$h$ (alarm threshold)")
for s in alarms:
    ax[3].plot(s, H, "rv", ms=11)
ax[3].scatter([], [], marker="v", color="red", label="ALARM ($G_t\\geq h$)")
ax[3].set_ylabel(r"$G_t$"); ax[3].set_xlabel("step t"); ax[3].legend(loc="upper right", ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "rats_fig1_overview.png"), dpi=130); plt.close()

# FIG 2: zoom on the first drift (true switch at t=EP)
sw = EP
lo, hi = sw - 12, sw + 40
sl = (t >= lo) & (t <= hi)
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ax[0].step(t[sl], rec["best"][sl], where="post", color="green", lw=2.5, alpha=0.6, label="best arm (truth)")
ax[0].step(t[sl], rec["a"][sl], where="post", color="tab:red", lw=1.4, label="arm played")
ax[0].axvline(sw, color="green", ls=":", lw=1.5); ax[0].set_ylabel("arm"); ax[0].set_yticks([0, 1, 2])
ax[0].legend(loc="center right", fontsize=9)
ax[0].set_title("Zoom on one drift: regime %d->%d at t=%d  (detector OFF in ACQUIRE)" %
                (sched[sw - 1], sched[sw], sw))
ax[1].plot(t[sl], np.abs(rec["e"][sl]), "o-", color="gray", ms=3, lw=0.8, label=r"$|e_t|$")
ax[1].axhline(NU, color="purple", ls="--", lw=1.2, label=r"$\nu$")
ax[1].axvline(sw, color="green", ls=":", lw=1.5); ax[1].set_ylabel("|residual|")
ax[1].legend(loc="upper right", fontsize=9)
ax[2].plot(t[sl], G[sl], "o-", color="tab:orange", ms=3, lw=1.2, label=r"$G_t$")
ax[2].axhline(H, color="red", ls="--", lw=1.2, label=r"$h$")
al = alarms[(alarms >= lo) & (alarms <= hi)]
for s in al:
    ax[2].plot(s, H, "rv", ms=11)
ax[2].axvline(sw, color="green", ls=":", lw=1.5, label="true switch")
for a0 in ax:
    shade_acquire(a0, {"mode": rec["mode"][sl], "t": t[sl]}, label=(a0 is ax[2]))
ax[2].set_ylabel(r"$G_t$"); ax[2].set_xlabel("step t"); ax[2].legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "rats_fig2_alarm_zoom.png"), dpi=120); plt.close()

# annotated numeric trace of that first alarm
print("\n--- trace around first drift (t=%d) ---" % sw)
print(" t | mode    | a | reward | base[a] | |e|   | G_t   | event")
for i in np.where((t >= sw - 2) & (t <= sw + 16))[0]:
    md = "ACQUIRE" if rec["mode"][i] == bp.RATS.ACQUIRE else "TRACK"
    ev = "ALARM" if rec["alarm"][i] else ("match" if rec["match"][i] else ("NEW" if rec["new"][i] else ""))
    baa = pol._base  # not per-step; recompute base is snapshot -> approx via e
    be = rec["r"][i] - rec["e"][i] if rec["istrk"][i] else np.nan
    ee = abs(rec["e"][i]) if rec["istrk"][i] else np.nan
    print("%3d| %-7s | %d | %.3f  | %s | %s | %.3f | %s" %
          (t[i], md, rec["a"][i], rec["r"][i],
           ("%.3f" % be) if rec["istrk"][i] else "  -  ",
           ("%.3f" % ee) if rec["istrk"][i] else "  -  ",
           G[i], ev))

# ============================ FIG 3: drift DURING exploration (straddle) ============================
# Pass 1: find an ACQUIRE start; Pass 2: force a true switch 4 steps into that probe.
rec1, _ = run(sched, seed=0)
acq_starts = [t[i] for i in range(1, len(t))
              if rec1["mode"][i] == bp.RATS.ACQUIRE and rec1["mode"][i - 1] == bp.RATS.TRACK]
t_probe = int([s for s in acq_starts if s > 400][0])       # an ACQUIRE well into the run
sched2 = sched.copy()
t_str = t_probe + 4                                        # switch lands mid-9-step probe
# force regime at t_str.. to a DIFFERENT regime than what the probe began in
z_before = sched2[t_probe]
z_new = (z_before + 1) % 3
# apply the forced switch until the next scheduled boundary
nb = ((t_str // EP) + 1) * EP
sched2[t_str:nb] = z_new
rec3, pol3 = run(sched2, seed=0)
G3 = reconstruct_G(rec3); t3 = rec3["t"]
lo, hi = t_probe - 8, t_probe + 45
sl = (t3 >= lo) & (t3 <= hi)
al3 = t3[(rec3["alarm"] == 1)]; al3 = al3[(al3 >= lo) & (al3 <= hi)]
mt3 = t3[(rec3["match"] == 1) | (rec3["new"] == 1)]; mt3 = mt3[(mt3 >= lo) & (mt3 <= hi)]
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
ax[0].step(t3[sl], rec3["z"][sl], where="post", color="k", lw=2, label="true regime")
gp = rec3["g"][sl].astype(float); gp[gp < 0] = np.nan
ax[0].step(t3[sl], gp, where="post", color="tab:blue", lw=1.4, ls="--", label="tracked regime g")
ax[0].axvline(t_str, color="green", ls=":", lw=1.5, label="drift (mid-probe)")
ax[0].set_ylabel("regime"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Drift DURING exploration: switch at t=%d lands inside the 9-step ACQUIRE probe" % t_str)
ax[1].step(t3[sl], rec3["best"][sl], where="post", color="green", lw=2.5, alpha=0.6, label="best arm")
ax[1].step(t3[sl], rec3["a"][sl], where="post", color="tab:red", lw=1.2, label="arm played")
ax[1].axvline(t_str, color="green", ls=":", lw=1.5); ax[1].set_ylabel("arm"); ax[1].set_yticks([0, 1, 2])
ax[1].legend(loc="center right", fontsize=8)
ax[2].plot(t3[sl], G3[sl], "o-", color="tab:orange", ms=3, lw=1.2, label=r"$G_t$")
ax[2].axhline(H, color="red", ls="--", lw=1.2, label=r"$h$")
for s in al3:
    ax[2].plot(s, H, "rv", ms=11)
ax[2].axvline(t_str, color="green", ls=":", lw=1.5)
for a0 in ax:
    shade_acquire(a0, {"mode": rec3["mode"][sl], "t": t3[sl]}, label=(a0 is ax[2]))
for s in mt3:
    ax[2].annotate("decide", (s, H * 0.2), fontsize=8, color="navy", ha="center")
ax[2].set_ylabel(r"$G_t$"); ax[2].set_xlabel("step t"); ax[2].legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "rats_fig3_straddle.png"), dpi=120); plt.close()
print("\nstraddle: probe began at t=%d in regime %d, forced switch to regime %d at t=%d"
      % (t_probe, z_before, z_new, t_str))
print("straddle: J before=%d after=%d ; decisions near probe:" % (rec1["J"][-1], len(pol3.N)),
      [(int(s), "match" if rec3["match"][t3 == s][0] else "new") for s in mt3])
print("saved: rats_fig1_overview.png, rats_fig2_alarm_zoom.png, rats_fig3_straddle.png")
