# --- repo path bootstrap (run as `python dev/paper_bandit_figs.py` from Lightsaber_X/) ---
import os as _os, sys as _sys
_DEV = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_DEV)
for _p in (_ROOT, _os.path.join(_ROOT, 'bandit'), _DEV):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)
# --- end bootstrap ---

"""Bandit figures for paper/main.tex: RA-TS variants + EXTERNAL baselines only
(the in-house TAR-UCB / LI-TAR-UCB / CG-ICLB predecessors are excluded -- the paper
presents RA-TS as the method and compares against reference/literature policies).

Reads saved checkpoints; writes directly into paper/figs/.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("paper", "figs")
HOLD = 100.0
TITLES = {"r1long": "r1long  (calmer: longer stable stretches)",
          "r1freq": "r1freq  (harder: switches ~2x more often)"}

def load(cache_dir, name):
    return dict(np.load(os.path.join(cache_dir, "experiment", "policies", name + ".npz"),
                        allow_pickle=True))

def rolling(x, w):
    return np.convolve(x.astype(float), np.ones(w) / w, mode="same")

# line style per policy: (color, lw, ls)
STYLE = [("RA-TS-F",    "tab:green",  2.6, "-"),
         ("RA-TS",      "gray",       1.4, "--"),
         ("Rule-based", "tab:brown",  1.4, "-"),
         ("D-UCB",      "tab:red",    1.4, "-"),
         ("Thompson",   "tab:purple", 1.4, "-"),
         ("Fixed-C0",   "silver",     1.2, ":")]

# ============ FIG 1: week regret curves (both caches) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, tag in zip(axes, ["r1long", "r1freq"]):
    cd = f"bandit_runs/cache_1w_{tag}"
    orc = np.cumsum(load(cd, "Oracle")["rewards"])
    days = np.arange(len(orc), dtype=np.int64) * HOLD / 86400.0
    for name, col, lw, ls in STYLE:
        r = np.cumsum(load(cd, name)["rewards"])
        ax.plot(days, orc - r, color=col, lw=lw, ls=ls, label=name)
    ax.set_xlabel("time [days]"); ax.set_ylabel("cumulative regret vs Oracle")
    ax.set_title(TITLES[tag]); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
fig.suptitle("Regret accumulated over the 7-day run (lower = better; flatter = fewer mistakes)",
             fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "week_regret_curves.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)
print("week_regret_curves.png")

# ============ FIG 2: RA-TS-F internals on r1long ============
tag = "r1long"; cd = f"bandit_runs/cache_1w_{tag}"
d = load(cd, "RA-TS-F"); orc = load(cd, "Oracle")
ndec = len(d["rewards"]); days = np.arange(ndec, dtype=np.int64) * HOLD / 86400.0
true_reg = np.argmax(np.asarray(d["ctx"]), axis=1)
opt_arm = np.asarray(orc["arms"])
mode = np.asarray(d["tar_mode"]); J = np.asarray(d["tar_J"])
acq = (mode == 0).astype(float)
W = 120

def fopt(name):
    a = np.asarray(load(cd, name)["arms"]); return rolling((a == opt_arm), W)

fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
ax[0].step(days, true_reg, where="post", color="k", lw=0.8)
ax[0].set_ylabel("true regime"); ax[0].set_yticks([0, 1, 2])
ax[0].set_title(f"RA-TS-F on {TITLES[tag]}")
nsw = int((np.diff(true_reg) != 0).sum())
ax[0].text(0.99, 0.05, f"{nsw} regime switches", transform=ax[0].transAxes,
           ha="right", fontsize=9, color="dimgray")
ax[1].fill_between(days, rolling(acq, W), color="tab:orange", alpha=0.6)
ax[1].set_ylabel("frac. in\nACQUIRE"); ax[1].set_ylim(0, max(0.3, rolling(acq, W).max() * 1.2))
ax[1].text(0.99, 0.8, f"overall {acq.mean()*100:.1f}% in ACQUIRE (probe)",
           transform=ax[1].transAxes, ha="right", fontsize=9, color="chocolate")
ax[2].step(days, J, where="post", color="tab:blue", lw=1.6)
ax[2].axhline(3, color="green", ls="--", lw=1, label="true # regimes = 3")
ax[2].set_ylabel("library size J"); ax[2].legend(fontsize=8, loc="upper left")
ax[2].set_ylim(0, J.max() + 1.5)
cr = int((np.diff(np.r_[0, J]) > 0).sum()); last = int(np.where(np.diff(np.r_[0, J]) > 0)[0][-1])
ax[2].text(0.99, 0.14, f"{cr} regimes created; last at day {days[last]:.1f}",
           transform=ax[2].transAxes, ha="right", fontsize=9, color="tab:blue")
ax[3].plot(days, fopt("RA-TS-F"), color="tab:green", lw=2.0, label="RA-TS-F")
ax[3].plot(days, fopt("Rule-based"), color="tab:brown", lw=1.2, label="Rule-based")
ax[3].plot(days, fopt("D-UCB"), color="tab:red", lw=1.2, label="D-UCB")
ax[3].set_ylabel("frac. optimal\n(rolling)"); ax[3].set_xlabel("time [days]")
ax[3].set_ylim(0, 1.02); ax[3].legend(fontsize=8, loc="lower right"); ax[3].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "week_internals_r1long.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)
print("week_internals_r1long.png")

# ============ FIG 3: six-month cumulative reward + regret ============
cd6 = "bandit_runs/cache_6mo"
POL6 = ["Oracle", "RA-TS-F", "Rule-based", "D-UCB", "Thompson",
        "Fixed-C0", "Fixed-C1", "Fixed-C2"]
data6 = {n: load(cd6, n) for n in POL6}
ndec6 = min(len(v["rewards"]) for v in data6.values())
days6 = np.arange(ndec6, dtype=np.int64) * HOLD / 86400.0
S6 = {"Oracle":     ("tab:blue",   2.4, "-"),
      "RA-TS-F":    ("tab:green",  2.2, "-"),
      "Rule-based": ("tab:brown",  1.3, "-"),
      "D-UCB":      ("tab:red",    1.3, "-"),
      "Thompson":   ("tab:purple", 1.3, "-"),
      "Fixed-C0":   ("dimgray",    1.1, "--"),
      "Fixed-C1":   ("darkgray",   1.1, "--"),
      "Fixed-C2":   ("silver",     1.1, "--")}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
oc = np.cumsum(data6["Oracle"]["rewards"][:ndec6])
for n in POL6:
    col, lw, ls = S6[n]
    r = np.cumsum(data6[n]["rewards"][:ndec6])
    ax[0].plot(days6, r, color=col, lw=lw, ls=ls, label=n)
    if n != "Oracle":
        ax[1].plot(days6, oc - r, color=col, lw=lw, ls=ls, label=n)
ax[0].set_xlabel("time [days]"); ax[0].set_ylabel("cumulative reward")
ax[0].set_title("Cumulative reward"); ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("time [days]"); ax[1].set_ylabel("cumulative regret vs Oracle")
ax[1].set_title("Regret vs Oracle"); ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sixmo_cumreward_regret.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)
print("sixmo_cumreward_regret.png")

# ============ FIG 4: six-month timelines ============
ctx = np.asarray(data6["RA-TS-F"]["ctx"])[:ndec6]
sev = 1.0 * ctx[:, 1] + 2.0 * ctx[:, 2]
show = ["Oracle", "RA-TS-F", "Rule-based"]
fig, axes = plt.subplots(len(show) + 1, 1, figsize=(13, 1.6 * (len(show) + 1)), sharex=True)
axes[0].plot(days6, sev, "k", lw=0.8); axes[0].set_ylabel("severity")
axes[0].set_title("regime severity and controller choices"); axes[0].grid(alpha=0.3)
for a, name in zip(axes[1:], show):
    arms = np.asarray(data6[name]["arms"])[:ndec6]
    a.step(days6, arms, where="post", lw=0.7)
    a.set_ylabel(name); a.set_yticks([0, 1, 2]); a.set_yticklabels(["C0", "C1", "C2"]); a.grid(alpha=0.3)
axes[-1].set_xlabel("time [days]")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sixmo_timelines.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)
print("sixmo_timelines.png")
