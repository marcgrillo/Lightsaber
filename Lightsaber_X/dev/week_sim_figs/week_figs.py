"""Detailed figures for the two week-dataset simulations, from saved checkpoints."""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lightsaber_X package root, resolved relative to this file (dev/week_sim_figs/week_figs.py)
LX = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
CACHES = ["r1long", "r1freq"]
TITLES = {"r1long": "r1long  (calmer: longer stable stretches)",
          "r1freq": "r1freq  (harder: switches ~2x more often)"}
HOLD = 100.0  # s per decision

def poldir(tag): return os.path.join(LX, f"bandit_runs/cache_1w_{tag}/experiment/policies")
def load(tag, name): return dict(np.load(os.path.join(poldir(tag), name + ".npz"), allow_pickle=True))
def summary(tag):
    with open(os.path.join(LX, f"bandit_runs/cache_1w_{tag}/experiment/summary.csv")) as f:
        return {r["policy"]: r for r in csv.DictReader(f)}

def rolling(x, w):
    k = np.ones(w) / w
    return np.convolve(x.astype(float), k, mode="same")

TOP = [("RA-TS-F", "tab:green", 2.6, "-"),
       ("CG-ICLB-TT", "tab:blue", 1.5, "-"),
       ("CG-ICLB-R2", "tab:purple", 1.5, "-"),
       ("LI-TAR-UCB", "tab:orange", 1.5, "-"),
       ("RA-TS", "gray", 1.3, "--")]

# ============ FIGURE 1: cumulative regret over the week ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
for ax, tag in zip(axes, CACHES):
    orc = np.cumsum(load(tag, "Oracle")["rewards"])
    ndec = len(orc); days = np.arange(ndec) * HOLD / 86400.0
    for name, col, lw, ls in TOP:
        r = np.cumsum(load(tag, name)["rewards"])
        ax.plot(days, orc - r, color=col, lw=lw, ls=ls, label=name)
    ax.set_xlabel("time [days]"); ax.set_ylabel("cumulative regret vs Oracle")
    ax.set_title(TITLES[tag]); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
fig.suptitle("Regret accumulated over the 7-day run (lower = better; flatter = fewer mistakes)",
             fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_regret_curves.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)

# ============ FIGURE 2: final regret bars (all policies) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tag in zip(axes, CACHES):
    s = summary(tag)
    items = [(p, float(v["regret_vs_oracle"])) for p, v in s.items() if p != "Oracle"]
    items.sort(key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = ["tab:green" if n == "RA-TS-F" else ("lightgray" if n.startswith("Fixed") else "steelblue")
            for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    for yi, v in zip(y, vals):
        ax.text(v + 2, yi, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("total regret vs Oracle (lower = better)")
    ax.set_title(TITLES[tag]); ax.grid(alpha=0.3, axis="x")
fig.suptitle("Final regret of every method (RA-TS-F in green, fixed-controller baselines in gray)",
             fontsize=12, y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_final_bars.png"), dpi=130,
                                bbox_inches="tight"); plt.close(fig)

# ============ FIGURES 3-4: RA-TS-F internals per cache ============
for tag in CACHES:
    d = load(tag, tag and "RA-TS-F")
    orc = load(tag, "Oracle")
    ndec = len(d["rewards"]); days = np.arange(ndec) * HOLD / 86400.0
    true_reg = np.argmax(np.asarray(d["ctx"]), axis=1)      # dominant true regime per decision
    opt_arm = np.asarray(orc["arms"])                        # oracle's arm = best action
    mode = np.asarray(d["tar_mode"]); J = np.asarray(d["tar_J"])
    acq = (mode == 0).astype(float)                          # 0 = ACQUIRE, 1 = TRACK
    W = 120                                                  # rolling window (~3.3 h)
    # rolling fraction-optimal for a few methods
    def fopt(name):
        a = np.asarray(load(tag, name)["arms"]); return rolling((a == opt_arm), W)

    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    # (a) true regime schedule
    ax[0].step(days, true_reg, where="post", color="k", lw=0.8)
    ax[0].set_ylabel("true regime"); ax[0].set_yticks([0, 1, 2])
    ax[0].set_title(f"RA-TS-F on {TITLES[tag]}")
    nsw = int((np.diff(true_reg) != 0).sum())
    ax[0].text(0.99, 0.05, f"{nsw} regime switches", transform=ax[0].transAxes,
               ha="right", fontsize=9, color="dimgray")
    # (b) mode: rolling fraction of time spent re-identifying (ACQUIRE)
    ax[1].fill_between(days, rolling(acq, W), color="tab:orange", alpha=0.6)
    ax[1].set_ylabel("frac. in\nACQUIRE"); ax[1].set_ylim(0, max(0.3, rolling(acq, W).max() * 1.2))
    ax[1].text(0.99, 0.8, f"overall {acq.mean()*100:.1f}% in ACQUIRE (probe)",
               transform=ax[1].transAxes, ha="right", fontsize=9, color="chocolate")
    # (c) library size J (regimes created)
    ax[2].step(days, J, where="post", color="tab:blue", lw=1.6)
    ax[2].axhline(3, color="green", ls="--", lw=1, label="true # regimes = 3")
    ax[2].set_ylabel("library size J"); ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_ylim(0, J.max() + 1.5)
    cr = int((np.diff(np.r_[0, J]) > 0).sum()); last = int(np.where(np.diff(np.r_[0, J]) > 0)[0][-1])
    ax[2].text(0.99, 0.14, f"{cr} regimes created; last at day {days[last]:.1f}",
               transform=ax[2].transAxes, ha="right", fontsize=9, color="tab:blue")
    # (d) rolling fraction-optimal, RA-TS-F vs two rivals
    ax[3].plot(days, fopt("RA-TS-F"), color="tab:green", lw=2.0, label="RA-TS-F")
    ax[3].plot(days, fopt("CG-ICLB-TT"), color="tab:blue", lw=1.2, label="CG-ICLB-TT")
    ax[3].plot(days, fopt("LI-TAR-UCB"), color="tab:orange", lw=1.2, label="LI-TAR-UCB")
    ax[3].set_ylabel("frac. optimal\n(rolling)"); ax[3].set_xlabel("time [days]")
    ax[3].set_ylim(0, 1.02); ax[3].legend(fontsize=8, loc="lower right"); ax[3].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig3_internals_{tag}.png"), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)

print("saved to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
