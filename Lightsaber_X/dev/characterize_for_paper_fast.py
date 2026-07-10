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
Paper characterization (sec.5) with the CORRECTED reward (full-window, continuous filtering)
on the fast engine. Regenerates reward_characterization.csv + the SINGLE reward-landscape figure
used in the paper (per-regime histograms + a logistic-reward bar row with SEM error bars).

Per (regime, controller): run 360s, continuous-filter reward (StreamReward), discard a 60s
warm-up, keep the settled per-2s-block scores. Multi-seed for statistics; the bar-row error
bars are the standard error over the NSEED independent noise realisations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
import fast_engine as fe
from reward_stream import StreamReward

# larger fonts for a legible paper figure
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

OUT = "paper_characterization"
FIGS = os.path.join("paper", "figs")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
SEEDS = [1, 2, 3, 4, 5]
DUR = 360
WARM_BLOCKS = 30           # 60s warm-up discard
LOGIT_C, LOGIT_S = 167.5, 2.5
SENS = {0: 120.0, 1: 12.0, 2: 0.25}
CTRL = ["C0", "C1", "C2"]


def logistic(x):
    return 1.0/(1.0 + np.exp(-(x - LOGIT_C)/LOGIT_S))


def main():
    fn, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    blocks = {}                       # (R,C) -> all settled raw blocks (for histograms)
    log_seed = {}                     # (R,C) -> per-seed mean logistic reward (for bar SEM)
    for R in range(3):
        for C in range(3):
            allb = []; seedmeans = []
            for s in SEEDS:
                p, c = fe.run_fast(R, C, DUR, seed=s, sens_scale=SENS[R])
                sc = StreamReward(fn, w).score(p, c)[WARM_BLOCKS:]   # settled blocks, continuous filter
                allb.append(sc); seedmeans.append(float(logistic(sc).mean()))
            blocks[(R, C)] = np.concatenate(allb)
            log_seed[(R, C)] = np.array(seedmeans)                   # (NSEED,)
            print(f"R{R} C{C}: n_blocks={len(blocks[(R, C)])}", flush=True)

    raw_mean = np.array([[blocks[(R, C)].mean() for C in range(3)] for R in range(3)])
    raw_std = np.array([[blocks[(R, C)].std(ddof=1) for C in range(3)] for R in range(3)])
    log_mean = np.array([[log_seed[(R, C)].mean() for C in range(3)] for R in range(3)])
    log_sem = np.array([[log_seed[(R, C)].std(ddof=1) / np.sqrt(len(SEEDS)) for C in range(3)]
                        for R in range(3)])
    adv_raw = np.array([[raw_mean[R].max() - raw_mean[R, C] for C in range(3)] for R in range(3)])

    with open(os.path.join(OUT, "reward_characterization.csv"), "w") as f:
        f.write("regime,controller,raw_mean,raw_std,logistic_mean,logistic_sem,regret_raw,is_best\n")
        for R in range(3):
            bc = int(np.argmax(raw_mean[R]))
            for C in range(3):
                f.write(f"R{R},{CTRL[C]},{raw_mean[R,C]:.4f},{raw_std[R,C]:.4f},"
                        f"{log_mean[R,C]:.5f},{log_sem[R,C]:.5f},{adv_raw[R,C]:.4f},{int(C==bc)}\n")

    print("\n=== RAW mean (corrected, continuous reward) ===")
    print("        C0        C1        C2    winner")
    for R in range(3):
        bc = int(np.argmax(raw_mean[R]))
        print(f"R{R}  " + "  ".join(f"{raw_mean[R,c]:8.3f}" for c in range(3)) + f"   C{bc}")
    print("\n=== LOGISTIC mean +/- SEM (over %d seeds) ===" % len(SEEDS))
    for R in range(3):
        print(f"R{R}  " + "  ".join(f"{log_mean[R,c]:.4f}+/-{log_sem[R,c]:.4f}" for c in range(3)))

    # ---- SINGLE reward-landscape figure: 3 per-regime histograms (top) + logistic bar row ----
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    allv = np.concatenate([blocks[(R, C)] for R in range(3) for C in range(3)])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.12)
    # top: histograms, one panel per regime, shared axes
    axh = []
    for R in range(3):
        ax = fig.add_subplot(gs[0, R], sharey=(axh[0] if axh else None),
                             sharex=(axh[0] if axh else None))
        for C in range(3):
            ax.hist(blocks[(R, C)], bins=50, range=(lo, hi), density=True, histtype="step",
                    lw=2.2, color=colors[C], label=f"C{C}")
        ax.set_title(f"Regime R{R}"); ax.grid(alpha=0.3)
        ax.set_xlabel("raw reward score (2 s blocks)")
        if R == 0:
            ax.set_ylabel("density"); ax.legend(loc="upper left", frameon=True)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        axh.append(ax)
    # bottom: logistic (bandit) reward bars with SEM error bars over seeds
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(3); width = 0.26
    for C in range(3):
        axb.bar(x + (C - 1) * width, log_mean[:, C], width, yerr=log_sem[:, C],
                capsize=5, color=colors[C], label=f"C{C}", error_kw=dict(lw=1.6))
    axb.set_xticks(x); axb.set_xticklabels(["R0", "R1", "R2"])
    axb.set_ylabel("logistic (bandit) reward")
    axb.set_title(f"Mean logistic reward per regime and controller "
                  f"(error bars: SEM over {len(SEEDS)} noise realisations)")
    axb.legend(loc="upper center", ncol=3); axb.set_ylim(0, 0.66); axb.grid(alpha=0.3, axis="y")
    fig.suptitle("Reward landscape: per-controller distributions (top) and mean bandit reward (bottom)",
                 fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for d in (OUT, FIGS):
        fig.savefig(os.path.join(d, "reward_landscape.png"), dpi=140)
    plt.close(fig)
    print("\nSaved reward_landscape.png to", OUT, "and", FIGS)


if __name__ == "__main__":
    main()
