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
on the fast engine. Regenerates reward_characterization.csv + the two figures used in the paper.

Per (regime, controller): run 360s, continuous-filter reward (StreamReward), discard a 60s
warm-up, keep the settled per-2s-block scores. Multi-seed for statistics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
import fast_engine as fe
from reward_stream import StreamReward

OUT = "paper_characterization"
os.makedirs(OUT, exist_ok=True)
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
    blocks = {}
    for R in range(3):
        for C in range(3):
            allb = []
            for s in SEEDS:
                p, c = fe.run_fast(R, C, DUR, seed=s, sens_scale=SENS[R])
                sc = StreamReward(fn, w).score(p, c)      # continuous filtering
                allb.append(sc[WARM_BLOCKS:])             # settled blocks only
            blocks[(R, C)] = np.concatenate(allb)
            print(f"R{R} C{C}: n_blocks={len(blocks[(R, C)])}", flush=True)

    raw_mean = np.array([[blocks[(R, C)].mean() for C in range(3)] for R in range(3)])
    raw_std = np.array([[blocks[(R, C)].std(ddof=1) for C in range(3)] for R in range(3)])
    log_mean = np.array([[logistic(blocks[(R, C)]).mean() for C in range(3)] for R in range(3)])
    adv_raw = np.array([[raw_mean[R].max() - raw_mean[R, C] for C in range(3)] for R in range(3)])

    with open(os.path.join(OUT, "reward_characterization.csv"), "w") as f:
        f.write("regime,controller,raw_mean,raw_std,logistic_mean,regret_raw,is_best\n")
        for R in range(3):
            bc = int(np.argmax(raw_mean[R]))
            for C in range(3):
                f.write(f"R{R},{CTRL[C]},{raw_mean[R,C]:.4f},{raw_std[R,C]:.4f},"
                        f"{log_mean[R,C]:.5f},{adv_raw[R,C]:.4f},{int(C==bc)}\n")

    print("\n=== RAW mean (corrected, continuous reward) ===")
    print("        C0        C1        C2    winner")
    for R in range(3):
        bc = int(np.argmax(raw_mean[R]))
        print(f"R{R}  " + "  ".join(f"{raw_mean[R,c]:8.3f}" for c in range(3)) + f"   C{bc}")
    print("\n=== LOGISTIC mean ===")
    for R in range(3):
        print(f"R{R}  " + "  ".join(f"{log_mean[R,c]:.4f}" for c in range(3)))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # fig 1: distributions by regime
    allv = np.concatenate([blocks[(R, C)] for R in range(3) for C in range(3)])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for R in range(3):
        ax = axes[R]
        for C in range(3):
            ax.hist(blocks[(R, C)], bins=50, range=(lo, hi), density=True, histtype="step",
                    lw=2, color=colors[C], label=f"C{C} (raw mu={raw_mean[R,C]:.2f})")
        ax.set_title(f"Regime {R}"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("raw reward score, 2s blocks (corrected, continuous-filter)")
    fig.suptitle("Per-controller reward distributions by regime")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "reward_by_regime_raw.png"), dpi=130); plt.close(fig)

    # fig 2: mean reward bars (raw + logistic)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3); width = 0.25
    nbl = np.array([[len(blocks[(R, C)]) for C in range(3)] for R in range(3)])
    for C in range(3):
        axes[0].bar(x + (C-1)*width, raw_mean[:, C], width, yerr=raw_std[:, C]/np.sqrt(nbl[:, C]),
                    capsize=3, color=colors[C], label=f"C{C}")
        axes[1].bar(x + (C-1)*width, log_mean[:, C], width, color=colors[C], label=f"C{C}")
    axes[0].set_title("Raw mean reward (SEM bars)"); axes[1].set_title("Logistic (bandit) mean reward")
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(["R0", "R1", "R2"]); a.legend(); a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mean_reward_bars.png"), dpi=130); plt.close(fig)
    print("\nSaved corrected sec.5 figures to", OUT)


if __name__ == "__main__":
    main()
