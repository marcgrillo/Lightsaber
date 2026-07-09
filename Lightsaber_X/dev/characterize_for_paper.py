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
Paper characterization of reward(controller | regime).

Produces, for all 9 (regime, controller) cells:
  - raw 2s-block reward score statistics (the physics-meaningful -Z)
  - the bandit-space logistic reward  r = 1/(1+exp(-(raw-167.5)/2.5))
  - the per-regime advantage/regret matrix (gap of each controller to the best in that regime)
  - separation z-score of the diagonal winner as a function of hold time T_hold

Outputs CSV + figures into ./paper_characterization/.
Multi-seed for robust statistics. Deterministic given the seed list.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import bandit_rewards
from characterize_rewards import run_case

OUT = "paper_characterization"
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
DURATION = 200            # s per (cell, seed) -> 100 2s-blocks each
LOGIT_CENTER = 167.5
LOGIT_SCALE = 2.5
CTRL = ["C0", "C1", "C2"]


def logistic(raw):
    return 1.0 / (1.0 + np.exp(-(raw - LOGIT_CENTER) / LOGIT_SCALE))


def main():
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")

    blocks = {}  # (r,c) -> array of raw per-2s-block scores (all seeds concatenated)
    for r in range(3):
        for c in range(3):
            allb = []
            for sd in SEEDS:
                allb.append(np.array(run_case(r, c, DURATION, feat_names, w, seed=sd)))
            blocks[(r, c)] = np.concatenate(allb)
            print(f"R{r} C{c}: n_blocks={len(blocks[(r,c)])}", flush=True)

    raw_mean = np.array([[blocks[(r, c)].mean() for c in range(3)] for r in range(3)])
    raw_std  = np.array([[blocks[(r, c)].std(ddof=1) for c in range(3)] for r in range(3)])
    log_mean = np.array([[logistic(blocks[(r, c)]).mean() for c in range(3)] for r in range(3)])
    log_std  = np.array([[logistic(blocks[(r, c)]).std(ddof=1) for c in range(3)] for r in range(3)])

    # Advantage / per-step regret matrix (raw): gap of each controller to the best in its regime
    adv_raw = np.array([[raw_mean[r].max() - raw_mean[r, c] for c in range(3)] for r in range(3)])
    adv_log = np.array([[log_mean[r].max() - log_mean[r, c] for c in range(3)] for r in range(3)])

    # ---- CSV table ----
    csv_path = os.path.join(OUT, "reward_characterization.csv")
    with open(csv_path, "w") as f:
        f.write("regime,controller,raw_mean,raw_std,logistic_mean,logistic_std,regret_raw,regret_logistic,is_best\n")
        for r in range(3):
            best_c = int(np.argmax(raw_mean[r]))
            for c in range(3):
                f.write(f"R{r},{CTRL[c]},{raw_mean[r,c]:.4f},{raw_std[r,c]:.4f},"
                        f"{log_mean[r,c]:.5f},{log_std[r,c]:.5f},"
                        f"{adv_raw[r,c]:.4f},{adv_log[r,c]:.5f},{int(c==best_c)}\n")
    print(f"\nWrote {csv_path}")

    # ---- console summary ----
    print("\n=== RAW mean (rows=regime, cols=controller) ===")
    print("        C0          C1          C2     winner")
    for r in range(3):
        bc = int(np.argmax(raw_mean[r]))
        print(f"R{r}  " + "  ".join(f"{raw_mean[r,c]:10.3f}" for c in range(3)) + f"   C{bc}")

    print("\n=== LOGISTIC mean (bandit reward space) ===")
    print("        C0          C1          C2")
    for r in range(3):
        print(f"R{r}  " + "  ".join(f"{log_mean[r,c]:10.4f}" for c in range(3)))

    print("\n=== Diagonal separation: winner gap to runner-up and T_hold for 2-sigma ===")
    for r in range(3):
        order = np.argsort(raw_mean[r])[::-1]
        best_c, second_c = int(order[0]), int(order[1])
        gap = raw_mean[r, best_c] - raw_mean[r, second_c]
        pooled = np.sqrt((raw_std[r, best_c] ** 2 + raw_std[r, second_c] ** 2) / 2)
        n_need = (2 * pooled * np.sqrt(2) / gap) ** 2 if gap > 0 else np.inf
        thold = n_need * 2
        print(f"R{r}: winner=C{best_c} gap={gap:.4f}  pooled_std={pooled:.3f}  "
              f"2sigma needs ~{n_need:.0f} blocks (~{thold:.0f}s hold)")

    # ---- figure 1: raw overlay histograms ----
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    allv = np.concatenate([blocks[(r, c)] for r in range(3) for c in range(3)])
    lo, hi = allv.min(), allv.max()
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for r in range(3):
        ax = axes[r]
        for c in range(3):
            ax.hist(blocks[(r, c)], bins=40, range=(lo, hi), density=True, histtype="step",
                    linewidth=2, color=colors[c],
                    label=f"C{c} (raw μ={raw_mean[r,c]:.2f}, logistic μ={log_mean[r,c]:.3f})")
        ax.set_title(f"Regime {r}")
        ax.set_ylabel("density")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("raw reward score (-Z), 2s blocks")
    fig.suptitle("Per-controller reward distributions by regime (raw 2s-block scores)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "reward_by_regime_raw.png"), dpi=130)
    plt.close(fig)

    # ---- figure 2: mean reward bar charts (raw and logistic) with std error bars ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3); width = 0.25
    for c in range(3):
        axes[0].bar(x + (c - 1) * width, raw_mean[:, c], width, yerr=raw_std[:, c] / np.sqrt([len(blocks[(r, c)]) for r in range(3)]),
                    capsize=3, color=colors[c], label=f"C{c}")
        axes[1].bar(x + (c - 1) * width, log_mean[:, c], width, color=colors[c], label=f"C{c}")
    axes[0].set_title("Raw mean reward (SEM bars)"); axes[0].set_xticks(x); axes[0].set_xticklabels(["R0", "R1", "R2"]); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    axes[1].set_title("Logistic (bandit) mean reward"); axes[1].set_xticks(x); axes[1].set_xticklabels(["R0", "R1", "R2"]); axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mean_reward_bars.png"), dpi=130)
    plt.close(fig)

    # ---- save raw arrays for later reuse ----
    np.savez_compressed(os.path.join(OUT, "reward_blocks.npz"),
                        **{f"R{r}C{c}": blocks[(r, c)] for r in range(3) for c in range(3)},
                        seeds=np.array(SEEDS), duration=DURATION,
                        logit_center=LOGIT_CENTER, logit_scale=LOGIT_SCALE)
    print(f"\nSaved figures + reward_blocks.npz to {OUT}/")


if __name__ == "__main__":
    main()
