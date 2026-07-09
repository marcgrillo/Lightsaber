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
Plot the environment regime schedule W(t) = (W_R0, W_R1, W_R2) from a noise cache
manifest, split into several figures of `--periods` diurnal cycles each (default 2),
so a multi-day/week/month horizon is readable instead of one crowded plot.

Usage:
    python bandit_plot_regime.py --cache bandit_runs/cache_1w
    python bandit_plot_regime.py --cache bandit_runs/cache_1w --periods 2 --out bandit_runs/cache_1w/regime_figs
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"R0": "tab:green", "R1": "tab:orange", "R2": "tab:red"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="noise cache dir (has manifest.npz)")
    ap.add_argument("--periods", type=float, default=2.0, help="diurnal periods per figure")
    ap.add_argument("--out", default=None, help="output dir (default <cache>/regime_figs)")
    ap.add_argument("--max-figs", type=int, default=12, help="safety cap on number of figures")
    ap.add_argument("--zoom-spikes", action="store_true",
                     help="also plot each R1 (microseism) episode individually, minute-scale")
    ap.add_argument("--zoom-margin", type=float, default=0.5,
                     help="margin around each R1 episode as a fraction of its own duration")
    args = ap.parse_args()

    m = np.load(os.path.join(args.cache, "manifest.npz"), allow_pickle=True)
    period = float(m["period"]); horizon = float(m["horizon"])
    Wsec = np.asarray(m["Wsec"]); tsec = np.asarray(m["tsec"])   # (3, N+1), (N+1,) at 1 Hz
    outdir = args.out or os.path.join(args.cache, "regime_figs")
    os.makedirs(outdir, exist_ok=True)

    win = args.periods * period
    n_fig = int(np.ceil(horizon / win))
    if n_fig > args.max_figs:
        print(f"[warn] {n_fig} figures requested (horizon={horizon/86400:.1f}d, "
              f"window={win/86400:.2f}d) -- capping at --max-figs={args.max_figs}")
        n_fig = args.max_figs

    sev = 1.0 * Wsec[1] + 2.0 * Wsec[2]                 # 0=R0, 1=R1, 2=R2 severity proxy
    dom = np.argmax(Wsec, axis=0)                       # dominant regime per second

    print(f"[plot] cache={args.cache}  horizon={horizon/86400:.2f}d  period={period/3600:.1f}h  "
          f"window={win/3600:.1f}h ({args.periods} periods)  -> {n_fig} figure(s)")

    for k in range(n_fig):
        t0, t1 = k * win, min((k + 1) * win, horizon)
        i0, i1 = int(t0), int(t1) + 1
        t = (tsec[i0:i1] - t0) / 3600.0                 # hours from window start

        fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                gridspec_kw=dict(height_ratios=[2.2, 1]))
        for j, name in enumerate(["R0", "R1", "R2"]):
            ax[0].plot(t, Wsec[j, i0:i1], color=COL[name], lw=1.1, label=name)
        ax[0].set_ylabel("regime weight $W_i(t)$")
        ax[0].set_ylim(-0.05, 1.05)
        ax[0].legend(loc="upper right", ncol=3, fontsize=9)
        ax[0].grid(alpha=0.3)
        ax[0].set_title(f"{os.path.basename(args.cache)} -- regime schedule, "
                         f"day {t0/86400:.2f}-{t1/86400:.2f} of {horizon/86400:.1f}")

        # shade the dominant-regime background on the severity panel
        d = dom[i0:i1]
        change = np.where(np.diff(d) != 0)[0] + 1
        bounds = np.concatenate(([0], change, [len(d) - 1]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            ax[1].axvspan(t[a], t[b], color=COL[["R0", "R1", "R2"][d[a]]], alpha=0.15, lw=0)
        ax[1].plot(t, sev[i0:i1], "k", lw=0.8)
        ax[1].set_ylabel("severity\n($W_{R1}+2W_{R2}$)")
        ax[1].set_xlabel("time within window [h]")
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        fp = os.path.join(outdir, f"regime_{k:02d}_day{t0/86400:.2f}-{t1/86400:.2f}.png")
        plt.savefig(fp, dpi=130); plt.close()
        print(f"  saved {fp}")

    if args.zoom_spikes:
        is_r1 = (dom == 1).astype(int)
        d = np.diff(np.concatenate(([0], is_r1, [0])))
        starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
        print(f"[plot] {len(starts)} R1 episode(s) -> zoomed figures")
        for i, (a, b) in enumerate(zip(starts, ends)):
            dur = b - a
            margin = max(30, int(args.zoom_margin * dur))
            i0, i1 = max(0, a - margin), min(len(tsec) - 1, b + margin)
            t = (tsec[i0:i1] - a)                            # seconds from episode start
            fig, ax = plt.subplots(figsize=(9, 4))
            for j, name in enumerate(["R0", "R1", "R2"]):
                ax.plot(t, Wsec[j, i0:i1], color=COL[name], lw=1.4, label=name)
            ax.axvspan(0, dur, color=COL["R1"], alpha=0.12, lw=0)
            ax.set_ylim(-0.05, 1.05); ax.set_xlabel("time from episode start [s]")
            ax.set_ylabel("regime weight $W_i(t)$")
            ax.set_title(f"{os.path.basename(args.cache)} -- R1 episode {i} "
                         f"(t0={a/3600:.2f}h, duration={dur}s)")
            ax.legend(loc="upper right", ncol=3, fontsize=9); ax.grid(alpha=0.3)
            plt.tight_layout()
            fp = os.path.join(outdir, f"r1_episode_{i:02d}_dur{dur}s.png")
            plt.savefig(fp, dpi=130); plt.close()
            print(f"  saved {fp}")


if __name__ == "__main__":
    main()
