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
Publication figures for paper/li_tar_ucb_draft.tex, generated from the actual
1-week experiment data (bandit_runs/cache_1w_r1long, hold=100s, logistic reward).

Outputs (PDF, vector) into paper/figs/:
  fig_env.pdf           -- regime weights + oracle reward over the week (environment)
  fig_algo_trace.pdf    -- LI-TAR-UCB internals over the week: reward/true regime,
                           operating mode, believed regime
  fig_blackout.pdf      -- case study of the first large drift transition, plain vs
                           safeguarded variant (the blackout failure and its fix)
  fig_regret.pdf        -- cumulative regret vs Oracle, all policies, both caches

The "plain" (no-safeguards) trace is recomputed once (~75 s) and cached in
paper/figs/_plain_run.npz so figure regeneration is instant afterwards.

    python bandit_paper_figs.py
"""
import os, sys
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FS = 256
HOLD = 100
CACHE = "bandit_runs/cache_1w_r1long"
CACHE2 = "bandit_runs/cache_1w_r1freq"
OUT = os.path.join("paper", "figs")
os.makedirs(OUT, exist_ok=True)

REG_COL = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}
REG_NAME = {0: "$R_0$", 1: "$R_1$", 2: "$R_2$"}
MODE_NAME = {0: "Init", 1: "Stable", 2: "Transition", 3: "Diagnostic"}
MODE_COL = {0: "0.6", 1: "#79b4d9", 2: "#b48ecb", 3: "#c69477"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load_env(cache):
    m = np.load(os.path.join(cache, "manifest.npz"), allow_pickle=True)
    Wsec = np.asarray(m["Wsec"]); tsec = np.asarray(m["tsec"])
    N = int(m["N"]); ndec = (N - 1) // (HOLD * FS)
    dec_t = np.arange(ndec) * HOLD
    Wdec = np.vstack([np.interp(dec_t, tsec, Wsec[i]) for i in range(3)])
    return Wsec, tsec, Wdec, np.argmax(Wdec, axis=0), dec_t, ndec


def load_policy(cache, name, ndec):
    d = np.load(os.path.join(cache, "experiment", "policies", f"{name}.npz"))
    return {k: d[k][:ndec] for k in d.files if d[k].ndim >= 1 and len(d[k]) >= ndec} | \
           {k: d[k] for k in d.files if d[k].ndim == 0}


def regime_bands(ax, td, reg, alpha=0.14):
    ch = np.where(np.diff(reg) != 0)[0] + 1
    bounds = np.concatenate(([0], ch, [len(reg)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.axvspan(td[a], td[min(b, len(td) - 1)], color=REG_COL[reg[a]], alpha=alpha, lw=0)


def get_plain_run():
    """Recompute (once) the no-safeguards LI-TAR-UCB trace on CACHE."""
    fp = os.path.join(OUT, "_plain_run.npz")
    if os.path.exists(fp):
        return np.load(fp)
    import bandit_rewards
    import bandit_policies as bp
    from bandit_noise_cache import load_cache
    from bandit_calibrate import calibrate
    from bandit_long_experiment import run_policy_stream, shape_separation
    from bandit_tune_tarucb import make_accessors
    cache = load_cache(CACHE)
    calib = calibrate(hold=HOLD, reward_mode="logistic")
    s = float(calib["sigma_hat"]); lo, hi = float(calib["lo"]), float(calib["hi"])
    tab = np.asarray(calib["oracle_table"], float)
    dsh = shape_separation(tab)
    r_sh = float(np.clip(0.45 * dsh, 0.6 * s, 1.2 * s))
    feat_names, w = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    W_of, sens_of, _, _ = make_accessors(cache)
    pol = bp.LITARUCB(3, sigma=s, delta=0.1, m0=6, nu=0.4*s, h=5.0*s,
                      m_batch=1, r_stab=1.0*s, q_stab=1, d_exit=1.5*s, r_coll=r_sh,
                      m_cls=6, r_sh=r_sh, q_new=2, r_merge=0.8*r_sh,
                      level_window=30, L_probe=90, m_blk=3, q_det=2, seed=0)
    print("[figs] recomputing plain (no-safeguards) run, ~75 s ...")
    h = run_policy_stream(pol, cache, sens_of, W_of, feat_names, w, HOLD, lo, hi,
                          reward_mode="logistic", progress_every=0, name="plain")
    np.savez_compressed(fp, rewards=h["rewards"], arms=h["arms"],
                        mode=np.array(pol.mode_hist, np.int8), Z=np.array(pol.Z_hist, np.int16))
    return np.load(fp)


def fig_env():
    Wsec, tsec, Wdec, reg, dec_t, ndec = load_env(CACHE)
    orc = load_policy(CACHE, "Oracle", ndec)
    td = dec_t / 86400.0
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))
    ts = tsec / 86400.0
    for i, lab in enumerate(["$W_0$ (quiet, $R_0$)", "$W_1$ (burst, $R_1$)",
                              "$W_2$ (elevated, $R_2$)"]):
        ax[0].plot(ts, Wsec[i], color=REG_COL[i], lw=0.9, label=lab)
    ax[0].set_ylabel("regime weight")
    ax[0].set_ylim(-0.04, 1.04)
    ax[0].legend(loc="center left", ncol=3, bbox_to_anchor=(0.02, 1.14), frameon=False)
    regime_bands(ax[1], td, reg)
    ax[1].plot(td, orc["rewards"], "k", lw=0.45)
    ax[1].set_ylabel("oracle reward")
    ax[1].set_xlabel("time [days]")
    ax[1].set_xlim(0, 7)
    # annotate one blackout and one burst
    bb = dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.9)
    ax[1].annotate("blackout (saturated)", xy=(0.30, 0.035), xytext=(0.52, 0.10),
                   fontsize=7.5, ha="left", bbox=bb,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[1].annotate("$R_1$ burst", xy=(5.93, 0.33), xytext=(5.05, 0.08), fontsize=7.5,
                   bbox=bb, arrowprops=dict(arrowstyle="->", lw=0.8))
    for a in ax:
        a.grid(alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(OUT, "fig_env.pdf")); fig.savefig(os.path.join(OUT, "fig_env.png"), dpi=150)
    plt.close(fig)
    print("[figs] fig_env.pdf")


def fig_algo_trace():
    _, _, Wdec, reg, dec_t, ndec = load_env(CACHE)
    li = load_policy(CACHE, "LI-TAR-UCB", ndec)
    mode = li["tar_mode"]; Z = li["tar_Z"]
    td = dec_t / 86400.0
    fig, ax = plt.subplots(3, 1, figsize=(6.3, 4.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.5, 0.55, 0.9], hspace=0.14))
    regime_bands(ax[0], td, reg)
    ax[0].plot(td, li["rewards"], "k", lw=0.4)
    ax[0].set_ylabel("reward")
    ch = np.where(np.diff(mode) != 0)[0] + 1
    bounds = np.concatenate(([0], ch, [len(mode)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax[1].axvspan(td[a], td[min(b, len(td) - 1)], color=MODE_COL[mode[a]], lw=0)
    ax[1].set_yticks([]); ax[1].set_ylabel("mode", rotation=0, ha="right", va="center")
    cmap = plt.get_cmap("tab10")
    for j in sorted(set(Z.tolist())):
        m = Z == j
        ax[2].scatter(td[m], Z[m], s=1.2, color=cmap(j % 10))
    ax[2].set_ylabel("believed\nregime $\\widehat Z$")
    ax[2].set_xlabel("time [days]")
    ax[2].set_yticks(sorted(set(Z.tolist())))
    ax[2].set_xlim(0, 7)
    for a in (ax[0], ax[2]):
        a.grid(alpha=0.25, lw=0.4)
    handles = [Patch(color=REG_COL[r], alpha=0.35, label="true " + REG_NAME[r]) for r in range(3)] \
        + [Patch(color=MODE_COL[m], label=MODE_NAME[m]) for m in (1, 2, 3)]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=1.1)
    fig.savefig(os.path.join(OUT, "fig_algo_trace.pdf")); fig.savefig(os.path.join(OUT, "fig_algo_trace.png"), dpi=150)
    plt.close(fig)
    print("[figs] fig_algo_trace.pdf")


def fig_blackout():
    _, _, Wdec, reg, dec_t, ndec = load_env(CACHE)
    li = load_policy(CACHE, "LI-TAR-UCB", ndec)          # safeguarded (production)
    plain = get_plain_run()
    th = dec_t / 3600.0
    i0, i1 = int(5.4 * 36), int(11.2 * 36)               # 36 windows/hour
    eps_floor = 0.163
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 3.9), sharex=True,
                            gridspec_kw=dict(hspace=0.14))
    for a, (rw, md, lab) in zip(ax, [
            (plain["rewards"][:ndec], plain["mode"][:ndec], "plain (no safeguards)"),
            (li["rewards"], li["tar_mode"], "with blackout guard and rise detector")]):
        seg = md[i0:i1]
        ch = np.where(np.diff(seg) != 0)[0] + 1
        bounds = np.concatenate(([0], ch, [len(seg)]))
        for x0, x1 in zip(bounds[:-1], bounds[1:]):
            a.axvspan(th[i0 + x0], th[i0 + min(x1, len(seg) - 1)],
                      color=MODE_COL[seg[x0]], alpha=0.85, lw=0)
        a.plot(th[i0:i1], rw[i0:i1], "k", lw=0.8)
        a.axhline(eps_floor, color="red", ls=":", lw=0.9)
        a.set_ylabel("reward")
        a.set_title(lab, fontsize=8.5, loc="left", pad=2)
        a.grid(alpha=0.25, lw=0.4)
    ax[0].text(5.45, eps_floor + 0.025, "$\\epsilon_{\\mathrm{floor}}$", color="red", fontsize=8)
    ax[0].annotate("settles mid-blackout,\nspurious flat regime", xy=(7.45, 0.05),
                   xytext=(7.9, 0.30), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[0].annotate("$R_2$ emerges: rise\ninvisible to drop CUSUM", xy=(8.75, 0.24),
                   xytext=(9.5, 0.42), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[1].annotate("guard holds Transition\nthrough the blackout", xy=(7.6, 0.05),
                   xytext=(6.3, 0.33), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[1].annotate("diagnosis on identifiable\npost-blackout data", xy=(9.1, 0.24),
                   xytext=(9.7, 0.45), fontsize=7.5,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    ax[1].set_xlabel("time [hours]")
    ax[1].set_xlim(th[i0], th[i1])
    fig.legend(handles=[Patch(color=MODE_COL[m], label=MODE_NAME[m]) for m in (1, 2, 3)],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.savefig(os.path.join(OUT, "fig_blackout.pdf")); fig.savefig(os.path.join(OUT, "fig_blackout.png"), dpi=150)
    plt.close(fig)
    print("[figs] fig_blackout.pdf")


def fig_regret():
    order = ["LI-TAR-UCB", "Rule-based", "TAR-UCB", "Fixed-C0", "D-UCB", "Thompson"]
    style = {"LI-TAR-UCB": dict(color="tab:blue", lw=1.4),
             "Rule-based": dict(color="tab:green", lw=1.0),
             "TAR-UCB": dict(color="tab:cyan", lw=1.0),
             "Fixed-C0": dict(color="0.45", lw=1.0, ls="--"),
             "D-UCB": dict(color="tab:purple", lw=1.0),
             "Thompson": dict(color="tab:brown", lw=1.0)}
    label = {"Fixed-C0": "Fixed-$C_0$"}
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True,
                            gridspec_kw=dict(wspace=0.08))
    for a, cache, ttl in [(ax[0], CACHE, "1 burst/day"), (ax[1], CACHE2, "2 bursts/day")]:
        _, _, _, _, dec_t, ndec = load_env(cache)
        orc = load_policy(cache, "Oracle", ndec)["rewards"]
        td = dec_t / 86400.0
        for name in order:
            p = load_policy(cache, name, ndec)
            a.plot(td, np.cumsum(orc - p["rewards"]), label=label.get(name, name),
                   **style[name])
        a.set_title(ttl, fontsize=9)
        a.set_xlabel("time [days]")
        a.set_xlim(0, 7); a.grid(alpha=0.25, lw=0.4)
    ax[0].set_ylabel("cumulative regret")
    ax[0].legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.0)
    fig.savefig(os.path.join(OUT, "fig_regret.pdf")); fig.savefig(os.path.join(OUT, "fig_regret.png"), dpi=150)
    plt.close(fig)
    print("[figs] fig_regret.pdf")


if __name__ == "__main__":
    fig_env()
    fig_algo_trace()
    fig_blackout()
    fig_regret()
    print("[figs] done ->", OUT)
