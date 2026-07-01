"""
Spec-compliance test for TAR-UCB (paper/tar_ucb_draft.tex) on a SYNTHETIC environment
that matches the draft's model exactly:

  * S recurrent stable regimes, each a mean vector mu^(z) in R^K, sub-Gaussian noise sigma.
  * Episodes alternate between regimes; between two stable episodes a TRANSITION period of
    length L_tr inserts a global reward drop of d_min on every arm (the detectability
    condition, eq. (transition_drop_condition)).

We verify the four behaviours the analysis promises:
  (1) Transition detection with short delay (Lemma detection_delay).
  (2) Correct regime classification / library size -> S (Lemma diagnostic_classification).
  (3) Recurrence reuse: regret in later visits to a regime << first visit.
  (4) Stable-regime regret far below restart baselines (SW-UCB / D-UCB), near the oracle.

Run:  python test_tarucb.py
Saves: bandit_runs/tarucb_test/{fig_synthetic.png, report.txt}
"""
import os, sys
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import numpy as np
import matplotlib.pyplot as plt
import bandit_policies as bp

OUT = os.path.join("bandit_runs", "tarucb_test")


class RecurrentRegimeEnv:
    """Alternating recurrent regimes with global-drop transitions between episodes."""
    def __init__(self, mus, order, L_ep, L_tr, sigma=0.05, d_min=0.4, seed=0):
        self.mus = [np.asarray(m, float) for m in mus]   # list of mean vectors (K,)
        self.K = len(self.mus[0])
        self.order = order                                # sequence of regime indices per episode
        self.L_ep, self.L_tr = L_ep, L_tr
        self.sigma, self.d_min = sigma, d_min
        self.rng = np.random.RandomState(seed)
        self._build()

    def _build(self):
        self.z_true = []          # true stable regime per round (-1 during transition)
        self.mean_t = []          # per-round mean vector (K,)
        for ei, z in enumerate(self.order):
            if ei > 0:            # transition before every episode after the first
                base = np.minimum(self.mus[self.order[ei-1]], self.mus[z])
                for _ in range(self.L_tr):
                    self.z_true.append(-1)
                    self.mean_t.append(base - self.d_min)   # global drop on all arms
            for _ in range(self.L_ep):
                self.z_true.append(z)
                self.mean_t.append(self.mus[z])
        self.z_true = np.array(self.z_true)
        self.mean_t = np.array(self.mean_t)               # (T,K)
        self.T = len(self.z_true)
        # per-round optimal stable arm + its mean (for regret; undefined in transition)
        self.best_arm = np.array([int(np.argmax(self.mus[z])) if z >= 0 else -1 for z in self.z_true])

    def reward(self, t, arm):
        return self.mean_t[t, arm] + self.rng.normal(0, self.sigma)

    def stable_regret(self, t, arm):
        z = self.z_true[t]
        if z < 0:
            return 0.0                                     # transition rounds excluded
        return float(self.mus[z].max() - self.mus[z][arm])


def run(policy, env, seed=0):
    env.rng = np.random.RandomState(seed)
    arms = np.empty(env.T, int); reg = np.zeros(env.T)
    for t in range(env.T):
        a = policy.choose(t, None)
        y = env.reward(t, a)
        policy.update(a, y)
        arms[t] = a; reg[t] = env.stable_regret(t, a)
    return arms, reg


def main():
    os.makedirs(OUT, exist_ok=True)
    K = 3
    muA = [0.70, 0.50, 0.40]      # regime A: optimal arm 0
    muB = [0.40, 0.55, 0.70]      # regime B: optimal arm 2
    mus = [muA, muB]
    Dreg = float(np.max(np.abs(np.array(muA) - np.array(muB))))  # sup-norm separation
    order = [0, 1, 0, 1, 0, 1, 0, 1]   # 8 episodes, A and B each recur 4x
    L_ep, L_tr, sigma, d_min = 250, 12, 0.05, 0.45
    env = RecurrentRegimeEnv(mus, order, L_ep, L_tr, sigma=sigma, d_min=d_min, seed=1)
    print(f"Synthetic env: S=2 regimes, K={K}, T={env.T}, Delta_reg={Dreg:.2f}, "
          f"sigma={sigma}, d_min={d_min}, {len(order)} episodes ({L_ep} each), L_tr={L_tr}")

    # parameters calibrated to this scale (sigma=0.05, d_min~0.45, Dreg~0.3)
    def fresh_tar():
        return bp.TARUCB(K, sigma=sigma, delta=0.1, m0=2, nu=0.05, h=0.35,
                         m_batch=1, r_stab=0.06, q_stab=2, m_cls=3, r_cls=0.15, add_diag=True, seed=0)

    policies = {
        "TAR-UCB":  fresh_tar(),
        "SW-UCB":   bp.SlidingWindowUCB(K, window=40, xi=0.4),
        "D-UCB":    bp.DiscountedUCB(K, gamma=0.97, xi=0.4),
        "D-TS":     bp.DiscountedTS(K, gamma=0.95, sigma=0.06),
        "UCB-stat": bp.GaussianTS(K, sigma=0.06),  # stationary TS as a no-forgetting baseline
    }
    SEEDS = list(range(20))
    results = {}
    for name, _ in policies.items():
        cregs = []
        last = None
        for s in SEEDS:
            pol = fresh_tar() if name == "TAR-UCB" else policies[name].__class__(
                K, **_ctor_kwargs(name))
            arms, reg = run(pol, env, seed=100 + s)
            cregs.append(np.cumsum(reg)); last = pol
        results[name] = (np.array(cregs), last)

    # ---- report ----
    lines = []
    lines.append(f"Synthetic recurrent-regime test (S=2, K={K}, T={env.T}, {len(SEEDS)} seeds)")
    lines.append(f"  Delta_reg={Dreg:.3f}  sigma={sigma}  d_min={d_min}  L_ep={L_ep}  L_tr={L_tr}\n")
    lines.append(f"{'policy':10s}  {'final stable regret (mean+/-sd)':32s}")
    for name, (cregs, _) in results.items():
        fr = cregs[:, -1]
        lines.append(f"  {name:10s}  {fr.mean():8.2f} +/- {fr.std():5.2f}")

    # TAR-UCB structural diagnostics (single representative run, seed 100)
    tar = bp.TARUCB(K, sigma=sigma, delta=0.1, m0=2, nu=0.05, h=0.35,
                    m_batch=1, r_stab=0.06, q_stab=2, m_cls=3, r_cls=0.15, add_diag=True, seed=0)
    arms, reg = run(tar, env, seed=100)
    Zh = np.array(tar.Z_hist); Jh = np.array(tar.J_hist); Mh = np.array(tar.mode_hist)
    lines.append("\nStructural diagnostics (representative run):")
    lines.append(f"  final library size J = {Jh[-1]}   (true S = 2)")
    lines.append(f"  learned regime mean vectors:")
    for j in range(len(tar.N)):
        lines.append(f"    regime {j}: mu_hat = [{', '.join(f'{x:.3f}' for x in tar._mu(j))}]  N={tar.N[j].astype(int)}")
    # detection delay per transition + classification correctness
    tr_starts = np.where(np.diff((env.z_true < 0).astype(int)) == 1)[0] + 1
    delays, correct = [], []
    for ts in tr_starts:
        fired = np.where((Mh[ts:] == bp.TARUCB.TRANS))[0]
        delays.append(int(fired[0]) if len(fired) else -1)
    # which library regime is active in the back-half of each stable episode, vs truth
    ep_bounds = []
    z = env.z_true
    i = 0
    while i < env.T:
        if z[i] >= 0:
            j = i
            while j < env.T and z[j] == z[i]:
                j += 1
            ep_bounds.append((i, j, z[i])); i = j
        else:
            i += 1
    # map each true regime to the library label TAR-UCB uses in that episode's stable tail
    map_true2lib = {}
    consistent = True
    for (a, b, ztrue) in ep_bounds:
        lib = np.bincount(Zh[a + (b - a)//2:b]).argmax()
        if ztrue in map_true2lib and map_true2lib[ztrue] != lib:
            consistent = False
        map_true2lib.setdefault(ztrue, lib)
    lines.append(f"  median transition-detection delay = {int(np.median(delays))} rounds "
                 f"(theory O(sigma^2/d_min^2 log) ~ {sigma**2/d_min**2*np.log(env.T/0.1):.1f})")
    lines.append(f"  regime labels consistent across recurrences: {consistent}  "
                 f"(true->lib map: {map_true2lib})")
    # recurrence reuse: (1) suboptimal-arm samples stay ~ m_cls*visits (pooled, not re-explored),
    # (2) per-episode overhead is constant and well below a restart baseline's per-episode cost.
    cum = np.cumsum(reg)
    perA = [cum[b - 1] - cum[a] for (a, b, zt) in ep_bounds if zt == 0]
    n_ep = len(ep_bounds)
    tar_per_ep = cum[-1]/n_ep
    restart_per_ep = results["D-UCB"][0][:, -1].mean()/n_ep
    sub_counts = {j: tar.N[j].astype(int) for j in range(len(tar.N))}
    lines.append(f"  pooled per-regime pull counts (opt arm dominates; suboptimals ~ m_cls*visits): {sub_counts}")
    lines.append(f"  per-visit stable regret for regime A (visits 1..{len(perA)}): "
                 f"[{', '.join(f'{x:.2f}' for x in perA)}]  (visit 1 has no preceding transition)")
    lines.append(f"  TAR-UCB regret/episode = {tar_per_ep:.2f}  vs  D-UCB(restart) regret/episode "
                 f"= {restart_per_ep:.2f}  -> reuse saves {restart_per_ep/max(tar_per_ep,1e-9):.1f}x")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(report + "\n")

    # ---- figure ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    cols = {"TAR-UCB": 'tab:blue', "SW-UCB": 'tab:orange', "D-UCB": 'tab:green',
            "D-TS": 'tab:red', "UCB-stat": 'tab:purple'}
    for name, (cregs, _) in results.items():
        m = cregs.mean(0); e = cregs.std(0)/np.sqrt(len(SEEDS))
        x = np.arange(env.T)
        axes[0].plot(x, m, color=cols[name], lw=1.8, label=f"{name} ({m[-1]:.1f})")
        axes[0].fill_between(x, m - e, m + e, color=cols[name], alpha=0.15)
    # shade transitions
    intr = env.z_true < 0
    axes[0].fill_between(np.arange(env.T), 0, axes[0].get_ylim()[1], where=intr,
                         color='k', alpha=0.06, step='mid')
    axes[0].set_ylabel("cumulative stable regret"); axes[0].legend(fontsize=9)
    axes[0].set_title(f"TAR-UCB on synthetic recurrent regimes (S=2, {len(order)} episodes, "
                      f"{len(SEEDS)} seeds): regret vs restart baselines")
    axes[0].grid(alpha=0.3)
    # truth + inferred regime
    axes[1].plot(np.arange(env.T), np.where(env.z_true < 0, np.nan, env.z_true),
                 'k', lw=2, label='true regime')
    axes[1].plot(np.arange(env.T), Zh, color='tab:blue', lw=1, alpha=0.8, label='TAR-UCB $\\hat Z$')
    axes[1].fill_between(np.arange(env.T), -0.5, 2.5, where=intr, color='k', alpha=0.06, step='mid')
    axes[1].set_ylabel("regime"); axes[1].set_yticks([0, 1, 2]); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    # mode + library growth
    axes[2].plot(np.arange(env.T), Mh, color='tab:blue', lw=1, label='mode (0=I,1=S,2=T,3=D)')
    axes[2].plot(np.arange(env.T), Jh, color='tab:red', lw=1.5, label='library size $J$')
    axes[2].axhline(2, color='tab:red', ls=':', alpha=0.6)
    axes[2].fill_between(np.arange(env.T), 0, 3.2, where=intr, color='k', alpha=0.06, step='mid')
    axes[2].set_ylabel("mode / $J$"); axes[2].set_xlabel("round"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_synthetic.png"), dpi=130); plt.close()
    print(f"\nSaved {OUT}/fig_synthetic.png + report.txt")


def _ctor_kwargs(name):
    return {
        "SW-UCB":   dict(window=40, xi=0.4),
        "D-UCB":    dict(gamma=0.97, xi=0.4),
        "D-TS":     dict(gamma=0.95, sigma=0.06),
        "UCB-stat": dict(sigma=0.06),
    }[name]


if __name__ == "__main__":
    main()
