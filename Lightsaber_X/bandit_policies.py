"""
Policy library for the online controller-selection experiment.

Uniform interface:
    choose(step, ctx) -> arm index            (ctx = regime weights W, a length-3 vector)
    update(arm, reward)                        (learning policies use it; others ignore)

Learning bandits (use only the reward; ctx ignored):
    AdSwitchPolicy        -- change-detection, distribution-free (wraps bandit/adswitch.py)
    SlidingWindowUCB      -- passive forgetting (Garivier & Moulines)
    DiscountedUCB         -- passive forgetting (Garivier & Moulines)
    DiscountedTS          -- discounted Gaussian Thompson sampling (forgetting)
    GaussianTS            -- stationary Thompson sampling (non-stationarity baseline)

Reference policies:
    FixedArm(a)           -- always play arm a
    Oracle(table)         -- argmax_c E[r_c | W] from the characterised reward table
    RuleBased(...)        -- operator-like switch on a noisy regime proxy with hysteresis
"""
import numpy as np
from bandit.adswitch import AdSwitch


class AdSwitchPolicy:
    def __init__(self, num_arms, horizon, C1=4.0):
        self.bandit = AdSwitch(num_arms=num_arms, horizon=horizon, C1=C1)
    def choose(self, step, ctx=None):
        return self.bandit.select_arm()
    def update(self, arm, reward):
        self.bandit.update(arm, reward)


class SlidingWindowUCB:
    def __init__(self, num_arms, window=30, xi=0.5):
        self.K, self.w, self.xi = num_arms, window, xi
        self.hist = []
    def choose(self, step, ctx=None):
        recent = self.hist[-self.w:]
        counts = np.zeros(self.K); sums = np.zeros(self.K)
        for a, r in recent:
            counts[a] += 1; sums[a] += r
        for a in range(self.K):
            if counts[a] == 0:
                return a
        t = len(recent)
        ucb = sums/counts + np.sqrt(self.xi*np.log(max(t, 2))/counts)
        return int(np.argmax(ucb))
    def update(self, arm, reward):
        self.hist.append((arm, reward))


class DiscountedUCB:
    def __init__(self, num_arms, gamma=0.95, xi=0.5):
        self.K, self.g, self.xi = num_arms, gamma, xi
        self.N = np.zeros(self.K); self.X = np.zeros(self.K)
    def choose(self, step, ctx=None):
        for a in range(self.K):
            if self.N[a] < 1e-9:
                return a
        nt = self.N.sum()
        ucb = self.X/self.N + np.sqrt(self.xi*np.log(nt)/self.N)
        return int(np.argmax(ucb))
    def update(self, arm, reward):
        self.N *= self.g; self.X *= self.g
        self.N[arm] += 1.0; self.X[arm] += reward


class DiscountedTS:
    # mu0 optimistic so each arm is explored before exploitation (rewards ~0.4-0.6, not 0).
    def __init__(self, num_arms, gamma=0.9, sigma=0.1, mu0=1.0):
        self.K, self.g, self.sigma = num_arms, gamma, sigma
        self.n = np.zeros(self.K); self.mu = np.full(self.K, mu0)
    def choose(self, step, ctx=None):
        samp = [np.random.normal(self.mu[a], self.sigma/np.sqrt(self.n[a]+1.0)) for a in range(self.K)]
        return int(np.argmax(samp))
    def update(self, arm, reward):
        self.n *= self.g
        self.n[arm] += 1.0
        self.mu[arm] += (reward - self.mu[arm])/self.n[arm]


class GaussianTS:
    def __init__(self, num_arms, sigma=0.1, mu0=1.0):
        self.K, self.sigma = num_arms, sigma
        self.n = np.zeros(self.K); self.mu = np.full(self.K, mu0)
    def choose(self, step, ctx=None):
        samp = [np.random.normal(self.mu[a], self.sigma/np.sqrt(self.n[a]+1.0)) for a in range(self.K)]
        return int(np.argmax(samp))
    def update(self, arm, reward):
        self.n[arm] += 1.0; self.mu[arm] += (reward - self.mu[arm])/self.n[arm]


class ContextualTS:
    """Thompson sampling with a (noisy) regime context. The context -- a coarse, imperfect
    proxy for the operating regime (e.g. a ground-motion / earthquake-forecast indicator) --
    is binned, and a separate Gaussian posterior over arm rewards is learned per bin. Unlike
    the rule-based policy it does not assume the regime->controller map; it learns it online."""
    def __init__(self, num_arms, n_bins=3, sigma=0.05, ctx_noise=0.2, lo=0.66, hi=1.34, mu0=1.0, seed=0):
        self.K, self.nb, self.sigma, self.ctx_noise = num_arms, n_bins, sigma, ctx_noise
        self.lo, self.hi = lo, hi
        self.rng = np.random.RandomState(seed)
        self.n = np.zeros((n_bins, num_arms)); self.mu = np.full((n_bins, num_arms), mu0)
        self._b = 0
    def _bin(self, ctx):
        s = float(np.asarray(ctx) @ np.array([0.0, 1.0, 2.0])) + self.rng.normal(0, self.ctx_noise)
        return 0 if s < self.lo else (1 if s < self.hi else 2)
    def choose(self, step, ctx=None):
        b = self._bin(ctx); self._b = b
        samp = [self.rng.normal(self.mu[b, a], self.sigma/np.sqrt(self.n[b, a]+1.0)) for a in range(self.K)]
        return int(np.argmax(samp))
    def update(self, arm, reward):
        b = self._b
        self.n[b, arm] += 1.0
        self.mu[b, arm] += (reward - self.mu[b, arm])/self.n[b, arm]


class TARUCB:
    """Transition-Aware Recurrent UCB (paper/tar_ucb_draft.tex).

    Non-contextual: the latent operating regime is inferred from the reward structure
    alone (ctx is ignored).  Three modes:
      STABLE     -- regime-conditioned UCB selection + a one-sided CUSUM drop detector
                    on the lower-confidence-bound residual R_t=[L_{j,A_t}-Y_t]_+.
      TRANSITION -- freeze stable-regime stats; round-robin batches until the reward
                    vector stabilises (||v^(b)-v^(b-1)||_inf <= r_stab for q_stab batches).
      DIAGNOSTIC -- round-robin m_cls/arm; match the mean vector to the regime library
                    (sup-norm <= r_cls) or create a new regime.
    A recurrent-regime library pools statistics across repeated visits, so a returning
    regime is recognised (Diagnostic) rather than relearned.  Faithful to Alg. 1 of the
    draft; defaults are calibrated to a reward scale ~[0,1] with within-regime sigma~0.05.
    """
    INIT, STABLE, TRANS, DIAG = 0, 1, 2, 3

    def __init__(self, num_arms, sigma=0.05, delta=0.1, m0=2,
                 nu=0.05, h=0.40, m_batch=1, r_stab=0.06, q_stab=2,
                 m_cls=3, r_cls=0.06, add_diag=True, seed=0):
        self.K = num_arms; self.sigma = sigma; self.delta = delta
        self.m0 = m0; self.nu = nu; self.h = h
        self.m_batch = m_batch; self.r_stab = r_stab; self.q_stab = q_stab
        self.m_cls = m_cls; self.r_cls = r_cls; self.add_diag = add_diag
        self.rng = np.random.RandomState(seed)
        self.N, self.S = [], []           # regime library: per-regime count / reward-sum vectors
        self.Z = 0                        # current inferred regime index
        self.G = 0.0; self.t = 0          # CUSUM statistic; round counter
        self.mode = self.INIT
        self._iseq = self._rr(m0); self._iptr = 0
        self._isum = np.zeros(self.K); self._icnt = np.zeros(self.K)
        self._reset_trans(); self._reset_diag()
        # diagnostics (per round)
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []

    def _rr(self, m):
        """round-robin schedule: m passes over all arms."""
        return [k for _ in range(int(m)) for k in range(self.K)]

    def _B(self, Njk):
        if Njk <= 0:
            return np.inf
        J = max(len(self.N), 1)
        return self.sigma*np.sqrt(2.0*np.log(4.0*self.K*J*max(self.t, 1)**2/self.delta)/Njk)

    def _mu(self, j):
        return self.S[j]/np.maximum(self.N[j], 1)

    def _reset_trans(self):
        self._tseq = self._rr(self.m_batch); self._tptr = 0
        self._tsum = np.zeros(self.K); self._tcnt = np.zeros(self.K)
        self._tprev = None; self._tstable = 0

    def _reset_diag(self):
        self._dseq = self._rr(self.m_cls); self._dptr = 0
        self._dsum = np.zeros(self.K); self._dcnt = np.zeros(self.K)

    def choose(self, step, ctx=None):
        if self.mode == self.INIT:
            return self._iseq[self._iptr]
        if self.mode == self.STABLE:
            j = self.Z; N = self.N[j]
            un = np.where(N == 0)[0]
            if len(un):
                return int(un[0])                       # initialise zero-sample arms first
            ucb = self._mu(j) + np.array([self._B(N[k]) for k in range(self.K)])
            return int(np.argmax(ucb))
        if self.mode == self.TRANS:
            return self._tseq[self._tptr]
        return self._dseq[self._dptr]                   # DIAG

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.INIT:
            self._isum[arm] += reward; self._icnt[arm] += 1; self._iptr += 1
            if self._iptr >= len(self._iseq):           # first regime from the init vector
                self.N.append(self._icnt.copy()); self.S.append(self._isum.copy())
                self.Z = 0; self.G = 0.0; self.mode = self.STABLE

        elif self.mode == self.STABLE:
            j = self.Z
            if self.N[j][arm] > 0:                       # one-sided drop residual vs LCB
                L = self.S[j][arm]/self.N[j][arm] - self._B(self.N[j][arm])
                R = max(L - reward, 0.0)
            else:
                R = 0.0
            self.G = max(0.0, self.G + R - self.nu)
            if self.G >= self.h:                         # transition alarm
                self.mode = self.TRANS; self.G = 0.0; self._reset_trans()
            else:                                        # assign to current regime
                self.N[j][arm] += 1.0; self.S[j][arm] += reward

        elif self.mode == self.TRANS:
            self._tsum[arm] += reward; self._tcnt[arm] += 1.0; self._tptr += 1
            if self._tptr >= len(self._tseq):            # batch complete
                v = self._tsum/np.maximum(self._tcnt, 1)
                if self._tprev is not None:
                    H = float(np.max(np.abs(v - self._tprev)))
                    self._tstable = self._tstable + 1 if H <= self.r_stab else 0
                self._tprev = v.copy()
                if self._tstable >= self.q_stab:
                    self.mode = self.DIAG; self._reset_diag()
                else:                                    # new batch
                    self._tseq = self._rr(self.m_batch); self._tptr = 0
                    self._tsum[:] = 0.0; self._tcnt[:] = 0.0

        elif self.mode == self.DIAG:
            self._dsum[arm] += reward; self._dcnt[arm] += 1.0; self._dptr += 1
            if self._dptr >= len(self._dseq):            # classify vs library
                v = self._dsum/np.maximum(self._dcnt, 1)
                D = [float(np.max(np.abs(v - self._mu(j)))) for j in range(len(self.N))]
                jstar = int(np.argmin(D))
                if D[jstar] <= self.r_cls:               # recognised regime -> reuse
                    self.Z = jstar
                    if self.add_diag:
                        self.N[jstar] += self._dcnt; self.S[jstar] += self._dsum
                else:                                    # genuinely new regime
                    self.N.append(self._dcnt.copy()); self.S.append(self._dsum.copy())
                    self.Z = len(self.N) - 1
                self.G = 0.0; self.mode = self.STABLE

        self.Z_hist.append(self.Z); self.J_hist.append(len(self.N)); self.mode_hist.append(self.mode)


class FixedArm:
    def __init__(self, arm):
        self.arm = arm
    def choose(self, step, ctx=None):
        return self.arm
    def update(self, arm, reward):
        pass


class Oracle:
    """Plays argmax_c E[r_c | W], with E[r_c|W] = sum_i W_i r_c(R_i)."""
    def __init__(self, reward_table):
        self.R = np.asarray(reward_table, float)  # shape (num_arms, 3)
    def choose(self, step, ctx=None):
        Er = self.R @ np.asarray(ctx, float)
        return int(np.argmax(Er))
    def update(self, arm, reward):
        pass


class RuleBased:
    """Operator-like: observe a noisy severity proxy s_hat in [0,2] and switch
    C0/C1/C2 on thresholds with hysteresis (mimics forecast-driven switching)."""
    def __init__(self, lo=0.66, hi=1.34, hyst=0.15, noise=0.15, seed=0):
        self.lo, self.hi, self.hyst, self.noise = lo, hi, hyst, noise
        self.rng = np.random.RandomState(seed)
        self.arm = 0
    def choose(self, step, ctx=None):
        # severity proxy from weights: s = 0*w0 + 1*w1 + 2*w2 (+ observation noise)
        s = float(np.asarray(ctx) @ np.array([0.0, 1.0, 2.0])) + self.rng.normal(0, self.noise)
        lo = self.lo + (self.hyst if self.arm >= 1 else 0.0)
        hi = self.hi + (self.hyst if self.arm >= 2 else 0.0)
        if s < lo:
            self.arm = 0
        elif s < hi:
            self.arm = 1
        else:
            self.arm = 2
        return self.arm
    def update(self, arm, reward):
        pass
