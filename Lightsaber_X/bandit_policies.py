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
    """Discounted Thompson sampling (DS-TS, Qi/Guo/Zhu, arXiv:2305.10718).

    Every round ALL arms' statistics are discounted by gamma and the played arm adds one
    observation, so each arm carries a discounted pull count N_a=sum_s gamma^{t-s} 1{a_s=a}
    (saturating at 1/(1-gamma)) and a discounted mean mu_a. The incremental update
    mu_a += (r-mu_a)/N_a is exactly discounted_sum/discounted_count -- recent rewards
    dominate, so the posterior tracks a non-stationary mean without any change detector or
    regime memory. Play argmax over one Gaussian draw per arm.

    Two sampling modes:
      known-variance (default, disc_var=False): draw ~ N(mu_a, sigma/sqrt(N_a+1)) with the
        calibrated within-regime noise sigma. This is the abruptly-changing DS-TS variant.
      discounted-variance (disc_var=True): estimate each arm's noise from a discounted second
        moment (var_a = E_gamma[r^2] - mu_a^2) and draw ~ N(mu_a, sqrt(var_eff/(N_a+1))),
        var_eff = max(var_a, var_floor)*var_inflate. This is the smoothly-changing variant --
        it lets the exploration width adapt to the locally-observed reward spread (relevant
        to our continuous regime ramps rather than clean discrete switches).

    mu0 optimistic so each arm is explored before exploitation (rewards ~0.4-0.6, not 0).
    """
    def __init__(self, num_arms, gamma=0.9, sigma=0.1, mu0=1.0,
                 disc_var=False, var_floor=None, var_inflate=1.0):
        self.K, self.g, self.sigma = num_arms, gamma, sigma
        self.disc_var = disc_var
        self.var_floor = (sigma**2) if var_floor is None else float(var_floor)
        self.var_inflate = float(var_inflate)
        self.n = np.zeros(self.K); self.mu = np.full(self.K, mu0)
        self.q = np.full(self.K, mu0**2)      # discounted mean of reward^2 (for disc_var)
    def choose(self, step, ctx=None):
        if self.disc_var:
            var = np.maximum(self.q - self.mu**2, self.var_floor) * self.var_inflate
            std = np.sqrt(var / (self.n + 1.0))
        else:
            std = self.sigma / np.sqrt(self.n + 1.0)
        samp = [np.random.normal(self.mu[a], std[a]) for a in range(self.K)]
        return int(np.argmax(samp))
    def update(self, arm, reward):
        self.n *= self.g
        self.n[arm] += 1.0
        self.mu[arm] += (reward - self.mu[arm])/self.n[arm]
        self.q[arm] += (reward*reward - self.q[arm])/self.n[arm]


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
        self.diag_log = []   # per DIAG event: (t, v, D_list, jstar, Dmin, created_new)

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
                created_new = D[jstar] > self.r_cls
                self.diag_log.append((self.t, v.copy(), list(D), jstar, D[jstar], created_new))
                if not created_new:                      # recognised regime -> reuse
                    self.Z = jstar
                    if self.add_diag:
                        self.N[jstar] += self._dcnt; self.S[jstar] += self._dsum
                else:                                    # genuinely new regime
                    self.N.append(self._dcnt.copy()); self.S.append(self._dsum.copy())
                    self.Z = len(self.N) - 1
                self.G = 0.0; self.mode = self.STABLE

        self.Z_hist.append(self.Z); self.J_hist.append(len(self.N)); self.mode_hist.append(self.mode)


class LITARUCB:
    """Level-Invariant Transition-Aware Recurrent UCB (paper/li_tar_ucb_draft.tex).

    Upgrade of TARUCB for regimes whose ABSOLUTE reward level drifts across visits while
    the action-relevant SHAPE (which arm is best) recurs. Key changes vs TARUCB:
      * Regimes are stored/matched by the CENTERED shape phi(v)=v-mean(v), not the absolute
        mean vector, so a returning regime is recognised regardless of level drift.
      * A nuisance-level tracker (mean of Y - theta_hat over a recent window) feeds a
        level-adjusted one-sided CUSUM drop detector.
      * TRANSITION has a collapse guard: it exits to DIAGNOSTIC only when the reward vector
        is stable AND no longer a collapsed version of the pre-transition regime, so a
        depressed-but-flat transient is not mistaken for a settled regime.
      * DIAGNOSTIC matches in shape space; a NEW regime is created only after q_new
        consecutive non-matching blocks (persistence), and near-duplicate library regimes
        are merged (r_merge). Faithful to Alg. 1 of the draft.
      * STABLE interleaves SCHEDULED BALANCED PROBING blocks (every L_probe rounds, m_blk
        samples per arm): each block's centered shape is compared to the current library
        entry. Consistent blocks (sup-dist <= r_det) feed the library; q_det consecutive
        deviating blocks raise a level-invariant SHAPE-CHANGE alarm. This removes the
        drop-detector blind spot for direct switches that keep the played arm's absolute
        reward constant while rearranging the arm ordering (draft Sec. 5.3 / Lemma 8).
    Shape statistics are accumulated from balanced blocks (m0 init, probing, m_cls diagnostic).

    Optional guarantee-preserving extensions (both OFF by default -> behaviour identical
    to the plain draft algorithm):
      * h_up (two-sided level CUSUM): a second one-sided CUSUM on UPWARD residuals,
        symmetric to the drop detector. Catches the end of a deep transition collapse
        that the algorithm mistakenly settled in (a reward RISE), which the one-sided
        drop detector is blind to. Same Page/Lorden delay and false-alarm bounds (it is
        the drop test applied to the negated residual); the union over the two sides
        only doubles the false-alarm probability.
      * eps_floor (blackout guard): a TRANSITION batch or DIAGNOSTIC block whose BEST
        arm-mean is below eps_floor is treated as unidentifiable ("blackout": e.g. a
        saturating reward map pins every arm near 0 mid-collapse, flattening the shape
        and voiding the additive-collapse model). Such batches never count toward the
        stability exit and never reach the library. Requires only the margin assumption
        that every genuine regime's best-arm mean exceeds eps_floor; blackout time is
        then a subset of the transition time already charged in the regret bound.
    """
    INIT, STABLE, TRANS, DIAG = 0, 1, 2, 3

    def __init__(self, num_arms, sigma=0.05, delta=0.1, m0=6, C_phi=1.0,
                 nu=0.03, h=0.35, m_batch=1, r_stab=0.05, q_stab=2,
                 d_exit=0.08, r_coll=0.06,
                 m_cls=8, r_sh=0.05, q_new=2, r_merge=0.045,
                 level_window=30, L_probe=60, m_blk=6, r_det=None, q_det=2, seed=0,
                 h_up=None, eps_floor=None):
        self.K = num_arms; self.sigma = sigma; self.delta = delta
        self.m0 = m0; self.C_phi = C_phi; self.nu = nu; self.h = h
        self.m_batch = m_batch; self.r_stab = r_stab; self.q_stab = q_stab
        self.d_exit = d_exit; self.r_coll = r_coll
        self.m_cls = m_cls; self.r_sh = r_sh; self.q_new = q_new; self.r_merge = r_merge
        self.level_window = level_window
        self.L_probe = int(L_probe); self.m_blk = int(m_blk)
        self.r_det = r_sh if r_det is None else float(r_det)
        self.q_det = int(q_det)
        self.h_up = None if h_up is None else float(h_up)
        self.eps_floor = None if eps_floor is None else float(eps_floor)
        self.rng = np.random.RandomState(seed)
        # library: per regime, per arm -> sum of block SHAPES, #blocks, #samples
        self.th_sum, self.nblk, self.nsamp = [], [], []
        self.Z = 0; self.G = 0.0; self.G_up = 0.0; self.t = 0
        self.mode = self.INIT
        self.ell = 0.0                       # nuisance-level estimate
        self._lvl = []                       # recent level residuals (window)
        self._iseq = self._rr(m0); self._iptr = 0
        self._isum = np.zeros(self.K); self._icnt = np.zeros(self.K)
        self._reset_trans(); self._reset_diag()
        self._cand = []                      # candidate shapes for persistent new-regime
        # scheduled probing state (shape-change detector)
        self._since_probe = 0; self._probing = False; self.c_det = 0
        self._pseq = self._rr(self.m_blk); self._pptr = 0
        self._psum = np.zeros(self.K); self._pcnt = np.zeros(self.K)
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []
        self.diag_log = []                   # (t, psi, D_list, jstar, Dmin, created_new)
        self.probe_log = []                  # (t, D_b, c_det, alarmed)

    # ---- helpers ----
    def _rr(self, m):
        return [k for _ in range(int(m)) for k in range(self.K)]

    @staticmethod
    def _phi(v):
        return v - v.mean()

    def _theta(self, j):
        return self.th_sum[j] / max(self.nblk[j], 1.0)

    def _B(self, j):
        """Per-arm shape confidence radius (same for all arms: blocks are balanced)."""
        n = self.nsamp[j]
        if n <= 0:
            return np.full(self.K, np.inf)
        J = max(len(self.nsamp), 1)
        r = self.C_phi * self.sigma * np.sqrt(2.0 * np.log(4.0 * self.K * J * max(self.t, 1) ** 2 / self.delta) / n)
        return np.full(self.K, r)

    def _reset_trans(self):
        self._tseq = self._rr(self.m_batch); self._tptr = 0
        self._tsum = np.zeros(self.K); self._tcnt = np.zeros(self.K)
        self._tprev = None; self._tstable = 0

    def _reset_diag(self):
        self._dseq = self._rr(self.m_cls); self._dptr = 0
        self._dsum = np.zeros(self.K); self._dcnt = np.zeros(self.K)

    def _reset_probe(self):
        self._pseq = self._rr(self.m_blk); self._pptr = 0
        self._psum = np.zeros(self.K); self._pcnt = np.zeros(self.K)

    def _enter_trans(self):
        self._jpre = self.Z; self._ellpre = self.ell
        self.G = 0.0; self.G_up = 0.0; self.c_det = 0; self._probing = False; self._since_probe = 0
        self._reset_trans(); self.mode = self.TRANS

    def _add_block(self, j, psi, msamp):
        self.th_sum[j] += psi; self.nblk[j] += 1.0; self.nsamp[j] += msamp

    def _new_regime(self, psi, msamp, nblk=1.0):
        self.th_sum.append(psi.copy()); self.nblk.append(float(nblk)); self.nsamp.append(float(msamp))
        return len(self.th_sum) - 1

    def _merge(self):
        """Pool library regimes whose shape estimates are within r_merge (sup-norm)."""
        merged = True
        while merged and len(self.th_sum) > 1:
            merged = False
            J = len(self.th_sum)
            for i in range(J):
                for j in range(i + 1, J):
                    if float(np.max(np.abs(self._theta(i) - self._theta(j)))) <= self.r_merge:
                        self.th_sum[i] += self.th_sum[j]; self.nblk[i] += self.nblk[j]; self.nsamp[i] += self.nsamp[j]
                        del self.th_sum[j]; del self.nblk[j]; del self.nsamp[j]
                        if self.Z == j: self.Z = i
                        elif self.Z > j: self.Z -= 1
                        merged = True
                        break
                if merged:
                    break

    # ---- interface ----
    def choose(self, step, ctx=None):
        if self.mode == self.INIT:
            return self._iseq[self._iptr]
        if self.mode == self.STABLE:
            if self._probing:                      # scheduled balanced probing block
                return self._pseq[self._pptr]
            j = self.Z
            if self.nsamp[j] <= 0:                 # regime not yet probed (balanced blocks)
                return 0
            ucb = self._theta(j) + self._B(j)
            return int(np.argmax(ucb))
        if self.mode == self.TRANS:
            return self._tseq[self._tptr]
        return self._dseq[self._dptr]

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.INIT:
            self._isum[arm] += reward; self._icnt[arm] += 1; self._iptr += 1
            if self._iptr >= len(self._iseq):
                v = self._isum / np.maximum(self._icnt, 1)
                self._new_regime(self._phi(v), self.m0)
                self.Z = 0; self.ell = float(v.mean()); self._lvl = [self.ell]
                self.G = 0.0; self.G_up = 0.0; self.mode = self.STABLE

        elif self.mode == self.STABLE:
            j = self.Z
            th = self._theta(j)
            self._lvl.append(reward - th[arm])                     # level residual (any arm)
            if len(self._lvl) > self.level_window:
                self._lvl.pop(0)
            self.ell = float(np.mean(self._lvl))
            if self._probing:                                      # scheduled probing block
                self._psum[arm] += reward; self._pcnt[arm] += 1.0; self._pptr += 1
                if self._pptr >= len(self._pseq):                  # block complete
                    v = self._psum / np.maximum(self._pcnt, 1)
                    psi = self._phi(v)
                    D_b = float(np.max(np.abs(psi - th)))
                    if D_b <= self.r_det:                          # consistent -> library
                        self._add_block(j, psi, self.m_blk); self.c_det = 0
                    else:                                          # deviating -> evidence
                        self.c_det += 1
                    alarmed = self.c_det >= self.q_det
                    self.probe_log.append((self.t, D_b, self.c_det, alarmed))
                    self._probing = False; self._since_probe = 0
                    if alarmed:                                    # shape-change alarm
                        self._enter_trans()
            else:
                L = self.ell + th[arm] - self._B(j)[arm]          # level-adjusted lower prediction
                R = max(L - reward, 0.0)
                self.G = max(0.0, self.G + R - self.nu)
                if self.h_up is not None:                         # symmetric rise detector
                    U = self.ell + th[arm] + self._B(j)[arm]      # level-adjusted upper prediction
                    Rup = max(reward - U, 0.0)
                    self.G_up = max(0.0, self.G_up + Rup - self.nu)
                self._since_probe += 1
                if self.G >= self.h or (self.h_up is not None and self.G_up >= self.h_up):
                    self._enter_trans()                           # drop OR rise alarm
                elif self._since_probe >= self.L_probe:           # schedule next probing block
                    self._probing = True; self._reset_probe()

        elif self.mode == self.TRANS:
            self._tsum[arm] += reward; self._tcnt[arm] += 1.0; self._tptr += 1
            if self._tptr >= len(self._tseq):
                v = self._tsum / np.maximum(self._tcnt, 1)
                psi = self._phi(v)
                # blackout guard: if even the BEST arm is below the floor, the reward map
                # is saturated -- the shape is unidentifiable, so the batch can neither
                # certify stability nor exit to DIAG. Keep batching until light returns.
                blackout = self.eps_floor is not None and float(v.max()) < self.eps_floor
                H = np.inf if self._tprev is None else float(np.max(np.abs(v - self._tprev)))
                m_pre = self._ellpre + self._theta(self._jpre)
                d_b = float(np.mean(m_pre - v))                   # estimated common drop
                collapsed = (d_b >= self.d_exit) and \
                            (float(np.max(np.abs(psi - self._theta(self._jpre)))) <= self.r_coll)
                self._tstable = self._tstable + 1 \
                    if (H <= self.r_stab and not collapsed and not blackout) else 0
                self._tprev = v.copy()
                if self._tstable >= self.q_stab:
                    self.mode = self.DIAG; self._reset_diag(); self._cand = []
                else:
                    self._tseq = self._rr(self.m_batch); self._tptr = 0
                    self._tsum[:] = 0.0; self._tcnt[:] = 0.0

        elif self.mode == self.DIAG:
            self._dsum[arm] += reward; self._dcnt[arm] += 1.0; self._dptr += 1
            if self._dptr >= len(self._dseq):
                v = self._dsum / np.maximum(self._dcnt, 1)
                psi = self._phi(v)
                if self.eps_floor is not None and float(v.max()) < self.eps_floor:
                    # blackout resumed mid-diagnosis: the block is unidentifiable --
                    # discard it and any pending candidates, go back to TRANSITION.
                    self._cand = []
                    self._reset_trans(); self.mode = self.TRANS
                    self.Z_hist.append(self.Z); self.J_hist.append(len(self.th_sum))
                    self.mode_hist.append(self.mode)
                    return
                D = [float(np.max(np.abs(psi - self._theta(j)))) for j in range(len(self.th_sum))]
                jstar = int(np.argmin(D)); Dmin = D[jstar]
                matched = Dmin <= self.r_sh
                self.diag_log.append((self.t, psi.copy(), list(D), jstar, Dmin, not matched))
                if matched:                                       # recurrence: reuse regime
                    self.Z = jstar; self._add_block(jstar, psi, self.m_cls)
                    self._cand = []
                    self._merge()
                    self.ell = float(v.mean()); self._lvl = [self.ell]
                    self.G = 0.0; self.G_up = 0.0
                    self.c_det = 0; self._probing = False; self._since_probe = 0
                    self.mode = self.STABLE
                else:
                    self._cand.append((psi.copy(), float(v.mean())))
                    if len(self._cand) >= self.q_new:             # persistent -> genuinely new
                        sum_psi = np.sum([c[0] for c in self._cand], axis=0)   # th_sum = SUM of block shapes
                        self.Z = self._new_regime(sum_psi, self.m_cls * len(self._cand), nblk=len(self._cand))
                        self.ell = float(np.mean([c[1] for c in self._cand]))
                        self._cand = []; self._merge()
                        self._lvl = [self.ell]
                        self.G = 0.0; self.G_up = 0.0
                        self.c_det = 0; self._probing = False; self._since_probe = 0
                        self.mode = self.STABLE
                    else:                                         # collect another diagnostic block
                        self._reset_diag()

        self.Z_hist.append(self.Z); self.J_hist.append(len(self.th_sum)); self.mode_hist.append(self.mode)


class CGICLB:
    """Confidence-Gated Identify-and-Commit Library Bandit (paper/cg_iclb_detailed_algorithm.tex).

    A recurrent-regime bandit with THREE modes and no UCB/Thompson mechanism in stable
    operation. Confidence intervals are used only as GATES (stable-block, regime match,
    new-regime creation, best-arm separation), never as a selection index.

      IDENTIFY  -- collect balanced blocks (each arm sampled m times), grow a stable
                   aggregate C, and match its centered shape psi(C)=P mu^C against the
                   library. Unique match -> assign C to that regime; if it already has a
                   certified arm, COMMIT immediately (library reuse), else CERTIFY. No
                   match -> treat C as a candidate new regime and CERTIFY it.
      CERTIFY   -- for an existing regime j, commit once LCB_{j,ahat} > max_{k} UCB_{j,k}+gamma;
                   for a candidate C, create a new regime only when its best arm is certified
                   AND its shape is separated from every library entry. Otherwise keep
                   collecting stable/compatible balanced blocks (a block incompatible with j
                   aborts back to IDENTIFY; incompatible with C resets the candidate).
      COMMIT    -- play the certified arm deterministically (no exploration). Watch the
                   committed arm for drift with a two-sided CUSUM on the absolute residual
                   |Y_t - (ell_cur + theta_{j,a_j})|; on alarm (G_t >= h) return to IDENTIFY.

    Regimes are stored by CENTERED shape (level-invariant), so a returning regime whose
    absolute reward level has drifted is still recognised, and a common up/down shift of all
    arms does not trigger a spurious drift alarm (the baseline ell_cur is re-estimated from
    the identifying aggregate at each commit).

    The confidence radius follows the draft, r(n)=c_rad*sigma*sqrt(2 log(8 K T/delta)/n),
    shape radius b(n)=2 r(n). Per the draft the exact numerical constant is not important for
    the algorithmic structure; c_rad anchors the BLOCK-level shape resolution b(m) to the
    calibrated noise/shape-separation band (see bandit_long_experiment.make_policies), while
    the sqrt(1/n) shrinkage still lets a well-visited regime certify.
    Balanced blocks are the ONLY observations used to estimate shapes; committed-arm rewards
    never update a shape (they carry no information about unplayed arms). Faithful to the
    complete algorithm / Table 1 of the draft.
    """
    ID, CERTE, CERTC, COMMIT = 0, 1, 2, 3   # CERTE=certify existing, CERTC=certify candidate

    def __init__(self, num_arms, sigma=0.05, delta=0.1, T=10000, m=6, c_rad=0.14,
                 gamma=0.0, nu=0.055, h=0.33, seed=0):
        self.K = num_arms; self.sigma = sigma; self.delta = delta; self.T = int(T)
        self.m = int(m); self.c_rad = float(c_rad); self.gamma = float(gamma)
        self.nu = float(nu); self.h = float(h)
        self.rng = np.random.RandomState(seed)
        self._L = 2.0 * np.log(8.0 * self.K * max(self.T, 2) / self.delta)   # radius log-term
        # library: per regime -> samples/arm N_j, reward-sum vector S_j (K), certified arm a_j
        self.N, self.S, self.a = [], [], []
        self.t = 0; self.mode = self.ID
        self._Zcur = -1                       # current identified/committed regime (-1 = none)
        self._reset_block()
        self._SC = None; self._nC = 0.0       # stable candidate aggregate C (ID / CERTC)
        self._cert_j = -1                     # regime being certified (CERTE)
        self._lastlevel = 0.0                 # level of the most recent accepted aggregate/block
        # COMMIT state
        self._cj = -1; self._ca = 0; self._mhat = 0.0; self.G = 0.0
        # diagnostics (shared J/mode/Z interface with TAR-UCB / LI-TAR-UCB)
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []
        self.diag_log = []                    # (t, event, psi/None, extra)

    # ---- helpers ----
    def _rr(self, m):
        return [k for _ in range(int(m)) for k in range(self.K)]

    @staticmethod
    def _phi(v):
        return v - v.mean()

    def _b(self, n):
        """Shape confidence radius b(n)=2 r(n); +inf for an unseen regime."""
        if n <= 0:
            return np.inf
        return 2.0 * self.c_rad * self.sigma * np.sqrt(self._L / n)

    def _theta(self, j):
        return self._phi(self.S[j] / self.N[j])

    def _reset_block(self):
        self._bseq = self._rr(self.m); self._bptr = 0
        self._bsum = np.zeros(self.K); self._bcnt = np.zeros(self.K)

    def _set_C(self, blk_sum):
        self._SC = blk_sum.copy(); self._nC = float(self.m)

    def _psiC(self):
        return self._phi(self._SC / self._nC)

    # ---- commitment ----
    def _commit(self, j, arm, ell_cur):
        self.mode = self.COMMIT
        self._cj = j; self._ca = int(arm); self._Zcur = j; self.G = 0.0
        self._mhat = float(ell_cur) + float(self._theta(j)[arm])
        self._SC = None; self._nC = 0.0
        self.diag_log.append((self.t, 'commit', None, (j, int(arm), self._mhat)))

    # ---- certification gates ----
    def _try_cert_existing(self, j, ell_cur):
        th = self._theta(j); bN = self._b(self.N[j])
        ahat = int(np.argmax(th))
        lcb = th[ahat] - bN
        ucb_other = max(th[k] + bN for k in range(self.K) if k != ahat)
        if lcb > ucb_other + self.gamma:
            self.a[j] = ahat
            self._commit(j, ahat, ell_cur)
            return True
        return False

    def _try_create_candidate(self):
        psi = self._psiC(); bC = self._b(self._nC)
        ahat = int(np.argmax(psi))
        cert = psi[ahat] - bC > max(psi[k] + bC for k in range(self.K) if k != ahat) + self.gamma
        separated = all(
            float(np.max(np.abs(psi - self._theta(j)))) > bC + self._b(self.N[j])
            for j in range(len(self.N)))
        if cert and separated:
            self.N.append(float(self._nC)); self.S.append(self._SC.copy()); self.a.append(ahat)
            j = len(self.N) - 1
            self._commit(j, ahat, float((self._SC / self._nC).mean()))
            self.diag_log.append((self.t, 'new_regime', psi.copy(), (j, ahat)))
            return True
        return False

    # ---- interface ----
    def choose(self, step, ctx=None):
        if self.mode == self.COMMIT:
            return self._ca
        return self._bseq[self._bptr]                 # balanced block (ID / CERTE / CERTC)

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.COMMIT:
            e = reward - self._mhat
            self.G = max(0.0, self.G + abs(e) - self.nu)
            if self.G >= self.h:                      # drift alarm -> back to IDENTIFY
                self.diag_log.append((self.t, 'alarm', None, (self._cj, self.G)))
                self.mode = self.ID; self.G = 0.0
                self._SC = None; self._nC = 0.0; self._reset_block()
            self._log()
            return

        # collecting a balanced block (ID / CERTE / CERTC)
        self._bsum[arm] += reward; self._bcnt[arm] += 1.0; self._bptr += 1
        if self._bptr >= len(self._bseq):
            v = self._bsum / np.maximum(self._bcnt, 1.0)
            psi = self._phi(v); blk_sum = self._bsum.copy()
            self._handle_block(v, psi, blk_sum)
            if self.mode != self.COMMIT:
                self._reset_block()
        self._log()

    def _handle_block(self, v, psi, blk_sum):
        bm = self._b(self.m)
        if self.mode == self.ID:
            # --- stable-block aggregation into candidate C ---
            if self._SC is None:
                self._set_C(blk_sum)
            elif float(np.max(np.abs(psi - self._psiC()))) <= bm + self._b(self._nC):
                self._SC += blk_sum; self._nC += self.m
            else:
                self._set_C(blk_sum)                  # transition block -> restart aggregate
            # --- match the aggregate against the library ---
            psiC = self._psiC(); bC = self._b(self._nC)
            M = [j for j in range(len(self.N))
                 if float(np.max(np.abs(psiC - self._theta(j)))) <= bC + self._b(self.N[j])]
            ell = float((self._SC / self._nC).mean())
            if len(M) == 1:
                j = M[0]
                self.S[j] += self._SC; self.N[j] += self._nC   # assign C to regime j
                self._Zcur = j; self._lastlevel = ell
                if self.a[j] is not None:                       # already certified -> reuse
                    self._commit(j, self.a[j], ell)
                else:
                    self.mode = self.CERTE; self._cert_j = j
                    self._try_cert_existing(j, ell)             # may commit immediately
            elif len(M) > 1:
                arms = set(self.a[j] for j in M)
                if None not in arms and len(arms) == 1:         # action-equivalent matches
                    self._commit(M[0], arms.pop(), ell)
                # else: ambiguous -> stay in IDENTIFY, keep aggregating
            else:                                               # no match -> candidate regime
                self.mode = self.CERTC
                self._try_create_candidate()

        elif self.mode == self.CERTE:
            j = self._cert_j; th = self._theta(j)
            if float(np.max(np.abs(psi - th))) <= bm + self._b(self.N[j]):   # compatible with j
                self.S[j] += blk_sum; self.N[j] += self.m
                self._lastlevel = float(v.mean())
                self._try_cert_existing(j, self._lastlevel)
            else:                                               # incompatible -> abort to IDENTIFY
                self.mode = self.ID; self._set_C(blk_sum); self._Zcur = -1

        elif self.mode == self.CERTC:
            if float(np.max(np.abs(psi - self._psiC()))) <= bm + self._b(self._nC):
                self._SC += blk_sum; self._nC += self.m         # compatible -> grow candidate
            else:
                self._set_C(blk_sum)                            # incompatible -> reset candidate
            self._try_create_candidate()

    def _log(self):
        self.Z_hist.append(self._Zcur)
        self.J_hist.append(len(self.N))
        self.mode_hist.append(self.mode)


class CGICLBTS:
    """Library Identify-and-Commit with Thompson-Guided Certification
    (paper/cg_iclb_ts_certification_algorithm.tex). Refinement of [[CGICLB]] that changes
    ONLY the certification phase; IDENTIFY (balanced-block shape matching) and COMMIT
    (deterministic play + two-sided |e_t| drift CUSUM) are unchanged.

    Two modifications relative to CGICLB:
      1. THOMPSON-GUIDED CERTIFY. In CERTIFY the algorithm no longer collects balanced
         blocks. It draws a Gaussian working posterior theta_k ~ N(thetahat_k^cert,
         sigma^2/N_k^cert), plays A_t=argmax of one sample per round, and updates only the
         played arm with the level-CENTERED reward X_t = Y_t - ell_C (ell_C = local level of
         the identifying aggregate). TS concentrates certification pulls on plausibly-optimal
         arms, so reward during certification is near-optimal even before an arm is certified
         -- this is the key win over CGICLB's balanced-block certification, which played every
         arm (including clearly-bad ones) 1/K of the time.
      2. eps_opt-OPTIMAL STOPPING. Certify ahat=argmax thetahat^cert once
         LCB_ahat + eps_opt >= max_{k!=ahat} UCB_k, with per-arm cert radius
         B_k = c_rad_cert*sigma*sqrt(2 log(8KT/delta)/N_k^cert). eps_opt is the smallest
         reward gap worth resolving: nearly-tied arms (this env's flat R1, gap 0.014, and
         R2's arm2~arm1, gap 0.005) certify a practically-optimal arm instead of stalling
         forever in exploration, at a bounded eps_opt*T_commit regret cost (draft Eq. 24).

    The library keeps TWO statistic sets per regime: raw-reward SHAPE stats (N_sh, S_sh; from
    balanced blocks only; give theta_sh=P mu_sh for MATCHING) and centered per-arm CERT stats
    (N_cert, S_cert; give theta_cert=S_cert/N_cert for the certification DECISION and the
    commit baseline mhat = ell_cur + theta_cert[a]). A block assigned to regime j (unique
    match) updates both: shape by (S_sh+=S^C, N_sh+=n_C) and cert by
    (S_cert[k]+=n_C*psi^C_k, N_cert[k]+=n_C). A candidate's cert stats are initialized from
    its aggregate (N_cert[k]=n_C, S_cert[k]=n_C*psi^C_k) and a new entry is created only once
    an arm is eps_opt-certified AND the shape stays separated from every library entry.

    Radius constants (draft: the literal 8KT/delta constant is 'not essential'): the SHAPE
    radius keeps CGICLB's calibrated c_rad (anchors block resolution b(m) to the measured
    shape-separation band, else every regime matches every other). The CERT radius uses a
    SEPARATE c_rad_cert (default 1.0, the literal bound) because certification accumulates
    genuine per-arm evidence and must not commit to the wrong arm on a couple of noisy
    samples; eps_opt -- not a shrunken radius -- is what makes near-ties terminate. Faithful
    to Table 1 / the complete-algorithm box of the draft.
    """
    ID, CERT, COMMIT = 0, 1, 2

    def __init__(self, num_arms, sigma=0.05, delta=0.1, T=10000, m=6, c_rad=0.14,
                 c_rad_cert=1.0, eps_opt=0.03, nu=0.055, h=0.33, seed=0,
                 baseline_shape=False, tt_beta=None):
        self.K = num_arms; self.sigma = sigma; self.delta = delta; self.T = int(T)
        self.m = int(m); self.c_rad = float(c_rad); self.c_rad_cert = float(c_rad_cert)
        self.eps_opt = float(eps_opt); self.nu = float(nu); self.h = float(h)
        # tt_beta: top-two Thompson sampling for certification (Russo 2016). None => plain TS
        # (faithful to the draft). A value in (0,1) fixes the draft's key weakness: plain TS
        # starves clearly-suboptimal arms, but the eps_opt stop needs max_{k!=ahat} UCB_k to
        # drop, i.e. samples ON those arms -> certification stalls (often past the episode).
        # Top-two plays the TS leader w.p. tt_beta and a re-sampled CHALLENGER (a different
        # argmax draw; fallback = the binding highest-UCB non-leader arm) otherwise, so the
        # runner-up is pulled ~(1-tt_beta) of the time and the stop fires fast. This does not
        # change the stopping rule, the statistics, or COMMIT -- only which arm CERT samples.
        self.tt_beta = None if tt_beta is None else float(tt_beta)
        self._tt_redraw = 8                       # max re-draws to find a challenger != leader
        # baseline_shape (OFF by default = faithful to draft Eq. local-baseline, which uses
        # theta^cert): when ON, the COMMIT drift baseline uses the clean balanced-block shape
        # theta_sh[a] instead of theta_cert[a]. theta_cert is a per-arm centered estimate that
        # pools TS samples across episodes and can be polluted by wrong-regime pulls collected
        # during boundary mis-identifications, biasing mhat and tripping the drift CUSUM during
        # a stable episode (false-alarm churn). theta_sh comes only from balanced blocks
        # accepted as compatible with the regime, so it stays level-consistent and unbiased.
        self.baseline_shape = bool(baseline_shape)
        self.rng = np.random.RandomState(seed)
        self._L = 2.0 * np.log(8.0 * self.K * max(self.T, 2) / self.delta)
        # library: shape stats (Nsh scalar, Ssh vector) + cert stats (Ncert, Scert vectors) + arm
        self.Nsh, self.Ssh, self.Ncert, self.Scert, self.a = [], [], [], [], []
        self.t = 0; self.mode = self.ID; self._Zcur = -1
        self._reset_block()
        self._SC = None; self._nC = 0.0             # stable identification aggregate C
        # CERT context: kind 'E' (existing regime _cj) or 'C' (candidate); level ell_C
        self._ck = None; self._cj = -1; self._cell = 0.0
        self._cNc = None; self._cSc = None; self._cSC = None; self._cnC = 0.0
        # COMMIT state
        self._cj_com = -1; self._ca = 0; self._mhat = 0.0; self.G = 0.0
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []
        self.diag_log = []

    # ---- helpers ----
    def _rr(self, m):
        return [k for _ in range(int(m)) for k in range(self.K)]

    @staticmethod
    def _phi(v):
        return v - v.mean()

    def _b(self, n):
        """Shape (matching) radius b(n)=2 r(n); +inf for an unseen shape."""
        if n <= 0:
            return np.inf
        return 2.0 * self.c_rad * self.sigma * np.sqrt(self._L / n)

    def _bcert(self, n):
        """Per-arm certification radius B_k (vectorized over n)."""
        n = np.maximum(np.asarray(n, float), 1e-9)
        return self.c_rad_cert * self.sigma * np.sqrt(self._L / n)

    def _theta_sh(self, j):
        return self._phi(self.Ssh[j] / self.Nsh[j])

    def _theta_cert_of(self, N, S):
        return S / np.maximum(N, 1e-9)

    def _reset_block(self):
        self._bseq = self._rr(self.m); self._bptr = 0
        self._bsum = np.zeros(self.K); self._bcnt = np.zeros(self.K)

    def _set_C(self, blk_sum):
        self._SC = blk_sum.copy(); self._nC = float(self.m)

    def _psiC(self):
        return self._phi(self._SC / self._nC)

    def _cert_NS(self):
        """The (N_cert, S_cert) arrays currently being certified (library entry or candidate)."""
        if self._ck == 'E':
            return self.Ncert[self._cj], self.Scert[self._cj]
        return self._cNc, self._cSc

    # ---- commitment ----
    def _commit(self, j, arm, ell_cur):
        self.mode = self.COMMIT
        self._cj_com = j; self._ca = int(arm); self._Zcur = j; self.G = 0.0
        th_a = (self._theta_sh(j)[arm] if self.baseline_shape
                else self._theta_cert_of(self.Ncert[j], self.Scert[j])[arm])
        self._mhat = float(ell_cur) + float(th_a)
        self._SC = None; self._nC = 0.0; self._ck = None
        self.diag_log.append((self.t, 'commit', (j, int(arm), self._mhat)))

    # ---- eps_opt certification test ----
    def _cert_stop(self, N, S):
        thc = self._theta_cert_of(N, S)
        ahat = int(np.argmax(thc))
        B = self._bcert(N)
        lcb_a = thc[ahat] - B[ahat]
        ucb_other = max(thc[k] + B[k] for k in range(self.K) if k != ahat)
        return (lcb_a + self.eps_opt >= ucb_other), ahat

    def _cert_try(self):
        """Check the eps_opt rule on the active cert stats; commit / create-regime if it holds."""
        N, S = self._cert_NS()
        ok, ahat = self._cert_stop(N, S)
        if not ok:
            return False
        if self._ck == 'E':
            j = self._cj; self.a[j] = ahat
            self.diag_log.append((self.t, 'certify_existing', (j, ahat)))
            self._commit(j, ahat, self._cell)
        else:                                          # candidate: require library separation
            psiC = self._phi(self._cSC / self._cnC)
            sep = all(float(np.max(np.abs(psiC - self._theta_sh(j))))
                      > self._b(self._cnC) + self._b(self.Nsh[j]) for j in range(len(self.Nsh)))
            if sep:
                self.Nsh.append(float(self._cnC)); self.Ssh.append(self._cSC.copy())
                self.Ncert.append(self._cNc.copy()); self.Scert.append(self._cSc.copy())
                self.a.append(ahat)
                j = len(self.Nsh) - 1
                self.diag_log.append((self.t, 'new_regime', (j, ahat)))
                self._commit(j, ahat, self._cell)
            # (certified but not separated cannot occur: a no-match entry already implies
            #  separation, and neither psi^C, n_C nor the library change during TS-CERT)
        return True

    def _enter_cert_existing(self, j, ell):
        self.mode = self.CERT; self._ck = 'E'; self._cj = j; self._cell = float(ell)
        self._cert_try()                               # may commit immediately (already certified)

    def _enter_cert_candidate(self, ell):
        self.mode = self.CERT; self._ck = 'C'; self._cell = float(ell)
        psiC = self._psiC()
        self._cSC = self._SC.copy(); self._cnC = float(self._nC)
        self._cNc = np.full(self.K, self._nC, float); self._cSc = self._nC * psiC
        self._cert_try()

    # ---- interface ----
    def choose(self, step, ctx=None):
        if self.mode == self.COMMIT:
            return self._ca
        if self.mode == self.CERT:                     # Thompson-guided certification sample
            N, S = self._cert_NS()
            thc = self._theta_cert_of(N, S)
            std = self.sigma / np.sqrt(np.maximum(N, 1e-9))
            leader = int(np.argmax(self.rng.normal(thc, std)))
            if self.tt_beta is None or self.rng.random() < self.tt_beta:
                return leader                          # plain TS, or top-two leader pull
            for _ in range(self._tt_redraw):           # top-two: re-sample a challenger != leader
                c = int(np.argmax(self.rng.normal(thc, std)))
                if c != leader:
                    return c
            ucb = thc + self._bcert(N); ucb[leader] = -np.inf   # fallback: binding non-leader arm
            return int(np.argmax(ucb))
        return self._bseq[self._bptr]                  # ID: balanced block

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.COMMIT:
            e = reward - self._mhat
            self.G = max(0.0, self.G + abs(e) - self.nu)
            if self.G >= self.h:
                self.diag_log.append((self.t, 'alarm', (self._cj_com, self.G)))
                self.mode = self.ID; self.G = 0.0
                self._SC = None; self._nC = 0.0; self._reset_block()
            self._log(); return

        if self.mode == self.CERT:                     # centered single-arm cert update
            N, S = self._cert_NS()
            S[arm] += reward - self._cell; N[arm] += 1.0
            self._cert_try()
            self._log(); return

        # ID: accumulate a balanced block
        self._bsum[arm] += reward; self._bcnt[arm] += 1.0; self._bptr += 1
        if self._bptr >= len(self._bseq):
            v = self._bsum / np.maximum(self._bcnt, 1.0)
            psi = self._phi(v); blk_sum = self._bsum.copy()
            self._handle_id_block(v, psi, blk_sum)
            if self.mode == self.ID:                   # stayed in ID -> fresh block next
                self._reset_block()
        self._log()

    def _handle_id_block(self, v, psi, blk_sum):
        bm = self._b(self.m)
        # --- stable-block aggregation into candidate C ---
        if self._SC is None:
            self._set_C(blk_sum)
        elif float(np.max(np.abs(psi - self._psiC()))) <= bm + self._b(self._nC):
            self._SC += blk_sum; self._nC += self.m
        else:
            self._set_C(blk_sum)
        psiC = self._psiC(); bC = self._b(self._nC)
        ell = float((self._SC / self._nC).mean())
        M = [j for j in range(len(self.Nsh))
             if float(np.max(np.abs(psiC - self._theta_sh(j)))) <= bC + self._b(self.Nsh[j])]
        if len(M) == 1:
            j = M[0]
            self.Ssh[j] += self._SC; self.Nsh[j] += self._nC          # shape update
            self.Scert[j] += self._nC * psiC; self.Ncert[j] += self._nC   # cert update from block
            self._Zcur = j
            if self.a[j] is not None:                                  # already certified?
                ok, _ = self._cert_stop(self.Ncert[j], self.Scert[j])
                if ok:
                    self._commit(j, self.a[j], ell); return
            self._enter_cert_existing(j, ell)                         # (re)certify with TS
        elif len(M) > 1:
            arms = set(self.a[j] for j in M)
            if None not in arms and len(arms) == 1:                   # action-equivalent
                self._commit(M[0], arms.pop(), ell)
            # else ambiguous -> stay in ID
        else:                                                         # no match -> candidate
            self._enter_cert_candidate(ell)

    def _log(self):
        self.Z_hist.append(self._Zcur)
        self.J_hist.append(len(self.Nsh))
        self.mode_hist.append(self.mode)


class CGICLBR:
    """Library Identify-and-RESUME with Thompson Sampling (CG-ICLB-R).

    Variant of [[CGICLBTS]] that removes the CERTIFY/COMMIT split entirely. Instead of
    certifying a single best arm (top-two TS + eps_opt stopping) and then committing to it
    deterministically, each regime STORES its accumulated per-arm Thompson-sampling evidence
    (Nts, Sts) and, on re-entry, RESTORES (resumes) that posterior rather than starting over.
    Stable operation is just Thompson sampling on the resumed posterior. Two consequences:

      * No stopping rule => no need for top-two. Top-two only ever existed to force the
        runner-up UCBs down so the eps_opt stop could fire (else plain TS starves the
        suboptimal arms and certification stalls). With no stop, plain TS suffices: TS's own
        concentration makes play converge onto the best arm, so commit-like determinism
        EMERGES as evidence accumulates. Drops eps_opt, c_rad_cert AND tt_beta.
      * Cross-visit accumulation. A recurrent regime's total exploration regret is
        O(sum_k log(cumulative time in j)/Delta_k), not per-visit -- strong under frequent
        switching, because each visit resumes a sharp posterior instead of re-paying a fresh
        certification. The trade-off vs deterministic COMMIT is the long-episode TAIL: TS
        never fully stops sampling losers within a visit, whereas COMMIT pays zero there.

    IDENTIFY is UNCHANGED from CGICLB/-TS: balanced blocks -> stable aggregate C ->
    shape-distance match set. Matching uses the balanced-block SHAPE stats (Nsh, Ssh;
    theta_sh = P mu_sh) -- the only reliable full-shape estimate under possible level shifts;
    unbalanced TS pulls cannot recover it. On a UNIQUE match we resume regime j (the block's
    centered evidence also folds into its TS stats). On NO match we create the regime
    IMMEDIATELY (a no-match aggregate is already shape-separated -- there is no certification
    to wait for) and resume it. On a MULTIPLE match we stay in ID and collect more blocks to
    disambiguate.

    DRIFT DETECTION replaces the single deterministic-arm |e_t| CUSUM with a PER-ARM
    centered-residual CUSUM, because RUN plays a shifting TS mixture, not one fixed arm. At
    RUN entry we SNAPSHOT a fixed baseline base_k = theta_ts_k (accumulated estimate) and the
    identifying level ell_j; for each played arm we accumulate
    G_k = max(0, G_k + |X_t - base_k| - nu), with X_t = Y_t - ell_j, and alarm (-> ID) if any
    G_k >= h. Detecting against the FIXED snapshot (not the live estimate) stops the detector
    from blinding itself by tracking the drift; a large accumulated Nts (whose live estimate
    barely moves) makes the played-arm detector sharp. TS keeps most pulls on the best arm, so
    coverage there is strong -- but drift confined to an UNPLAYED arm stays invisible (the
    same committed-arm visibility limit as COMMIT). We detect FIRST, then fold X_t into
    (Nts, Sts) so the live sample is measured before it is absorbed.

    Restoration is FULL by default (cap_N=None). cap_N caps the effective per-arm count used
    for the posterior WIDTH on resume, so a very sharp restored posterior can still explore/
    adapt if a regime's means drift ACROSS recurrences (real-deployment concept drift). The
    recurrent-regime simulator has exactly-recurring fixed shapes, so full restore is correct
    there. NOTE (level re-anchoring): ell_j is snapshotted from the identifying block; a
    benign WITHIN-regime level shift would bias residuals and could false-alarm, exactly as
    COMMIT's |Y-(ell+theta)| does -- in this sim the per-regime level is constant so it does
    not arise; a real deployment would re-anchor ell from periodic balanced blocks.
    """
    ID, RUN = 0, 1

    def __init__(self, num_arms, sigma=0.05, delta=0.1, T=10000, m=6, c_rad=0.14,
                 nu=0.065, h=0.4, cap_N=None, fast_reacquire=False, reacq_margin=0.0,
                 seed=0):
        self.K = num_arms; self.sigma = sigma; self.delta = delta; self.T = int(T)
        self.m = int(m); self.c_rad = float(c_rad)
        self.nu = float(nu); self.h = float(h)
        self.cap_N = None if cap_N is None else float(cap_N)
        # fast_reacquire (CG-ICLB-R2): cut the balanced-block ID cost that R re-pays every
        # short episode (28.6% of steps on r1freq). Strict unique-match needs the aggregate
        # radius b(n_C) to shrink until EVERY other regime leaves its band -- many round-robin
        # blocks. But after an alarm the env almost always RETURNED to a known library regime,
        # so instead we resume the NEAREST regime as soon as (a) it is a valid match (within
        # radius) and (b) it beats the runner-up by reacq_margin (a fixed shape-separation
        # margin guarding against a wrong resume + its baseline pollution). Fires often after
        # ONE block. New-regime creation stays STRICT (no shortcut) so we never merge/spawn
        # spuriously. reacq_margin=0 recovers the plain nearest-valid rule (use ~0.5*dsh).
        self.fast_reacquire = bool(fast_reacquire); self.reacq_margin = float(reacq_margin)
        self.rng = np.random.RandomState(seed)
        self._L = 2.0 * np.log(8.0 * self.K * max(self.T, 2) / self.delta)
        # library: shape stats (Nsh scalar, Ssh vector; for MATCHING) + TS stats
        # (Nts, Sts per-arm vectors; centered evidence, restored/resumed across visits)
        self.Nsh, self.Ssh, self.Nts, self.Sts = [], [], [], []
        self.t = 0; self.mode = self.ID; self._Zcur = -1
        self._reset_block()
        self._SC = None; self._nC = 0.0                 # stable identification aggregate C
        # RUN state: regime, level, snapshot baseline, per-arm CUSUM
        self._rj = -1; self._rell = 0.0; self._base = None; self.G = None
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []
        self.diag_log = []

    # ---- helpers (shared shape/block machinery with CGICLBTS) ----
    def _rr(self, m):
        return [k for _ in range(int(m)) for k in range(self.K)]

    @staticmethod
    def _phi(v):
        return v - v.mean()

    def _b(self, n):
        if n <= 0:
            return np.inf
        return 2.0 * self.c_rad * self.sigma * np.sqrt(self._L / n)

    def _theta_sh(self, j):
        return self._phi(self.Ssh[j] / self.Nsh[j])

    def _reset_block(self):
        self._bseq = self._rr(self.m); self._bptr = 0
        self._bsum = np.zeros(self.K); self._bcnt = np.zeros(self.K)

    def _set_C(self, blk_sum):
        self._SC = blk_sum.copy(); self._nC = float(self.m)

    def _psiC(self):
        return self._phi(self._SC / self._nC)

    def _theta_ts(self, j):
        return self.Sts[j] / np.maximum(self.Nts[j], 1e-9)

    # ---- resume / run ----
    def _resume_with_block(self, j, ell, psiC):
        self.Ssh[j] += self._SC; self.Nsh[j] += self._nC              # shape update
        self.Sts[j] += self._nC * psiC; self.Nts[j] += self._nC       # TS update from block
        self._enter_run(j, ell)

    def _enter_run(self, j, ell):
        self.mode = self.RUN; self._rj = j; self._rell = float(ell); self._Zcur = j
        self._base = self._theta_ts(j).copy()           # fixed drift baseline snapshot
        self.G = np.zeros(self.K)
        self.diag_log.append((self.t, 'resume', (j, int(np.argmax(self._base)))))

    # ---- interface ----
    def choose(self, step, ctx=None):
        if self.mode == self.RUN:
            j = self._rj
            th = self._theta_ts(j)
            Neff = self.Nts[j] if self.cap_N is None else np.minimum(self.Nts[j], self.cap_N)
            std = self.sigma / np.sqrt(np.maximum(Neff, 1e-9))
            return int(np.argmax(self.rng.normal(th, std)))     # plain Thompson sampling
        return self._bseq[self._bptr]                           # ID: balanced block

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.RUN:
            j = self._rj
            X = reward - self._rell                              # level-centered reward
            e = X - self._base[arm]                              # residual vs fixed snapshot
            self.G[arm] = max(0.0, self.G[arm] + abs(e) - self.nu)
            if self.G[arm] >= self.h:                            # drift alarm -> re-identify
                self.diag_log.append((self.t, 'alarm', (j, int(arm), float(self.G[arm]))))
                self.mode = self.ID; self._SC = None; self._nC = 0.0
                self._reset_block(); self._base = None; self.G = None
                self._log(); return
            self.Sts[j][arm] += X; self.Nts[j][arm] += 1.0      # fold into resumed posterior
            self._log(); return

        # ID: accumulate a balanced block
        self._bsum[arm] += reward; self._bcnt[arm] += 1.0; self._bptr += 1
        if self._bptr >= len(self._bseq):
            v = self._bsum / np.maximum(self._bcnt, 1.0)
            psi = self._phi(v); blk_sum = self._bsum.copy()
            self._handle_id_block(v, psi, blk_sum)
            if self.mode == self.ID:
                self._reset_block()
        self._log()

    def _handle_id_block(self, v, psi, blk_sum):
        bm = self._b(self.m)
        if self._SC is None:
            self._set_C(blk_sum)
        elif float(np.max(np.abs(psi - self._psiC()))) <= bm + self._b(self._nC):
            self._SC += blk_sum; self._nC += self.m
        else:
            self._set_C(blk_sum)
        psiC = self._psiC(); bC = self._b(self._nC)
        ell = float((self._SC / self._nC).mean())
        nreg = len(self.Nsh)
        if nreg > 0:
            D = np.array([float(np.max(np.abs(psiC - self._theta_sh(j)))) for j in range(nreg)])
            radius = np.array([bC + self._b(self.Nsh[j]) for j in range(nreg)])
            inM = D <= radius
            M = np.nonzero(inM)[0]
        else:
            D = None; inM = None; M = np.array([], int)
        if len(M) == 1:                                          # strict unique match -> RESUME
            self._resume_with_block(int(M[0]), ell, psiC); return
        if self.fast_reacquire and nreg >= 1:                    # fast re-acquire: nearest+margin
            order = np.argsort(D)
            j1 = int(order[0]); d2 = D[int(order[1])] if nreg > 1 else np.inf
            if inM[j1] and (d2 - D[j1]) >= self.reacq_margin:
                self.diag_log.append((self.t, 'reacquire', (j1, float(d2 - D[j1]))))
                self._resume_with_block(j1, ell, psiC); return
        if len(M) > 1:                                          # ambiguous -> collect more blocks
            pass
        else:                                                  # no match -> create regime NOW
            self.Nsh.append(float(self._nC)); self.Ssh.append(self._SC.copy())
            self.Nts.append(np.full(self.K, self._nC, float)); self.Sts.append(self._nC * psiC)
            j = len(self.Nsh) - 1
            self.diag_log.append((self.t, 'new_regime', (j, int(np.argmax(psiC)))))
            self._enter_run(j, ell)

    def _log(self):
        self.Z_hist.append(self._Zcur)
        self.J_hist.append(len(self.Nsh))
        self.mode_hist.append(self.mode)


class RATS:
    """Recurrence-Aware Thompson Sampling (RA-TS).

    The distilled version of the CG-ICLB family: *just* Thompson sampling + a drift detector +
    a warm-start library, combined so that regime identification EMERGES from TS's own
    exploration. There is no balanced-block IDENTIFY scheduler, no CERT/COMMIT split, no
    eps_opt, no top-two, and no explicit level estimation. Two internal phases only:

      ACQUIRE -- the working posterior is reset to a DIFFUSE prior, so plain Thompson sampling
        naturally spreads across arms (the wide prior makes early argmax draws rotate). No
        forced round-robin: exploration is 100% TS-native. Once every arm has been sampled at
        least n_min times we have a shape estimate and try to identify the regime.
      TRACK -- once a regime is identified, TS runs on that regime's accumulated posterior
        (RESTORED from the library and kept accumulating -> cross-visit recurrence benefit).
        Commit-like determinism emerges as the posterior concentrates. A per-arm CUSUM on the
        played arm's |Y - base_k| (base = fixed snapshot of the regime mean at lock time)
        detects drift; an alarm returns to ACQUIRE (the library entry keeps its evidence).

    IDENTIFICATION is level-free by construction. The library stores RAW per-arm posteriors
    (N_g, S_g -> mu_g); matching compares the CENTERED shape psi = mu - mean(mu) of the
    acquisition estimate to each regime's centered shape theta_g = mu_g - mean(mu_g). Centering
    over the (fully covered) arm set cancels any common reward level, so a regime is recognized
    regardless of level shifts without ever estimating the level -- equivalently, matching on
    the level-invariant pairwise gaps mu_i - mu_j. A regime is adopted (warm start: fold the
    acquisition buffer into the library entry) only when it is the NEAREST within match_tol AND
    beats the runner-up by match_margin; otherwise ACQUIRE keeps collecting until the estimate
    sharpens. A genuinely new shape (all arms matured to n_new and separated from every entry
    by match_tol+match_margin) is stored as a new regime. These guards keep a noisy unbalanced
    shape estimate from mis-adopting or spuriously spawning regimes.

    Trade vs CG-ICLB-R2: R2 identifies from a clean balanced block; RA-TS identifies from
    unbalanced TS samples (noisier) but pays no separate round-robin phase -- exploration does
    double duty (learn + identify). The drift detector shares R2's snapshot-baseline design so
    it cannot blind itself by tracking the drift.

    DELAYED COMMIT (commit_delay=q, paper/ra_ts_paper_skeleton.tex Sec. "Track Mode"): a proof
    device against library contamination. Between a hidden regime switch and the alarm that
    catches it, TRACK keeps playing and would otherwise commit those post-switch samples into
    the OLD regime's library entry -- corrupting (N_g,S_g) for every future visit. With q>0,
    a played (arm,reward) is held in a short FIFO queue and only folded into (N_g,S_g) once it
    has sat there q rounds with NO alarm in between; if an alarm fires, the whole queue is
    discarded (nothing in it is committed), so at most the last q pre-alarm samples are ever at
    risk and none of them reach the persistent library. The detector itself is unaffected: it
    reads every reward immediately off the (uncommitted) observation, exactly as before -- only
    the PERSISTENT statistics lag by q. Consequently the Thompson posterior used to CHOOSE arms
    (built from (N_g,S_g)) also lags the true committed count by up to q samples: this is the
    "delayed-feedback bandit" cost the theory must charge for purity, not a free correctness fix.
    q=0 (default) recovers the original immediate-commit RA-TS-F exactly (bit-identical).

    ROLLBACK (rollback=True): the better purity mechanism -- commit every TRACK sample
    IMMEDIATELY (so the acting posterior is never stale, unlike commit_delay), but on an alarm
    RETROACTIVELY un-commit the contiguous run of samples that drove the detector up, i.e. all
    samples committed since the ALARMING arm's per-arm CUSUM last reset to zero (the standard
    CUSUM change-point estimate). This removes the post-switch dirty tail -- the large-residual
    samples carrying the inter-regime mean shift -- from the OLD entry, shrinking the impurity
    bias eps_imp (and hence the tolerance eps_tol) that the immediate-commit analysis must
    otherwise carry. CRUCIAL: the discarded set is a CONTIGUOUS RUN (in commit order), not the
    individually large-|residual| samples; a size-filtered (non-contiguous) discard would be a
    value-dependent selection that biases the retained clean mean (asymmetric truncation) and
    falls outside the good-event stretch machinery. Because it un-commits rather than delays, it
    avoids the delayed-feedback (staleness) cost that made commit_delay hurt empirically. Mutually
    exclusive with commit_delay in spirit; rollback bookkeeping runs only in the immediate-commit
    path. rollback=False (default) is bit-identical to plain RA-TS-F.
    """
    ACQUIRE, TRACK = 0, 1

    def __init__(self, num_arms, sigma=0.05, mu0=0.5, tau0=0.5, n_min=2, n_new=8,
                 match_tol=0.08, match_margin=0.04, nu=0.065, h=0.4, explore_floor=False,
                 commit_delay=0, rollback=False, seed=0):
        self.K = num_arms; self.sigma = float(sigma)
        self.mu0 = float(mu0); self.tau0 = float(tau0)
        self.n_min = int(n_min); self.n_new = int(n_new)
        self.match_tol = float(match_tol); self.match_margin = float(match_margin)
        self.nu = float(nu); self.h = float(h)
        # explore_floor: the minimal identification probe. Pure diffuse-TS provably STALLS in
        # ACQUIRE -- identification needs loser-arm samples, but TS's job is to stop sampling
        # losers, so coverage (n_min per arm) is never reached and the detector (TRACK-only)
        # stays off while regime switches corrupt the buffer. With the floor ON, ACQUIRE plays
        # the LEAST-sampled arm until every arm has n_min samples, then hands off to TS/matching
        # -- a bounded probe of <= n_min*K forced pulls per re-acquire, far lighter than a
        # balanced-block ID phase. OFF = faithful 'pure TS diffuse reset' (stalls; kept for A/B).
        self.explore_floor = bool(explore_floor)
        self.commit_delay = int(commit_delay)            # q: rounds an observation must wait
        self.rollback = bool(rollback)                   # un-commit the change-point run on alarm
        self.rng = np.random.RandomState(seed)
        self.N, self.S = [], []                         # library: raw per-arm (count, sum)
        self.t = 0; self.mode = self.ACQUIRE; self._Zcur = -1
        self._n = np.zeros(self.K); self._s = np.zeros(self.K)   # ACQUIRE working buffer
        self._g = -1; self._base = None; self.G = None          # TRACK state
        self._queue = []                                 # pending (arm, reward), FIFO, len<=q
        self._tlog = []                                  # rollback: committed (arm,reward) this TRACK
        self._rpos = None                                # rollback: per-arm _tlog len at last G reset
        self.Z_hist, self.J_hist, self.mode_hist = [], [], []
        self.surp_hist = []                              # per-step detector surprise |Y-b| (nan in ACQUIRE)
        self.diag_log = []

    @staticmethod
    def _center(mu):
        return mu - mu.mean()

    def _post(self, n, s):
        """Conjugate Gaussian posterior over the mean: prior N(mu0,tau0^2), noise var sigma^2."""
        prec = 1.0 / self.tau0**2 + n / self.sigma**2
        var = 1.0 / prec
        mean = var * (self.mu0 / self.tau0**2 + s / self.sigma**2)
        return mean, var

    def _theta_g(self, g):
        return self._center(self.S[g] / np.maximum(self.N[g], 1e-9))

    def _enter_track(self, g):
        self.mode = self.TRACK; self._g = g; self._Zcur = g
        self._base = (self.S[g] / np.maximum(self.N[g], 1e-9)).copy()   # fixed drift baseline
        self.G = np.zeros(self.K)
        self._n[:] = 0.0; self._s[:] = 0.0                             # clear acquire buffer
        self._queue = []                                               # clear commit-delay queue
        self._tlog = []; self._rpos = np.zeros(self.K, int)            # rollback change-point log

    def choose(self, step, ctx=None):
        if self.mode == self.ACQUIRE:
            if self.explore_floor and np.any(self._n < self.n_min):    # bounded coverage probe
                return int(np.argmin(self._n))
            n, s = self._n, self._s
        else:
            n, s = self.N[self._g], self.S[self._g]
        mean, var = self._post(n, s)
        return int(np.argmax(self.rng.normal(mean, np.sqrt(var))))     # Thompson sample

    def update(self, arm, reward):
        self.t += 1
        if self.mode == self.TRACK:
            g = self._g
            # detector reads the raw reward immediately, regardless of commit delay
            surp = abs(reward - self._base[arm]); self.surp_hist.append(surp)
            self.G[arm] = max(0.0, self.G[arm] + surp - self.nu)
            if self.G[arm] >= self.h:                                  # drift alarm -> ACQUIRE
                nroll = 0
                if self.rollback and self._rpos is not None:           # un-commit change-point run
                    start = int(self._rpos[arm])
                    for oarm, oreward in self._tlog[start:]:
                        self.S[g][oarm] -= oreward; self.N[g][oarm] -= 1.0
                    nroll = len(self._tlog) - start
                self.diag_log.append((self.t, 'alarm', (g, int(arm), float(self.G[arm]),
                                                          len(self._queue), nroll)))
                self.mode = self.ACQUIRE; self._n[:] = 0.0; self._s[:] = 0.0
                self._base = None; self.G = None; self._Zcur = -1
                self._queue = []                        # discard: nothing pending is committed
                self._tlog = []; self._rpos = None
                self._log(); return
            if self.commit_delay <= 0:                                 # immediate commit (q=0)
                self.S[g][arm] += reward; self.N[g][arm] += 1.0
                if self.rollback:
                    self._tlog.append((arm, reward))
                    if self.G[arm] <= 0.0:              # detector reset here -> new change-point
                        self._rpos[arm] = len(self._tlog)
            else:                                                      # delayed commit (q>0)
                self._queue.append((arm, reward))
                if len(self._queue) > self.commit_delay:
                    oarm, oreward = self._queue.pop(0)
                    self.S[g][oarm] += oreward; self.N[g][oarm] += 1.0
            self._log(); return
        # ACQUIRE
        self.surp_hist.append(np.nan)
        self._s[arm] += reward; self._n[arm] += 1.0
        self._try_identify()
        self._log()

    def _try_identify(self):
        # DECIDE-AT-COVERAGE. Pure diffuse-TS covers all arms in a few steps but then abandons
        # the loser arms, so we must decide the moment coverage (n_min per arm) is reached --
        # a high maturity gate would never be met (TS stops sampling losers) and ACQUIRE would
        # stall with the detector off, letting a regime switch corrupt the buffer. So: adopt the
        # nearest library regime if within match_tol, else create a new regime. The margin is a
        # soft tie-break, not a wait condition (waiting cannot help under pure TS). Residual
        # match errors are caught by TRACK's drift detector, which re-triggers ACQUIRE.
        if np.any(self._n < self.n_min):                              # need coverage on every arm
            return
        psi = self._center(self._s / np.maximum(self._n, 1e-9))
        nreg = len(self.N)
        if nreg > 0:
            D = np.array([float(np.max(np.abs(psi - self._theta_g(g)))) for g in range(nreg)])
            g1 = int(np.argmin(D))
            if D[g1] <= self.match_tol:                               # nearest regime is a match
                self.N[g1] += self._n; self.S[g1] += self._s          # warm start (fold buffer)
                self.diag_log.append((self.t, 'match', (g1, float(D[g1]))))
                self._enter_track(g1); return
        self.N.append(self._n.copy()); self.S.append(self._s.copy())  # no match -> new regime
        g = len(self.N) - 1
        self.diag_log.append((self.t, 'new_regime', (g, int(np.argmax(psi)))))
        self._enter_track(g)

    def _log(self):
        self.Z_hist.append(self._Zcur)
        self.J_hist.append(len(self.N))
        self.mode_hist.append(self.mode)


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


def rule_based_floor(oracle_table):
    """Reward level that separates a genuinely degraded regime (low achievable reward
    for EVERY arm) from a normal one -- a naive, reward-only alarm threshold. Set midway
    between the worst regime's BEST achievable reward and the worst reward achievable by
    ANY arm in any of the OTHER regimes. This is the kind of static level an operator
    could set once from historical performance data; it requires no real-time regime
    label. Used to calibrate RuleBased.floor."""
    T = np.asarray(oracle_table, float)              # (arm, regime)
    best_per_regime = T.max(axis=0)
    worst_idx = int(np.argmin(best_per_regime))
    others = np.delete(T, worst_idx, axis=1)
    return float(0.5 * (best_per_regime[worst_idx] + others.min()))


class RuleBased:
    """Naive 'operator practice' baseline -- NO access to the true regime/context, only
    to the OBSERVED reward stream (mirrors what a control room actually sees). Two
    triggers decide when to act:
      (1) an alarm: an EWMA of the observed reward on the currently active controller
          stays below a calibrated `floor` for `q_bad` consecutive windows ("this looks
          degraded");
      (2) a routine periodic re-check every `t_periodic` windows, independent of the
          alarm (operators re-survey configurations on a schedule, not only when alarmed).
    Either trigger starts the SAME short round-robin probe (m_probe samples/arm, all K
    arms including the current one) and commits to whichever measured the highest mean
    reward, then re-arms after a `dwell` cooldown. This is deliberately simple/heuristic
    (no confidence bounds, no shape signal) and shares the blind spot of any pure
    reward-level detector: regime changes that don't move the achievable reward LEVEL
    (e.g. this env's R0<->R2) are invisible to trigger (1) and can only be caught by the
    periodic re-check (2), with real latency -- a genuine, expected limitation, not a bug.
    """
    def __init__(self, num_arms=3, floor=0.45, ewma_beta=0.15, q_bad=3, m_probe=5,
                 dwell=300, t_periodic=180, seed=0):
        self.K = num_arms
        self.floor, self.beta = floor, ewma_beta
        self.q_bad, self.m_probe, self.dwell, self.t_periodic = q_bad, m_probe, dwell, t_periodic
        self.rng = np.random.RandomState(seed)
        self.arm = 0
        self.ewma = None
        self.bad_count = 0
        self.cooldown = 0
        self.since_check = 0
        self.probing = False
        self.probe_arm = 0
        self.probe_sum = np.zeros(num_arms)
        self.probe_n = np.zeros(num_arms, int)

    def _start_probe(self):
        self.probing = True
        self.probe_arm = 0
        self.probe_sum[:] = 0.0
        self.probe_n[:] = 0
        self.bad_count = 0
        self.since_check = 0

    def choose(self, step, ctx=None):
        return self.probe_arm if self.probing else self.arm

    def update(self, arm, reward):
        if self.probing:
            self.probe_sum[arm] += reward
            self.probe_n[arm] += 1
            if self.probe_n[arm] >= self.m_probe:
                self.probe_arm += 1
                if self.probe_arm >= self.K:
                    means = self.probe_sum / np.maximum(self.probe_n, 1)
                    self.arm = int(np.argmax(means))
                    self.probing = False
                    self.ewma = float(means[self.arm])
                    self.cooldown = self.dwell
            return
        self.ewma = reward if self.ewma is None else (1 - self.beta) * self.ewma + self.beta * reward
        self.since_check += 1
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        self.bad_count = self.bad_count + 1 if self.ewma < self.floor else 0
        if self.bad_count >= self.q_bad or self.since_check >= self.t_periodic:
            self._start_probe()
