import numpy as np
import scipy.stats as stats
from collections import deque
from abc import ABC, abstractmethod

class BanditAlgorithm(ABC):
    """
    Abstract base class for Multi-Armed Bandit algorithms.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.t = 0 # Time step
        self.c = {a: 0 for a in range(num_arms)} # Default cluster assignment

    @property
    def num_clusters(self):
        """Returns the number of latent states/clusters."""
        return 1

    @abstractmethod
    def select_arm(self):
        """Selects an arm (controller index) to play."""
        pass

    @abstractmethod
    def update(self, arm, reward):
        """Updates internal state based on observed reward."""
        pass

    @abstractmethod
    def reset(self):
        """Resets algorithm state."""
        pass


class FixedBandit(BanditAlgorithm):
    """
    Static Bandit that always selects the same arm.
    Used for 'static controller' mode.
    """
    def __init__(self, num_arms, arm_index=0):
        super().__init__(num_arms)
        self.arm_index = arm_index

    def select_arm(self):
        return self.arm_index

    def update(self, arm, reward):
        pass

    def reset(self):
        pass


class DPSKF(BanditAlgorithm):
    """
    Dirichlet Process Switching Kalman Filter (DP-SKF) Bandit.
    
    This algorithm is designed for non-stationary environments where the reward 
    distribution can jump between different regimes (e.g., environmental states).
    
    Logic:
    1. Inference: Uses a library of Kalman Filters (KFs), each representing a latent state.
       Bayesian Model Averaging (BMA) with a Dirichlet Process prior determines the 
       likelihood of the current state being an existing one or a new one.
    2. Selection: Thompson Sampling is performed by drawing samples from the 
       weighted mixture of Kalman Filter posteriors.
    3. Update: KFs are updated proportionally to their responsibility (Soft Assignment)
       using standard Kalman equations. System noise (Q) allows means to drift over time.
    """
    def __init__(self, num_arms, dp_alpha=1.0, window_size=50, Q=1e-4, R=1e-2, 
                 prior_mu=167.5, prior_var=10.0, log_file=None):
        """
        Args:
            num_arms: Number of available controllers.
            dp_alpha: Concentration parameter for Dirichlet Process (higher means more clusters).
            window_size: History window for likelihood evaluation.
            Q: Process noise covariance (drift rate).
            R: Observation noise covariance.
            prior_mu: Initial guess for reward mean.
            prior_var: Initial guess for reward variance.
        """
        super().__init__(num_arms)
        self.dp_alpha = dp_alpha
        self.window_size = window_size
        self.Q = Q 
        self.R = R 
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.log_file = log_file
        
        self.reset()
        
        print(f"Initializing DP-SKF (alpha={dp_alpha}, W={window_size}, Q={Q}, R={R})")

    def reset(self):
        self.t = 0
        self.kfs = []
        self._add_new_kf()
        
        # Hard assignment for simplified logging (maps arm -> most likely cluster)
        self.c = {a: 0 for a in range(self.num_arms)}
        
        # Recent reward history per arm for likelihood evaluation
        self.arm_history = {a: deque(maxlen=self.window_size) for a in range(self.num_arms)}
        
    def _add_new_kf(self):
        """Creates a new latent state represented by a Kalman Filter for all arms."""
        new_kf = {
            'mu': np.ones(self.num_arms) * self.prior_mu,
            'P': np.ones(self.num_arms) * self.prior_var,
            'count': 0.0 # Effective sample count
        }
        self.kfs.append(new_kf)
        
    def _evaluate_kf_likelihood(self, kf_idx, arm, rewards):
        """Evaluates log-likelihood of history sequence given a specific KF."""
        if not rewards:
            return 0.0
            
        # Use a short evaluation window (last 3 points) to avoid being too 
        # sensitive to historical drifts within a regime.
        eval_window = min(len(rewards), 3)
        eval_rewards = list(rewards)[-eval_window:]
        
        kf = self.kfs[kf_idx]
        mu = kf['mu'][arm]
        var = kf['P'][arm] + self.R 
        
        return np.sum(stats.norm.logpdf(eval_rewards, loc=mu, scale=np.sqrt(var)))

    def _get_soft_assignments(self, arm):
        """
        Calculates posterior probability of each active KF and a potential new KF.
        Uses the CRP (Chinese Restaurant Process) logic for the DP prior.
        """
        num_kfs = len(self.kfs)
        recent_rewards = list(self.arm_history[arm])
        
        log_liks = np.zeros(num_kfs)
        for k in range(num_kfs):
            log_liks[k] = self._evaluate_kf_likelihood(k, arm, recent_rewards)
            
        # Log-Likelihood for a hypothetical new cluster (pure prior)
        temp_prior_var = self.prior_var + self.R
        eval_window = min(len(recent_rewards), 3)
        eval_rewards = recent_rewards[-eval_window:] if recent_rewards else []
        log_lik_new = np.sum(stats.norm.logpdf(eval_rewards, loc=self.prior_mu, scale=np.sqrt(temp_prior_var))) if eval_rewards else 0.0
        
        # Dirichlet Process (CRP) Prior
        # N = number of arms with any data
        N = sum(len(self.arm_history[a]) > 0 for a in range(self.num_arms))
        counts = np.zeros(num_kfs)
        for a in range(self.num_arms):
            if len(self.arm_history[a]) > 0:
                counts[self.c[a]] += 1
                
        if N == 0:
            priors = np.ones(num_kfs) / num_kfs
            prior_new = self.dp_alpha
        else:
            priors = counts / (N - 1 + self.dp_alpha + 1e-9)
            prior_new = self.dp_alpha / (N - 1 + self.dp_alpha + 1e-9)
            
        # Normalization with LogSumExp
        log_posteriors = np.zeros(num_kfs + 1)
        for k in range(num_kfs):
            log_posteriors[k] = np.log(priors[k] + 1e-300) + log_liks[k]
        log_posteriors[-1] = np.log(prior_new + 1e-300) + log_lik_new
        
        max_log = np.max(log_posteriors)
        probs = np.exp(log_posteriors - max_log)
        probs /= np.sum(probs)
        
        return probs

    def select_arm(self):
        """Thompson Sampling from the weighted mixture of KFs."""
        self.t += 1
        sampled_theta = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            probs = self._get_soft_assignments(arm)
            sampled_k = np.random.choice(len(probs), p=probs)
            
            if sampled_k == len(self.kfs):
                # Sample from prior for a hypothesized new cluster
                self.c[arm] = np.argmax(probs[:-1]) if len(probs) > 1 else 0
                sampled_theta[arm] = np.random.normal(self.prior_mu, np.sqrt(self.prior_var + self.R))
            else:
                self.c[arm] = np.argmax(probs[:-1]) 
                kf = self.kfs[sampled_k]
                sampled_theta[arm] = np.random.normal(kf['mu'][arm], np.sqrt(kf['P'][arm]))
            
        # Tie-break randomization
        return int(np.random.choice(np.flatnonzero(sampled_theta == np.max(sampled_theta))))

    def update(self, arm, reward):
        """Standard Kalman Update weighted by responsibilities."""
        self.arm_history[arm].append(reward)
        
        # Predict: Increase uncertainty due to process noise
        for kf in self.kfs:
            kf['P'] += self.Q
            
        probs = self._get_soft_assignments(arm)
        
        # Instantiate new cluster if it is the most likely mode
        if np.argmax(probs) == len(self.kfs):
            self._add_new_kf()
            probs = self._get_soft_assignments(arm)
            self.c[arm] = len(self.kfs) - 1
            
        # Measurement Update across all KFs
        for k in range(len(self.kfs)):
            w_k = probs[k]
            if w_k < 1e-3: continue
                
            kf = self.kfs[k]
            mu_prev = kf['mu'][arm]
            P_pred = kf['P'][arm]
            
            # The key logic for soft assignment:
            # We scale the observation noise R by 1/w_k. 
            # If responsibility is low, trust in the sample is low.
            effective_R = self.R / w_k 
            K_gain = P_pred / (P_pred + effective_R)
            
            kf['mu'][arm] = mu_prev + K_gain * (reward - mu_prev)
            kf['P'][arm] = (1.0 - K_gain) * P_pred
            kf['count'] += w_k

    @property
    def num_clusters(self):
        return len(self.kfs)
