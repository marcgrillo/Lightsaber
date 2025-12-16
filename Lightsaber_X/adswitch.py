import numpy as np
import math
from bandit_algorithm import BanditAlgorithm

class AdSwitch(BanditAlgorithm):
    """
    Implementation of the AdSwitch algorithm for non-stationary stochastic bandits.
    Reference: Auer, P., Gajane, P., & Ortner, R. (2019). Adaptively tracking the best bandit arm with an unknown number of distribution changes.
    """

    def __init__(self, num_arms, horizon, C1=4.0):
        """
        Initialize the AdSwitch algorithm.

        Args:
            num_arms (int): Number of arms (N).
            horizon (int): Time horizon (T).
            C1 (float): Constant for the confidence bound (default 4.0, should be sufficiently large).
        """
        super().__init__(num_arms)
        self.T = horizon
        self.C1 = C1
        self.log_T = math.log(self.T) if self.T > 1 else 1.0
        self.reset()

    def reset(self):
        """Resets the algorithm to its initial state."""
        self.t = 0
        self.l = 0 # Episode counter
        self.t_l = 1 # Start time of current episode
        
        # Sets of GOOD and BAD arms
        self.good_arms = set(range(self.num_arms))
        self.bad_arms = set()

        # Sampling obligations for BAD arms: dict mapping arm_index -> set of tuples (epsilon, n, s)
        self.sampling_obligations = {arm: set() for arm in range(self.num_arms)}

        # History of plays: list of (arm, reward) tuples, 1-indexed for convenience with math
        self.history = [] 
        
        # Eviction statistics: stores (mu_tilde, delta_tilde) for evicted arms
        self.eviction_stats = {} 
        
        # Track sampling probabilities for diagnostics
        self.current_sampling_probs = {arm: 0.0 for arm in range(self.num_arms)} 
        
        # Track sampling probabilities for diagnostics
        self.current_sampling_probs = {arm: 0.0 for arm in range(self.num_arms)} 

    def _get_stats(self, arm, start_time, end_time):
        """
        Computes the empirical mean and count for an arm in a given time interval [start_time, end_time].
        Note: time indices are 1-based, inclusive.
        """
        # Bounds check
        if start_time > end_time or start_time < 1:
            return 0.0, 0

        # Filter history for the given interval
        # self.history is 0-indexed, so round t is at index t-1
        # Optimization: Calculating sum/count by iterating is O(t). 
        # For strict O(log T) per check, we would need cumulative sum arrays (prefix sums).
        # However, the geometric grid reduces the *number of checks*.
        # Implementing full prefix sums is better for performance but adds code complexity.
        # Let's stick to slicing for clarity, but acknowledge prefix sums would be the production approach.
        
        relevant_history = self.history[start_time-1 : end_time]
        
        rewards = [r for a, r in relevant_history if a == arm]
        count = len(rewards)
        mean = sum(rewards) / count if count > 0 else 0.0
        return mean, count

    def _confidence_term(self, count, C_val=2.0):
        """Calculates the confidence term sqrt(C * log(T) / n)."""
        if count == 0:
            return float('inf')
        return math.sqrt((C_val * self.log_T) / count)

    def select_arm(self):
        """
        Selects an arm according to the AdSwitch policy:
        Select the arm that has been selected least recently among the eligible arms.
        Eligible = GOOD arms U BAD arms with active sampling obligations.
        """
        self.t += 1 # Increment time step for the new round
        
        # Reset diagnostic probabilities
        for arm in range(self.num_arms):
            self.current_sampling_probs[arm] = 0.0

        # 1. Add checks for bad arms (sampling obligations)
        # For all a in BAD_t
        for arm in self.bad_arms:
            # Iterate through scales i >= 1
            # We need to check condition: 2^-i >= delta_tilde / 16
            delta_tilde = self.eviction_stats[arm][1]
            
            if delta_tilde <= 0: continue 

            # i starts at 1. Max i is when 2^-i < delta_tilde / 16
            i = 1
            while True:
                epsilon = 2.0**(-i)
                if epsilon < delta_tilde / 16.0:
                    break
                
                # Probability p_epsilon = epsilon * sqrt(l / (K * T * log T))
                current_episode_idx = self.l + 1
                prob = epsilon * math.sqrt(current_episode_idx / (self.num_arms * self.T * self.log_T))
                
                # Update diagnostic (store the max probability encountered for this arm)
                self.current_sampling_probs[arm] = max(self.current_sampling_probs[arm], prob)
                
                if np.random.random() < prob:
                    n_samples = math.ceil(2 * (2**(2*i + 1)) * self.log_T)
                    # Add obligation (epsilon, n, s=t)
                    self.sampling_obligations[arm].add((epsilon, n_samples, self.t))
                
                i += 1

        # 2. Determine Eligible Arms
        eligible_arms = []
        for arm in range(self.num_arms):
            is_good = arm in self.good_arms
            has_obligation = len(self.sampling_obligations[arm]) > 0
            if is_good or has_obligation:
                eligible_arms.append(arm)

        # 3. Select Least Recently Used Eligible Arm
        # We look backwards in history to find the last time each eligible arm was played.
        last_played = {arm: -1 for arm in eligible_arms}
        
        # Optimization: In production code, maintain a 'last_played' array to avoid O(t) scan.
        for t_idx, (arm, _) in enumerate(self.history):
            if arm in eligible_arms:
                last_played[arm] = t_idx + 1 
        
        # Find arm with min last_played time
        # If never played (among the eligible), last_played is -1, which is minimum.
        selected_arm = min(eligible_arms, key=lambda a: last_played[a])
        
        return selected_arm

    def update(self, arm, reward):
        """
        Updates state with reward, performs checks, and potentially restarts episode.
        Includes Geometric Grid Optimization for Condition 3.
        """
        # Store observation
        self.history.append((arm, reward))
        
        # Current time t is already incremented in select_arm
        
        restart_episode = False

        # --- Check for changes of GOOD arms (Condition 3) ---
        # Geometric Grid Optimization (Remark 3 in paper):
        # Instead of checking all intervals [s1, s2] and [s, t], we check intervals 
        # of dyadic lengths L_k = 2^k * log(T).
        # We check if |mu_[t-L_k+1, t] - mu_[t_l, t]| > threshold (Simplified check)
        # OR more strictly following paper: Check mu over intervals of specific lengths.
        
        # The paper suggests checking intervals of lengths proportional to 2^k * log T.
        # Let's check intervals [t - length + 1, t] against [t_l, t].
        # Valid lengths must fit within the current episode.
        
        episode_len = self.t - self.t_l + 1
        
        if episode_len > 0:
            # Calculate k_max such that 2^k * log T <= episode_len
            if self.log_T > 0:
                 # We only care about lengths > 0
                 # Start k such that length >= 1
                 min_k = int(math.floor(math.log2(1.0/self.log_T)))
                 # Max k such that length <= episode_len
                 max_k = int(math.floor(math.log2(episode_len / self.log_T)))
                 
                 ks_to_check = range(min_k, max_k + 1)
            else:
                 ks_to_check = []

            # Pre-calculate Long-term mean (Episode mean) for good arms
            # This serves as the reference "mu_[s, t]" in Condition 3 approx.
            # Note: Paper is general, checking arbitrary s1, s2. 
            # We approximate by checking "Suffix of Length X" vs "Whole Episode".
            
            episode_stats = {}
            for a in self.good_arms:
                 episode_stats[a] = self._get_stats(a, self.t_l, self.t)

            for k_val in ks_to_check:
                length = int(math.floor(2**k_val * self.log_T))
                if length <= 0: continue
                
                # Interval [s_short, t]
                s_short = self.t - length + 1
                
                # Only check if interval is valid and starts after t_l
                if s_short < self.t_l: continue

                for good_arm in list(self.good_arms):
                    mu_long, n_long = episode_stats[good_arm]
                    if n_long == 0: continue

                    mu_short, n_short = self._get_stats(good_arm, s_short, self.t)
                    if n_short == 0: continue
                    
                    # Condition 3 check
                    diff = abs(mu_long - mu_short)
                    threshold = self._confidence_term(n_long) + self._confidence_term(n_short)
                    
                    if diff > threshold:
                        restart_episode = True
                        break
                if restart_episode: break
        
        if restart_episode:
            self._start_new_episode()
            return

        # --- Check for changes of BAD arms (Condition 4) ---
        # We can also apply geometric grid here, but usually bad arms are checked less often.
        # For strict optimization, we apply it here too. We only check suffixes [s, t]
        # where t-s+1 is a dyadic length.
        
        for bad_arm in list(self.bad_arms):
            mu_tilde = self.eviction_stats[bad_arm][0]
            delta_tilde = self.eviction_stats[bad_arm][1]
            
            # Geometric grid for s: s = t - length + 1
            if self.log_T > 0:
                 # We check suffixes of dyadic lengths that fit in [t_l, t]
                 max_k = int(math.floor(math.log2(episode_len / self.log_T)))
                 # Also ensure we check at least one small window if episode is short
                 # Start loop similar to above
                 
                 # Simplified: Check a few key lengths (1, 2, 4... up to episode length)
                 # to ensure we catch changes quickly even without log T scaling
                 # Just using powers of 2 for simplicity in simulation if log T is large
                 
                 current_len = 1
                 while current_len <= episode_len:
                     s = self.t - current_len + 1
                     current_len *= 2
                     
                     mu_curr, n_curr = self._get_stats(bad_arm, s, self.t)
                     if n_curr == 0: continue
                     
                     diff = abs(mu_curr - mu_tilde)
                     threshold = (delta_tilde / 4.0) + self._confidence_term(n_curr)
                     
                     if diff > threshold:
                         restart_episode = True
                         break
                
            if restart_episode: break

        if restart_episode:
            self._start_new_episode()
            return

        # --- Update sampling obligations ---
        for bad_arm in self.bad_arms:
            active_obligations = set()
            for (eps, n_req, s_start) in self.sampling_obligations[bad_arm]:
                _, count = self._get_stats(bad_arm, s_start, self.t)
                if count < n_req:
                    active_obligations.add((eps, n_req, s_start))
            self.sampling_obligations[bad_arm] = active_obligations

        # --- Evict arms from GOOD set (Condition 1) ---
        # Optimization: Check max_mu only over the whole episode [t_l, t]
        # Doing this check over all sub-intervals is expensive. 
        # A common relaxation is to check [t_l, t] and perhaps dyadic suffixes.
        # We check [t_l, t] (Cumulative mean)
        
        # 1. Get stats for all good arms over [t_l, t]
        current_stats = {}
        max_mu = -float('inf')
        
        for a in self.good_arms:
            mu, n = self._get_stats(a, self.t_l, self.t)
            current_stats[a] = (mu, n)
            if n > 0:
                max_mu = max(max_mu, mu)
        
        if max_mu > -float('inf'):
            arms_to_evict = []
            for a in list(self.good_arms):
                if a in arms_to_evict: continue
                
                mu_a, n_a = current_stats[a]
                if n_a < 2: continue
                
                gap = max_mu - mu_a
                threshold = math.sqrt( (self.C1 * self.log_T) / (n_a - 1) )
                
                if gap > threshold:
                    arms_to_evict.append(a)
                    self.eviction_stats[a] = (mu_a, gap)
                    self.sampling_obligations[a] = set()

            for a in arms_to_evict:
                self.good_arms.remove(a)
                self.bad_arms.add(a)

    def _start_new_episode(self):
        """Starts a new episode as per algorithm description."""
        self.l += 1
        self.t_l = self.t + 1 
        self.good_arms = set(range(self.num_arms))
        self.bad_arms = set()