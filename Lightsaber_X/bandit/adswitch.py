import numpy as np
import math
from bandit_algorithm import BanditAlgorithm

class AdSwitch(BanditAlgorithm):
    """
    Implementation of the AdSwitch algorithm for non-stationary stochastic bandits.
    Reference: Auer, P., Gajane, P., & Ortner, R. (2019). Adaptively tracking the best bandit arm with an unknown number of distribution changes.
    """

    def __init__(self, num_arms, horizon, C1=4.0, log_file=None):
        super().__init__(num_arms)
        self.T = horizon
        self.C1 = C1
        self.log_T = math.log(self.T, 10) if self.T > 1 else 1.0
        self.log_file = log_file
        
        # Clear log file if exists
        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write(f"AdSwitch Log Started. T={self.T}, C1={self.C1}\n")
                
        self.reset()
        
    def log(self, type_str, msg):
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"[t={self.t}] [{type_str}] {msg}\n")

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
        self.log("INFO", "Reset complete.")

    def _get_stats(self, arm, start_time, end_time):
        """
        Computes the empirical mean and count for an arm in a given time interval [start_time, end_time].
        Note: time indices are 1-based, inclusive.
        """
        if start_time > end_time or start_time < 1:
            return 0.0, 0

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
        Selects an arm according to the AdSwitch policy.
        """
        self.t += 1 
        
        # Add visual separation in log
        if self.log_file and self.t > 1:
            with open(self.log_file, "a") as f:
                f.write("\n\n")
                
        self.log("SELECT", f"--- Step {self.t} ---")
        
        # Log play counts
        counts = [0] * self.num_arms
        for (a, _) in self.history:
            counts[a] += 1
        self.log("COUNTS", f"Arm Counts: {counts}")
        
        # Reset diagnostic probabilities
        for arm in range(self.num_arms):
            self.current_sampling_probs[arm] = 0.0

        # 1. Add checks for bad arms (sampling obligations)
        self.log("BAD_ARM_CHECK", f"Checking bad arms: {self.bad_arms}")
        for arm in self.bad_arms:
            delta_tilde = self.eviction_stats[arm][1]
            if delta_tilde <= 0: continue 

            i = 1
            max_prob = 0.0
            self.log("BAD_ARM_CALC", f"Arm {arm}: delta_tilde={delta_tilde:.2e}")
            while True:
                epsilon = 2.0**(-i)
                if epsilon < delta_tilde / 16.0:
                    self.log("BAD_ARM_CALC", f"  Arm {arm}: Breaking w/ i={i} (eps={epsilon:.2e} < delta/16)")
                    break
                
                current_episode_idx = self.l + 1
                prob = epsilon * math.sqrt(current_episode_idx / (self.num_arms * self.T * self.log_T))
                
                max_prob = max(max_prob, prob)
                
                # self.log("BAD_ARM_CALC", f"  Arm {arm} i={i}: eps={epsilon:.2e}, prob={prob:.2e}")

                if np.random.random() < prob:
                    n_samples = math.ceil((2**(2*i + 1)) * self.log_T)
                    self.sampling_obligations[arm].add((epsilon, n_samples, self.t))
                    self.log("OBLIGATION", f"  Arm {arm}: Added obligation i={i}, n={n_samples}, Prob={prob}, Epsilon={epsilon}")
                
                i += 1
            self.current_sampling_probs[arm] = max_prob

        # 2. Determine Eligible Arms
        eligible_arms = []
        for arm in range(self.num_arms):
            is_good = arm in self.good_arms
            has_obligation = len(self.sampling_obligations[arm]) > 0
            if is_good or has_obligation:
                eligible_arms.append(arm)
        
        self.log("ELIGIBLE", f"Eligible arms: {eligible_arms}")

        if not eligible_arms: 
             # Fallback if empty (shouldnt happen usually)
             self.log("WARNING", "No eligible arms! Picking random.")
             return np.random.randint(0, self.num_arms)

        # 3. Select Least Recently Used Eligible Arm
        last_played = {arm: -1 for arm in eligible_arms}
        for t_idx, (arm, _) in enumerate(self.history):
            if arm in eligible_arms:
                last_played[arm] = t_idx + 1 
        
        # Break ties by arm index to be deterministic
        selected_arm = min(eligible_arms, key=lambda a: (last_played[a], a))
        self.log("SELECT", f"Selected Arm {selected_arm} (Last Played {last_played[selected_arm]})")
        return selected_arm

    def update(self, arm, reward):
        """
        Updates state with reward, performs checks, and potentially restarts episode.
        Strict Implementation of Algorithm 1.
        """
        # Store observation
        self.history.append((arm, reward))
        self.log("UPDATE", f"Arm {arm} received Reward {reward:.4f}. History len={len(self.history)}")
        
        restart_needed = False

        # --- Condition 3: Good Arm Change (Strict Vectorized) ---
        self.log("COND3", "Starting Strict Condition 3 Check...")
        
        for a in self.good_arms:
            # Get rewards in current episode
            raw_data = [r for (hist_arm, r) in self.history[self.t_l-1:] if hist_arm == a]
            L = len(raw_data)
            if L < 2: 
                # self.log("COND3", f"Arm {a}: Not enough data (L={L})")
                continue 
            
            # Vectorized Check
            r_vec = np.array(raw_data)
            S = np.concatenate(([0.0], np.cumsum(r_vec))) # Size L+1
            
            j_grid, i_grid = np.meshgrid(np.arange(1, L+1), np.arange(0, L))
            mask = j_grid > i_grid
            
            # Counts N[i, j]
            N_mat = j_grid - i_grid
            
            # Means M[i, j]
            Sums_mat = S[j_grid] - S[i_grid]
            
            valid_N = N_mat[mask]
            valid_Means = Sums_mat[mask] / valid_N
            
            # Confidence terms
            conf_numer = 2.0 * self.log_T
            valid_Conf = np.sqrt(conf_numer / valid_N)
            
            valid_LB = valid_Means - valid_Conf
            valid_UB = valid_Means + valid_Conf
            
            # 1. Aggregates for "All Intervals" (Inner)
            Max_LB_Inner = np.max(valid_LB)
            Min_UB_Inner = np.min(valid_UB)
            
            # 2. Aggregates for "Suffixes" (Ending at t, i.e., index L)
            suffix_mask_flat = (j_grid[mask] == L)
            
            Suffix_LB = valid_LB[suffix_mask_flat]
            Suffix_UB = valid_UB[suffix_mask_flat]
            
            Max_LB_Suffix = np.max(Suffix_LB)
            Min_UB_Suffix = np.min(Suffix_UB)
            
            # --- Debugging Stats extraction ---
            Suffix_Means = valid_Means[suffix_mask_flat]
            Suffix_Conf = valid_Conf[suffix_mask_flat]

            # Reconstruct indices
            i_flat = i_grid[mask]
            j_flat = j_grid[mask]
            
            # Suffix start times (relative to raw_data start)
            # Suffixes are where j_grid == L
            valid_suffix_indices = np.where(suffix_mask_flat)[0]
            suffix_starts = i_grid[mask][suffix_mask_flat]

            # Log ALL Intervals (Verbose!) - COMMENTED OUT as per user request
            # if self.log_file:
            #     self.log("COND3_ALL_INTER", f"Arm {a}: Listing all {len(valid_Means)} internal intervals:")
            #     for k in range(len(valid_Means)):
            #          s1 = self.t_l + i_flat[k]
            #          s2 = self.t_l + j_flat[k] - 1
            #          self.log("COND3_INTER", f"  [{s1}, {s2}] mu={valid_Means[k]:.4f} conf={valid_Conf[k]:.4f} n={valid_N[k]}")
            
            # if self.log_file:
            #      self.log("COND3_ALL_SUFFIX", f"Arm {a}: Listing all {len(Suffix_Means)} suffixes:")
            #      for k in range(len(Suffix_Means)):
            #           s_start = self.t_l + suffix_starts[k]
            #           t_end = self.t
            #           self.log("COND3_SUFFIX", f"  [{s_start}, {t_end}] mu={Suffix_Means[k]:.4f} conf={Suffix_Conf[k]:.4f}")

            # Case 1: Inner LB > Suffix UB
            idx_inner_lb = np.argmax(valid_LB)
            idx_suffix_ub = np.argmin(Suffix_UB)
            
            margin_1 = valid_LB[idx_inner_lb] - Suffix_UB[idx_suffix_ub]
            
            # Case 2: Suffix LB > Inner UB
            idx_suffix_lb = np.argmax(Suffix_LB)
            idx_inner_ub = np.argmin(valid_UB)
            
            margin_2 = Suffix_LB[idx_suffix_lb] - valid_UB[idx_inner_ub]
            
            if margin_1 > margin_2:
                mu1 = valid_Means[idx_inner_lb]
                mu2 = Suffix_Means[idx_suffix_ub]
                c1 = valid_Conf[idx_inner_lb]
                c2 = Suffix_Conf[idx_suffix_ub]
                best_margin = margin_1
                pair_type = "InnerLB > SuffixUB"
                
                # Indices
                # Inner Interval [s1, s2]
                s1_rel = i_flat[idx_inner_lb]
                s2_rel = j_flat[idx_inner_lb] - 1 # Inclusive end
                # Suffix Interval [s, t]
                s_rel = suffix_starts[idx_suffix_ub]
                
            else:
                mu1 = Suffix_Means[idx_suffix_lb]
                mu2 = valid_Means[idx_inner_ub]
                c1 = Suffix_Conf[idx_suffix_lb]
                c2 = valid_Conf[idx_inner_ub]
                best_margin = margin_2
                pair_type = "SuffixLB > InnerUB"
                
                # Indices
                # Inner Interval [s, t] (Suffix is the 'mu1' here)
                s1_rel = suffix_starts[idx_suffix_lb]
                s2_rel = L - 1
                # Other Interval (Inner)
                s_rel = i_flat[idx_inner_ub]
                # Wait, "s" usually refers to the suffix start.
                # Here we are comparing Suffix vs Inner.
                # Let's verify labels. 
                # If margin2 wins, SuffixLB - InnerUB is maximized.
                # Suffix (mu1) is [s_rel, L-1]. Inner (mu2) is [s1_inner, s2_inner].
                # User asks for s1, s2, s.
                # s1, s2 distinct from s.
                
                # Let's map correctly:
                # Pair is (Interval_1, Interval_2).
                # Interval_1 = Suffix [s_rel, L-1]. Interval_2 = Inner [s_inner, e_inner].
                # We can report both.
                s_suffix_rel = suffix_starts[idx_suffix_lb]
                s_inner_start = i_flat[idx_inner_ub]
                s_inner_end = j_flat[idx_inner_ub] - 1
                
                # Normalize names for output
                s1_rel = s_inner_start
                s2_rel = s_inner_end
                s_rel = s_suffix_rel

            # Convert to Absolute Times
            # self.history[self.t_l-1:] starts at t_l.
            # Index 0 is t_l.
            abs_s1 = self.t_l + s1_rel
            abs_s2 = self.t_l + s2_rel
            abs_s = self.t_l + s_rel

            # --- New Debugging: Max Delta Mu Pair ---
            # Also find the pair with the largest absolute mean difference, regardless of confidence.
            # Max(Inner) - Min(Suffix) OR Max(Suffix) - Min(Inner)
            idx_inner_max = np.argmax(valid_Means)
            idx_inner_min = np.argmin(valid_Means)
            idx_suffix_max = np.argmax(Suffix_Means)
            idx_suffix_min = np.argmin(Suffix_Means)
            
            delta_1 = valid_Means[idx_inner_max] - Suffix_Means[idx_suffix_min]
            delta_2 = Suffix_Means[idx_suffix_max] - valid_Means[idx_inner_min]
            
            if delta_1 > delta_2:
                max_delta = delta_1
                md_mu1 = valid_Means[idx_inner_max]
                md_mu2 = Suffix_Means[idx_suffix_min]
                md_conf1 = valid_Conf[idx_inner_max]
                md_conf2 = Suffix_Conf[idx_suffix_min]
                
                md_s1_rel = i_flat[idx_inner_max]
                md_s2_rel = j_flat[idx_inner_max] - 1
                md_s_rel = suffix_starts[idx_suffix_min]
                md_type = "InnerHigh - SuffixLow"
            else:
                max_delta = delta_2
                md_mu1 = Suffix_Means[idx_suffix_max]
                md_mu2 = valid_Means[idx_inner_min]
                md_conf1 = Suffix_Conf[idx_suffix_max]
                md_conf2 = valid_Conf[idx_inner_min]
                
                md_s1_rel = suffix_starts[idx_suffix_max]
                md_s2_rel = L - 1
                md_s_rel = i_flat[idx_inner_min] # Inner Interval Start
                
                # Careful mapping if type is Suffix - Inner
                # Pair is (Interval1, Interval2). |mu1 - mu2|
                md_type = "SuffixHigh - InnerLow"
                
                # Normalize names for output (always show [s1, s2] as Inner, [s, t] as Suffix)
                # But here we just report the raw pair causing the diff
                # Inner Index
                s_inner_start = i_flat[idx_inner_min]
                s_inner_end = j_flat[idx_inner_min] - 1
                s_suffix_start = suffix_starts[idx_suffix_max]
                
                md_s1_rel = s_inner_start
                md_s2_rel = s_inner_end
                md_s_rel = s_suffix_start

            abs_md_s1 = self.t_l + md_s1_rel
            abs_md_s2 = self.t_l + md_s2_rel
            abs_md_s = self.t_l + md_s_rel

            # ----------------------------------

            # print(f"Cond3 Check Arm {a}: MaxMargin={best_margin:.4f} ({pair_type}) | |mu1-mu2|={abs(mu1-mu2):.4f}, c1+c2={c1+c2:.4f}")
            self.log("COND3", f"Arm {a}: MaxMargin={best_margin:.4f} | |d_mu|={abs(mu1-mu2):.4f}, 2C={c1+c2:.4f}")
            self.log("COND3_DETAIL", f"  MaxMargin Pair: [s1, s2]=[{abs_s1}, {abs_s2}], [s, t]=[{abs_s}, {self.t}]. (Type: {pair_type})")
            self.log("COND3_DETAIL", f"  MaxDelta  Pair: [s1, s2]=[{abs_md_s1}, {abs_md_s2}], [s, t]=[{abs_md_s}, {self.t}]. (Type: {md_type})")
            self.log("COND3_DETAIL", f"    Delta={max_delta:.4f} | Margin={max_delta - (md_conf1 + md_conf2):.4f} | 2C={md_conf1+md_conf2:.4f}")
            # ----------------------------------
            
            if (Max_LB_Inner > Min_UB_Suffix) or (Max_LB_Suffix > Min_UB_Inner):
                msg = f"Restart Cond 3 (Strict): Arm {a} violated. MaxLB_In={Max_LB_Inner:.4f}, MinUB_Suf={Min_UB_Suffix:.4f}"
                # print(msg)
                self.log("VIOLATION", msg)
                restart_needed = True
                break
                
        if restart_needed:
            self.log("RESTART", "Condition 3 Violation -> New Episode")
            self._start_new_episode()
            return
            
        # --- Condition 4: Bad Arm Change (Strict) ---
        self.log("COND4", "Starting Condition 4 (Bad Arm) Check")
        check_starts = range(self.t_l, self.t + 1)
        
        for a in list(self.bad_arms):
            mu_tilde = self.eviction_stats[a][0]
            delta_tilde = self.eviction_stats[a][1]
            
            # Vectorized Suffix Check for Bad Arm
            raw_data = [r for (hist_arm, r) in self.history[self.t_l-1:] if hist_arm == a]
            L = len(raw_data)
            if L == 0: continue
            
            r_vec = np.array(raw_data)
            S = np.concatenate(([0.0], np.cumsum(r_vec)))
            
            # Suffixes
            i_arr = np.arange(0, L)
            N_arr = L - i_arr
            Means_arr = (S[L] - S[i_arr]) / N_arr
            
            # Check condition: |mu - mu_tilde| > delta/4 + sqrt(2 log T / n)
            diffs = np.abs(Means_arr - mu_tilde)
            thresholds = (delta_tilde / 4.0) + np.sqrt((2.0 * self.log_T) / N_arr)
            
            # USER EDIT (Step 511): argmax(diffs) instead of diffs-thresholds
            idx = np.argmax(diffs)
            self.log("COND4", f"Arm {a} Closest: diff={diffs[idx]:.4f}, thresh={thresholds[idx]:.4f}, N_suff={len(diffs)}, s={idx}, n(s,t)={N_arr[idx]}")
            # print(f"Condition 4 (Bad arm Improves): Check Arm {a} Closest: diff={diffs[idx]}, threshold={thresholds[idx]}. s={idx}")
            
            if np.any(diffs > thresholds):
                 restart_needed = True
                 msg = f"RESTART EPISODE Condition 4 (Strict): Arm {a} violated."
                 # print(msg)
                 self.log("VIOLATION", msg)
                 break
        
        if restart_needed:
            self.log("RESTART", "Condition 4 Violation -> New Episode")
            self._start_new_episode()
            return

        # --- Update sampling obligations ---
        for a in self.bad_arms:
            active = set()
            for (eps, n_req, s_start) in self.sampling_obligations[a]:
                _, count = self._get_stats(a, s_start, self.t)
                if count < n_req:
                    active.add((eps, n_req, s_start))
            self.sampling_obligations[a] = active

        # --- Condition 1: Eviction (Strict) ---
        self.log("COND1", "Starting Condition 1 (Eviction) Check")
        arms_to_evict = []
        # Track max violation: arm -> (margin, gap, thresh, s)
        max_violation = {} 

        for s in check_starts:
            # Gather stats for this suffix
            stats_s = {}
            for a in self.good_arms:
                mu, n = self._get_stats(a, s, self.t)
                if n > 0:
                    stats_s[a] = (mu, n)
            
            if not stats_s: continue
            
            best_a = max(stats_s.keys(), key=lambda x: stats_s[x][0])
            mu_best = stats_s[best_a][0]
            
            for a in list(self.good_arms):
                if a in [x[0] for x in arms_to_evict]: continue
                if a not in stats_s: continue
                
                mu_a, n_a = stats_s[a]
                if n_a < 2: continue
                
                # Condition 1 Constant C1
                gap = mu_best - mu_a
                thresh = math.sqrt( (self.C1 * self.log_T) / (n_a - 1) )
                
                # Track max violation (closest to eviction)
                margin = gap - thresh
                if a not in max_violation or margin > max_violation[a]['margin']:
                    max_violation[a] = {'margin': margin, 'gap': gap, 'thresh': thresh, 's': s}

                if gap > thresh:
                    # print(f"Evicting Arm {a} at s={s}, gap={gap:.4f}, thresh={thresh:.4f}")
                    self.log("VIOLATION", f"Evicting Arm {a} at s={s}, gap={gap:.4f}, thresh={thresh:.4f}")
                    arms_to_evict.append((a, mu_a, gap))
        
        # Print summary of closest eviction checks
        for a, stats in max_violation.items():
            if a in [x[0] for x in arms_to_evict]: continue # Already printed eviction
            # print(f"Cond1 Check Arm {a} (Closest s={stats['s']}): gap={stats['gap']:.4f}, thresh={stats['thresh']:.4f}, margin={stats['margin']:.4f}")
            self.log("COND1", f"Arm {a} Closest s={stats['s']}: gap={stats['gap']:.4f}, thresh={stats['thresh']:.4f}, margin={stats['margin']:.4f}")

        # Apply evictions
        for (a, mu_evict, delta_evict) in arms_to_evict:
            if a in self.good_arms:
                self.good_arms.remove(a)
                self.bad_arms.add(a)
                self.eviction_stats[a] = (mu_evict, delta_evict)
                self.sampling_obligations[a] = set()
                self.log("EVICTION", f"Arm {a} evicted. good_arms={self.good_arms}, bad_arms={self.bad_arms}")

    def _start_new_episode(self):
        """Starts a new episode as per algorithm description."""
        self.l += 1
        self.t_l = self.t + 1
        self.good_arms = set(range(self.num_arms))
        self.bad_arms = set()
        self.eviction_stats = {}
        self.log("INFO", f"New Episode Started (l={self.l}). t_l set to {self.t_l}.")
