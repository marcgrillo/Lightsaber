import numpy as np
import matplotlib.pyplot as plt
from adswitch import AdSwitch
from tqdm import tqdm

class NonStationaryEnvironment:
    """
    Simulates a multi-armed bandit environment where arm means change over time.
    """
    def __init__(self, num_arms, horizon, change_points=None):
        self.num_arms = num_arms
        self.T = horizon
        self.means = np.zeros((self.T, self.num_arms))
        
        # Default initialization: random means
        # We will override this in run_simulation for specific scenarios
        for i in range(self.num_arms):
             self.means[:, i] = np.random.uniform(0.2, 0.8) # base mean

    def get_reward(self, arm, t):
        """Returns a Bernoulli reward for the given arm at time t."""
        mean = self.means[t-1, arm]
        # Return 0 with probability (1 - mean), 1 with probability mean
        return np.random.binomial(1, mean)

    def get_optimal_reward(self, t):
        return np.max(self.means[t-1, :])

    def get_mean(self, arm, t):
        return self.means[t-1, arm]

def run_simulation():
    NUM_ARMS = 10
    HORIZON = 5000 # Increased horizon for 10 arms to see trends better
    
    # Create Environment
    env = NonStationaryEnvironment(NUM_ARMS, HORIZON)
    
    # Define a scenario with 10 arms and multiple change points
    # Base means - mostly mediocre
    for i in range(NUM_ARMS):
        env.means[:, i] = 0.4 

    # Phase 1: Arm 0 is best (Steps 0-1000)
    env.means[:1000, 0] = 0.9
    
    # Phase 2: Arm 3 becomes best (Steps 1000-2500)
    # Arm 0 drops
    env.means[1000:2500, 0] = 0.3
    env.means[1000:2500, 3] = 0.85
    
    # Phase 3: Arm 7 becomes best (Steps 2500-4000)
    # Arm 3 drops
    env.means[2500:4000, 3] = 0.35
    env.means[2500:4000, 7] = 0.9
    
    # Phase 4: Arm 9 becomes best (Steps 4000-5000)
    # Arm 7 drops
    env.means[4000:, 7] = 0.2
    env.means[4000:, 9] = 0.8

    # Initialize AdSwitch
    # Note: C1 should be "sufficiently large". 
    agent = AdSwitch(NUM_ARMS, HORIZON, C1=4.0)

    rewards = []
    optimal_rewards = []
    regrets = []
    
    # To track regret per arm
    arm_regrets = np.zeros((HORIZON, NUM_ARMS))
    
    print(f"Starting simulation with {NUM_ARMS} arms over {HORIZON} steps...")
    
    for t in tqdm( range(HORIZON) ):
        # 1. Agent selects arm
        arm = agent.select_arm()
        
        # 2. Environment generates reward
        reward = env.get_reward(arm, t+1)
        
        # 3. Agent updates
        agent.update(arm, reward)
        
        # Stats
        rewards.append(reward)
        opt_mean = env.get_optimal_reward(t+1)
        curr_mean = env.get_mean(arm, t+1)
        
        # Instantaneous regret (based on means, not realized reward, for cleaner plots)
        inst_regret = opt_mean - curr_mean
        regrets.append(inst_regret)
        
        # Record regret for the specific arm played
        arm_regrets[t, arm] = inst_regret

    cumulative_regret = np.cumsum(regrets)
    cumulative_arm_regrets = np.cumsum(arm_regrets, axis=0)
    
    print(f"Total Regret: {cumulative_regret[-1]}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Total Cumulative Regret
    ax1.plot(cumulative_regret, label='Total AdSwitch Regret', color='black', linewidth=2)
    
    # Mark change points
    change_points = [1000, 2500, 4000]
    for cp in change_points:
        ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5, label=f'Change at {cp}')
        ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        
    ax1.set_ylabel('Cumulative Dynamic Regret')
    ax1.set_title(f'AdSwitch Total Performance: {NUM_ARMS} Arms')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Cumulative Regret per Arm
    # We only plot arms that have significant regret to avoid clutter
    for i in range(NUM_ARMS):
        # Check if arm contributed > 1% of total regret
        if cumulative_arm_regrets[-1, i] > 0.01 * cumulative_regret[-1]:
             ax2.plot(cumulative_arm_regrets[:, i], label=f'Arm {i}')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret (Per Arm)')
    ax2.set_title('Regret Contribution by Arm')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('adswitch_detailed_regret.png')
    print("Detailed plot saved to adswitch_detailed_regret.png")

if __name__ == "__main__":
    run_simulation()