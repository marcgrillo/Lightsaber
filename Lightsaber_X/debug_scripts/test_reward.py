import numpy as np
import os
from reward_utils import ResearchRewardCalculator

# Dummy data (microradians, non-zero)
err = np.random.normal(0, 1.0, 300 * 256)
u = np.random.normal(0, 10.0, 300 * 256)

weights_path = r"e:\workwithtomi_06_04\Lightsaber\Lightsaber_X\data\bandit_reward_weights_rel2base.npz"
calc = ResearchRewardCalculator(weights_path)

print(f"Num Weights: {len(calc.w_weights)}")
print(f"Num Names: {len(calc.feat_names)}")

reward, raw = calc.calculate_reward(err, u)
print(f"RESULT -> Reward: {reward}, Raw: {raw}")
