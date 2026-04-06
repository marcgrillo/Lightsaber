import numpy as np
import os

path = r"e:\workwithtomi_06_04\Lightsaber\Lightsaber_X\data\bandit_reward_weights.npz"
if not os.path.exists(path):
    print(f"ERROR: {path} not found")
else:
    w = np.load(path, allow_pickle=True)
    print("Keys:", list(w.keys()))
    if 'feat_names' in w:
        print("Feature Names:", list(w['feat_names']))
    else:
        print("ERROR: 'feat_names' not found in NPZ")
