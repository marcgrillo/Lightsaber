import numpy as np
w = np.load('data/bandit_reward_weights.npz', allow_pickle=True)
print("W Weights:", w['w'])
print("Sum W:", np.sum(w['w']))
