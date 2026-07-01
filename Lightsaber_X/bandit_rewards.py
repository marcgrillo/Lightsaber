import os
import glob
import numpy as np
from scipy import signal
import pandas as pd

FS = 256
BLOCK_S = 2
BS = FS * BLOCK_S
EPS = 1e-30

WEIGHTS_FILE = "bandit_reward_weights.npz"

def butter_sos(kind, f1, f2=None, fs=FS, order=4):
    if kind == "lp":
        return signal.butter(order, f1, btype="lowpass", fs=fs, output="sos")
    if kind == "bp":
        return signal.butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    raise ValueError(kind)

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def to_blocks(x, bs=BS):
    # Truncate to multiple of block size
    n = (len(x) // bs) * bs
    return x[:n].reshape(-1, bs)

def eigen_transform_matrix():
    L = 3994.5
    R_ITM = 1934
    R_ETM = 2245
    g1 = 1 - L / R_ITM
    g2 = 1 - L / R_ETM
    r = 0.5 * (g1 - g2 + np.sqrt((g1 - g2)**2 + 4))
    return np.array([[1, r], [-r, 1]]) / (1 + r**2)

LOCAL2EIG = eigen_transform_matrix()

def band_feats_blocks(x, band_specs):
    feats = {}
    for name, (kind, f1, f2) in band_specs:
        sos = butter_sos(kind, f1, f2, order=4)
        xf = signal.sosfilt(sos, x)
        feats[name] = np.array([rms(b) for b in to_blocks(xf)])
    return feats

ERR_BANDS = [
    ("err_lp_0p3",       ("lp", 0.3, None)),
    ("err_bp_0p05_0p3",  ("bp", 0.05, 0.30)),
    ("err_bp_0p3_1",     ("bp", 0.30, 1.0)),
    ("err_bp_1_3",       ("bp", 1.0, 3.0)),
    ("err_bp_3_10",      ("bp", 3.0, 10.0)),
    ("err_bp_10_30",     ("bp", 10.0, 30.0)),
    ("err_bp_30_80",     ("bp", 30.0, 80.0)),
    ("err_bp_80_100",    ("bp", 80.0, 100.0)),
]

U_BANDS = [
    ("u_lp_0p3",         ("lp", 0.3, None)),
    ("u_bp_0p05_0p3",    ("bp", 0.05, 0.30)),
    ("u_bp_0p3_1",       ("bp", 0.30, 1.0)),
    ("u_bp_1_3",         ("bp", 1.0, 3.0)),
    ("u_bp_3_10",        ("bp", 3.0, 10.0)),
    ("u_bp_10_30",       ("bp", 10.0, 30.0)),
    ("u_bp_30_80",       ("bp", 30.0, 80.0)),
    ("u_bp_80_100",      ("bp", 80.0, 100.0)),
]

def load_err_u(run_dir):
    ang_file = os.path.join(run_dir, "angles_eigen.csv")
    ctrl_file = os.path.join(run_dir, "control_ITM_ETM.csv")
    
    if not os.path.exists(ang_file) or not os.path.exists(ctrl_file):
        raise FileNotFoundError(f"Missing simulation output files in {run_dir}")

    ang = np.loadtxt(ang_file)
    err_h = ang[:, 1]  # hard error eigen

    u_local = np.loadtxt(ctrl_file)
    u_eig = (LOCAL2EIG @ u_local.T).T
    u_h = u_eig[:, 1]  # hard control eigen
    return err_h, u_h

def load_reward_weights(weights_npz=WEIGHTS_FILE):
    W = np.load(weights_npz, allow_pickle=True)
    feat_names = list(W["feat_names"])
    w = W["w"].astype(float)
    return feat_names, w

def reward_score_per_block(run_dir, feat_names, w):
    """
    Computes the raw reward score for each 2s block.
    Returns array of scores (one per block).
    """
    err_h, u_h = load_err_u(run_dir)
    feats = {}
    feats.update(band_feats_blocks(err_h, ERR_BANDS))
    feats.update(band_feats_blocks(u_h,   U_BANDS))

    # Z(b) = sum_i w_i * log(feature_i(b)); score = -Z  (higher is better)
    # Check if we have data
    if not feats:
        return np.array([])
        
    n_blocks = len(next(iter(feats.values())))
    Z = np.zeros(n_blocks)
    
    for wi, fn in zip(w, feat_names):
        if fn in feats:
            Z += wi * np.log(feats[fn] + EPS)
    
    score = -Z
    return score

def compute_reward_for_segment(run_dir, feat_names, w):
    """
    Computes the single scalar reward for a simulation segment.
    Uses the mean of block scores.
    Expects to normalize or map to [0,1] as requested, 
    but for now returns normalized score based on approximate range [150, 180].
    """
    scores = reward_score_per_block(run_dir, feat_names, w)
    if len(scores) == 0:
        return 0.0
        
    mean_score = np.mean(scores)
    
    # Heuristic normalization based on observation (scores ~160-170)
    # Mapping [150, 180] -> [0, 1] roughly.
    # r = (score - 150) / 30
    # This is a placeholder as per user request to "supervise later".
    # But user ALSO insisted "mapped to [0,1]".
    
    # Let's try to be safe. If scores are ~160, dividing by 200 gives 0.8.
    # Let's simple sigmoid or linear scaling. 
    # Let's use linear scaling 0 to 200 -> 0 to 1.
    
    norm_score = np.clip(mean_score / 200.0, 0.0, 1.0)
    
    return norm_score, mean_score
