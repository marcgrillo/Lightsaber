import numpy as np
from scipy import signal

# Frequency bands (in Hz) for feature extraction
ERR_BANDS = [
    ("err_lp_0p3",       ("lowpass", 0.3, None)),
    ("err_bp_0p05_0p3",  ("bandpass", 0.05, 0.30)),
    ("err_bp_0p3_1",     ("bandpass", 0.30, 1.0)),
    ("err_bp_1_3",       ("bandpass", 1.0, 3.0)),
    ("err_bp_3_10",      ("bandpass", 3.0, 10.0)),
    ("err_bp_10_30",     ("bandpass", 10.0, 30.0)),
    ("err_bp_30_80",     ("bandpass", 30.0, 80.0)),
    ("err_bp_80_100",    ("bandpass", 80.0, 100.0)),
]

U_BANDS = [
    ("u_lp_0p3",         ("lowpass", 0.3, None)),
    ("u_bp_0p05_0p3",    ("bandpass", 0.05, 0.30)),
    ("u_bp_0p3_1",       ("bandpass", 0.30, 1.0)),
    ("u_bp_1_3",         ("bandpass", 1.0, 3.0)),
    ("u_bp_3_10",        ("bandpass", 3.0, 10.0)),
    ("u_bp_10_30",       ("bandpass", 10.0, 30.0)),
    ("u_bp_30_80",       ("bandpass", 30.0, 80.0)),
    ("u_bp_80_100",      ("bandpass", 80.0, 100.0)),
]

class AdvancedRewardCalculator:
    """
    Evaluates empirical control performance identically to external high-fidelity baseline mathematics.
    Computes a weighted Log-RMS composite penalty score based on multi-band decompositions of 
    the resulting errors (pitch traces) and control forces (actuation outputs).
    The algorithm utilizes 2-second sub-blocks to correctly weigh low-frequency drift versus transient high-frequency peaks.
    """
    def __init__(self, weights_path, fs=256, order=4):
        self.fs = fs
        self.order = order
        self.eps = 1e-30
        
        # Load empirical diagnostic component weights representing real-world objective tuning parameters.
        try:
            W = np.load(weights_path, allow_pickle=True)
            self.feat_names = list(W["feat_names"])
            self.w_weights = W["w"].astype(float)
        except Exception as e:
            print(f"Warning: Could not load reward weights from {weights_path}: {e}")
            self.feat_names = []
            self.w_weights = np.array([])

        # Pre-compute SOS filters enabling real-time fast convolutions over the physical features.
        self.sos_filters = {}
        for name, spec in ERR_BANDS + U_BANDS:
            kind, f1, f2 = spec
            if kind == "lowpass":
                sos = signal.butter(order, f1, btype="lowpass", fs=fs, output="sos")
            else:
                sos = signal.butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
            self.sos_filters[name] = sos

    def calculate_reward(self, err_samples, u_samples):
        """
        Calculates the definitive Log-RMS reward scalar. 
        Uses 2-second sub-blocking to match the original 'average of logs' optimization structure.
        """
        if len(self.feat_names) == 0:
            return 0.0, 0.0

        # Physical Calibration: Maintained mathematically purely in direct 1.0 SI Radians.
        # Removing previous arbitrary sigmoid bounds returns the final composite `Raw` values explicitly 
        # around the established theoretical maximum of ~167.5 when perfectly tuned.
        err_scaled = err_samples * 1.0
        u_scaled = u_samples * 1.0
        
        # 2. Split into 2-second blocks (bs = 512 samples @ 256Hz)
        bs = int(self.fs * 2)
        n_blocks = len(err_scaled) // bs
        if n_blocks == 0:
            return 0.0, 0.0
            
        def to_blocks(x):
            return x[:n_blocks * bs].reshape(n_blocks, bs)
            
        feats = {}
        
        # 3. Calculate Average-Log-RMS for each band
        # Actuation features (u_h)
        for name, _ in U_BANDS:
            sos = self.sos_filters[name]
            xf = signal.sosfilt(sos, u_scaled)
            blocks = to_blocks(xf)
            # rms per block -> log(rms) per block -> mean of logs
            block_rms = np.sqrt(np.mean(np.square(blocks), axis=1))
            feats[name] = np.mean(np.log(block_rms + self.eps))
            
        # Error features (err_h)
        for name, _ in ERR_BANDS:
            sos = self.sos_filters[name]
            xf = signal.sosfilt(sos, err_scaled)
            blocks = to_blocks(xf)
            block_rms = np.sqrt(np.mean(np.square(blocks), axis=1))
            feats[name] = np.mean(np.log(block_rms + self.eps))

        # 4. Final weighted sum Z
        # Z = sum w_i * feat_i  (Note: feat_i is already log(rms) here)
        Z = 0.0
        for wi, fn in zip(self.w_weights, self.feat_names):
            if fn in feats:
                Z += wi * feats[fn]
        
        # Return raw score as both reward and diagnostic score
        raw_score = -Z
        return float(raw_score), float(raw_score)
