"""
Continuous (stateful) reward filtering -- the old_ref convention.

Each band-pass/low-pass filter keeps its state across calls, so the reward signal is filtered
as one continuous stream (no per-window reset / start-up transient). RMS is taken per 2s block
of the continuously-filtered signal; the window reward is the mean of the per-block raw scores.

Use one StreamReward per closed-loop run: reset() at the start, then score() once per window.
"""
import numpy as np
from scipy import signal
import bandit_rewards as br

BS = br.BS  # 512 samples = 2s at 256 Hz
LOCAL2EIG = br.LOCAL2EIG


class StreamReward:
    def __init__(self, feat_names, w):
        self.feat_names = list(feat_names)
        self.w = np.asarray(w, float)
        self.sos, self.zi = {}, {}
        for name, (kind, f1, f2) in (br.ERR_BANDS + br.U_BANDS):
            s = br.butter_sos(kind, f1, f2)
            self.sos[name] = s
            self.zi[name] = np.zeros((s.shape[0], 2))

    def reset(self):
        for n in self.zi:
            self.zi[n][:] = 0.0

    def _feats(self, x, bands):
        out = {}
        for name, _ in bands:
            xf, self.zi[name] = signal.sosfilt(self.sos[name], x, zi=self.zi[name])  # continuous
            n = (len(xf)//BS)*BS
            blocks = xf[:n].reshape(-1, BS)
            out[name] = np.sqrt(np.mean(blocks*blocks, axis=1))
        return out

    def score(self, pitch, ctl):
        """Returns per-2s-block raw scores for this window (filters continue from prior state)."""
        err_h = (LOCAL2EIG @ pitch.T).T[:, 1]
        u_h = (LOCAL2EIG @ ctl.T).T[:, 1]
        feats = {}
        feats.update(self._feats(err_h, br.ERR_BANDS))
        feats.update(self._feats(u_h, br.U_BANDS))
        nb = len(next(iter(feats.values())))
        Z = np.zeros(nb)
        for wi, fn in zip(self.w, self.feat_names):
            if fn in feats:
                Z += wi*np.log(feats[fn] + 1e-30)
        return -Z
