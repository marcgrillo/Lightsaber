import os
import numpy as np
import pandas as pd
from scipy import signal

class ColoredNoiseFIR:
    """
    Streamable colored-noise generator: White Gaussian -> FIR -> Colored.
    Allows for real-time updates of the target Power Spectral Density (PSD)
    without resetting the internal filter state.
    """
    def __init__(self, fs, asd_ff, asd_vals, n_taps=8193, sigma=1.0, seed=None, window='hann'):
        self.fs = float(fs)
        self.sigma = float(sigma)
        self.rng = np.random.RandomState(seed=seed)
        self.window = window
        self.h = None
        self.zi = None
        self.update_asd(asd_ff, asd_vals, n_taps=n_taps)

    def _asd_white(self):
        # 1-sided ASD of discrete white noise: sqrt(2 * sigma^2 / fs)
        return np.sqrt(2.0 * (self.sigma ** 2) / self.fs)

    def update_asd(self, asd_ff, asd_vals, n_taps=None):
        if n_taps is None:
            n_taps = len(self.h) if self.h is not None else 8193

        # Design FIR filter taps from the desired ASD
        fmax = 0.5 * self.fs
        f_grid = np.unique(np.clip(np.concatenate(([0.0], asd_ff, [fmax])), 0.0, fmax))
        asd_grid = np.interp(f_grid, asd_ff, asd_vals, left=0.0, right=0.0)
        
        # Desired magnitude response relative to white noise floor
        Hmag = asd_grid / self._asd_white()
        f_norm = f_grid / fmax
        
        self.h = signal.firwin2(numtaps=int(n_taps), freq=f_norm, gain=Hmag, window=self.window)
        
        # Maintain state continuity when taps change
        L = len(self.h)
        if self.zi is None:
            self.zi = np.zeros(L - 1, dtype=np.float64)
        elif len(self.zi) < L - 1:
            self.zi = np.pad(self.zi, (L - 1 - len(self.zi), 0))
        elif len(self.zi) > L - 1:
            self.zi = self.zi[-(L - 1):]

    def sample(self, n):
        x = self.rng.normal(0, self.sigma, int(n))
        y, self.zi = signal.lfilter(self.h, [1.0], x, zi=self.zi)
        return y


class StreamableNoise:
    """Wrapper that maintains state for a specific noise channel."""
    def __init__(self, name, fs, initial_ff, initial_asd, seed=None):
        self.name = name
        self.fs = fs
        self.gen = ColoredNoiseFIR(fs, initial_ff, initial_asd, seed=seed)
        
    def step(self, n_samples, target_ff, target_asd):
        self.gen.update_asd(target_ff, target_asd)
        return self.gen.sample(n_samples)


class SinusoidalPoissonEnv:
    """
    Advanced environment module for generating complex, non-stationary environmental disturbances.
    Models Seismic events and overlapping Optical Sensor and Electro-Magnetic (OSEM) noise vectors 
    using a sinusoidal underlying probability envelope augmented by Poisson-style randomized spike events.
    """
    def __init__(self, total_duration, mu=30.0, std_dev=3600.0, trans_mean=60.0, exponent=0.33, 
                 r1_rate=3.0, r1_attack=60.0, r1_hold=1200.0, r1_decay=60.0, seed=42):
        self.total_duration = total_duration
        self.rng = np.random.RandomState(seed)
        self.period = 86400.0 # 24 Hours
        
        # 1. Base Sinusoidal Wave Generation (Drift between R0 and R2)
        num_days = int(np.ceil(total_duration / self.period))
        drift_starts = []
        for day in range(num_days):
            k = self.rng.poisson(mu)
            centers = [day*self.period + self.period/4.0, day*self.period + 3*self.period/4.0]
            for _ in range(k):
                t_cand = self.rng.normal(self.rng.choice(centers), std_dev)
                drift_starts.append(np.clip(t_cand, 0, total_duration))
        drift_starts.sort()
        
        def true_sine(t):
            raw = np.cos(2 * np.pi * t / self.period)
            mod = np.sign(raw) * np.abs(raw)**exponent
            return 0.5 + 0.5 * mod

        self.drift_events = []
        curr_v = true_sine(0)
        last_t = 0.0
        for ts in drift_starts:
            ts = max(ts, last_t)
            dur = self.rng.exponential(trans_mean)
            te = ts + dur
            target_v = true_sine(te)
            self.drift_events.append((ts, te, curr_v, target_v))
            curr_v, last_t = target_v, te
        self.start_val = true_sine(0)

        # 2. Poisson Events (Regime 1 Spikes)
        lambda_rate = r1_rate / 86400.0 # events/sec
        event_times = np.cumsum(self.rng.exponential(1.0/lambda_rate, int(lambda_rate*total_duration*2)))
        self.events = event_times[event_times < total_duration]
        self.intervals = []
        for ts in self.events:
            ta, th, td = self.rng.exponential(r1_attack), self.rng.exponential(r1_hold), self.rng.exponential(r1_decay)
            self.intervals.append({'t0': ts, 't1': ts+ta, 't2': ts+ta+th, 't3': ts+ta+th+td})

    def get_weights(self, t):
        """
        Determines the current active noise regime (R0: Nominal, R1: High Microseism, R2: Extreme Microseism + High OSEM).
        Evaluates the current threshold probabilities. Hard-switches (>0.75 or <0.25) explicitly 
        drive the environment into steady-state pure regimes before drifting again.
        """
        # Calculate base sinusoid state
        val = self.start_val
        for ts, te, vs, ve in self.drift_events:
            if ts <= t <= te:
                val = vs + (t - ts) / (te - ts + 1e-9) * (ve - vs)
                break
            elif t > te:
                val = ve
        
        # Restore hard-switching thresholds for research parity
        if val > 0.75: val = 1.0
        elif val < 0.25: val = 0.0
        
        w0_base, w2_base = val, 1.0 - val
        w1 = 0.0
        
        # Overlay Poisson spike (Regime 1)
        for evt in self.intervals:
            if evt['t0'] <= t <= evt['t3']:
                if t < evt['t1']: w1 = (t - evt['t0']) / (evt['t1'] - evt['t0'] + 1e-9)
                elif t <= evt['t2']: w1 = 1.0
                else: w1 = 1.0 - (t - evt['t2']) / (evt['t3'] - evt['t2'] + 1e-9)
                break
        
        # Normalize: R1 overrides R0/R2 balance
        w1 = np.clip(w1, 0, 1)
        w0 = w0_base * (1.0 - w1)
        w2 = w2_base * (1.0 - w1)
        return (w0, w1, w2)
    def get_noise_streams(self, data_path, fs):
        """
        Generates the full non-stationary noise streams for the physical mirror suspensions (ITM & ETM).
        Constructs a total of 6 distinct independent colored noise vectors:
        - 2x Seismic base disturbances (ITM/ETM)
        - 2x Local OSEM read noise (ITM/ETM)
        - 2x Pitch-coupled OSEM noise (ITM/ETM)
        
        The arrays are structurally transformed via explicit physical Transfer Functions simulating the top-mass 
        coupling down to the test mass, establishing rigorously authentic suspension mechanics.
        """
        total_samples = int(self.total_duration * fs)
        
        # 1. Load the 3 empirical foundational ASDs 
        r0_asd = np.genfromtxt(os.path.join(data_path, "noise", "ASD_R0_nominal.csv"))
        r1_asd = np.genfromtxt(os.path.join(data_path, "noise", "ASD_R1_high_micro.csv"))
        r2_asd = np.genfromtxt(os.path.join(data_path, "noise", "ASD_R2_high_micro2.csv"))
        
        # 1.5 Load and Parse the Suspension Transfer Functions (Research Parity)
        def load_tf(filename):
            ff, mag = [], []
            p = os.path.join(data_path, "transfers", filename)
            with open(p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            freq = float(parts[0])
                            val = complex(parts[1].replace('i', 'j'))
                            ff.append(freq)
                            mag.append(np.abs(val))
                        except: pass
            return np.array(ff), np.array(mag)

        tf_l_ff, tf_l_mag = load_tf("tf_topL_2_tstP.txt")
        tf_nl_ff, tf_nl_mag = load_tf("tf_topNL_2_tstP.txt")
        tf_np_ff, tf_np_mag = load_tf("tf_topNP_2_tstP.txt")
        
        # 1.6 Load Common OSEM Noises
        def load_osem(filename):
            p = os.path.join(data_path, "noise", filename)
            d = np.genfromtxt(p)
            return d[:, 0], d[:, 1]

        osem_l_ff, osem_l_asd = load_osem("n_osem_L.txt")
        osem_p_ff, osem_p_asd = load_osem("n_osem_P.txt")

        # Interpolate TF magnitudes and OSEM ASDs to match mirror ASD frequency grid
        target_ff = r0_asd[:, 0]
        tf_l_interp = np.interp(target_ff, tf_l_ff, tf_l_mag, left=0.0, right=0.0)
        tf_nl_interp = np.interp(target_ff, tf_nl_ff, tf_nl_mag, left=0.0, right=0.0)
        tf_np_interp = np.interp(target_ff, tf_np_ff, tf_np_mag, left=0.0, right=0.0)
        
        osem_l_interp = np.interp(target_ff, osem_l_ff, osem_l_asd, left=0.0, right=0.0)
        osem_p_interp = np.interp(target_ff, osem_p_ff, osem_p_asd, left=0.0, right=0.0)
        
        # 2. Setup Noise Generators (6 total as per research)
        # 0: Seismic ITM, 1: OSEM L ITM, 2: OSEM P ITM
        # 3: Seismic ETM, 4: OSEM L ETM, 5: OSEM P ETM
        gens = [StreamableNoise(f"G{i}", fs, target_ff, r0_asd[:, 1], seed=self.rng.randint(10000)) for i in range(6)]
        
        itm_noise = np.zeros(total_samples)
        etm_noise = np.zeros(total_samples)
        power_scale_tt = np.zeros(total_samples)
        sens_scale_tt = np.zeros(total_samples)
        
        # Authentic 6th_clean Scale definitions
        power_scales = [1.0, 0.95, 0.9]
        sens_scales = [120.0, 12.0, 0.25]
        seis_scales = [1.0, 6.0, 20.0]
        
        # 3. Generate noise in 1-second chunks to handle spectral transitions
        print(f"Generating {self.total_duration}s of authentic non-stationary noise...")
        for t_sec in range(int(self.total_duration)):
            w0, w1, w2 = self.get_weights(float(t_sec))
            
            p_scale = w0 * power_scales[0] + w1 * power_scales[1] + w2 * power_scales[2]
            s_scale = w0 * sens_scales[0] + w1 * sens_scales[1] + w2 * sens_scales[2]
            seis_scale = w0 * seis_scales[0] + w1 * seis_scales[1] + w2 * seis_scales[2]
            
            # Composite ASD Terms: Seismic * TF_L + OSEM_L * TF_NL + OSEM_P * TF_NP
            asd_sei = (w0 * r0_asd[:, 1] + w1 * r1_asd[:, 1] + w2 * r2_asd[:, 1]) * seis_scale * tf_l_interp
            asd_l_term = osem_l_interp * tf_nl_interp # osem_scale is 1.0
            asd_p_term = osem_p_interp * tf_np_interp # osem_scale is 1.0
            
            start_idx = t_sec * fs
            end_idx = (t_sec + 1) * fs
            if end_idx > total_samples: break
            
            # Synthesize and Sum components
            itm_noise[start_idx:end_idx] = (gens[0].step(fs, target_ff, asd_sei) + 
                                            gens[1].step(fs, target_ff, asd_l_term) + 
                                            gens[2].step(fs, target_ff, asd_p_term))
            
            etm_noise[start_idx:end_idx] = (gens[3].step(fs, target_ff, asd_sei) + 
                                            gens[4].step(fs, target_ff, asd_l_term) + 
                                            gens[5].step(fs, target_ff, asd_p_term))
            power_scale_tt[start_idx:end_idx] = p_scale
            sens_scale_tt[start_idx:end_idx] = s_scale
            
            if t_sec % 3600 == 0 and t_sec > 0:
                print(f"   -> {t_sec//3600} hours complete...")
        
        return itm_noise, etm_noise, power_scale_tt, sens_scale_tt
