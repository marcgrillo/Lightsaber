"""
Disk-backed, chunked noise generation for the LONG online controller-selection
experiment (months-scale horizons).

A literal months-long closed-loop run at f_s = 256 Hz is ~10^9-10^10 samples/policy;
holding the injected noise in RAM (three regime streams, ~100+ GB) is impossible.
Instead we GENERATE THE ENVIRONMENT NOISE ONCE, blended over the (policy-independent)
regime schedule W(t), and STORE it to disk as memory-mapped float32 arrays that the
experiment streams back in chunks.

Why we can pre-blend and store a single stream
----------------------------------------------
All policies see the IDENTICAL environment (common random numbers): the regime
schedule W(t), the injected test-mass noise and the input power are the same for
every policy; only the arm choices (and hence the closed-loop trajectory) differ.
So the expensive, PSD-shaped injected noise
    n(t) = sum_i W_i(t) n_i(t),   P_in(t) = sum_i W_i(t) P_i(t)
is computed once and shared. We store the two blended test-mass streams
(ITM, ETM) and the blended input power.  Sensor (readout) noise is white and
cheap, so it is regenerated deterministically per window at run time.

How the noise is generated
---------------------------
The reference plant synthesises each batch in the frequency domain (white
Fourier amplitudes x root-PSD x suspension TF, then irfft) -- see
Lightsaber.Plant.create_tst_noise_from_top / read_optical_noise.  We replicate
that EXACT construction block-by-block: each block is an independent stationary
realisation of the regime PSDs.  The three regimes are made COHERENT within a
block by resetting the block RNG to the same seed before shaping each regime
(regimes then differ only through their ASD file and seismic scale, exactly as in
bandit_experiment.generate_regime_noises).  Blocks are large (default 4096 s) so
the only artefact is a short filter transient at each block edge, negligible
against the multi-month horizon (the reward's slowest band settles in ~60 s).

Outputs (in <cache_dir>/):
    sus_itm.f32, sus_etm.f32, power.f32   -- memmaps, length N = horizon*fs
    manifest.npz                          -- horizon, fs, block, seed, env params,
                                             per-second W, sens_t (for plots/oracle)

CLI:
    python bandit_noise_cache.py --horizon 3600 --out bandit_runs/cache_demo
    python bandit_noise_cache.py --horizon 15552000 --block 4096 \
        --out bandit_runs/cache_6mo          # 6 months (~48 GB fp32)
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.append(os.getcwd()); sys.path.append(os.path.join(os.getcwd(), 'bandit'))
import switching_dynamics as sd
from bandit_simulation import (REGIME_FILES, REGIME_POWER_SCALE, REGIME_SEISMIC_SCALE,
                               REGIME_OSEM_SCALE, REGIME_SENSING_SCALE, FS)
from bandit_experiment import RegimeEnv, SENS

REGIMES = (0, 1, 2)
_II = np.array([0, 2, 3, 1, 2, 3])          # ns/tf index per tst component (Plant convention)


class RegimeNoiseGen:
    """Block-wise FFT synthesiser for one regime, reproducing Plant's construction.

    Loads the regime's PSDs/TFs once (via sd.setup) and precomputes the complex
    spectral shaping for a fixed block length; block() then only draws white noise
    and irffts -- fast enough for months of samples.
    """
    def __init__(self, regime, block_n, fs=FS):
        self.regime = regime; self.block_n = int(block_n); self.fs = fs
        plant, _, _, _, _ = sd.setup(regime, 8, seed=0)          # load PSD/TF models
        self.P = plant.P; self.scale_RIN = plant.scale_RIN
        scale = np.array([plant.scale_ITM_ISI_L, plant.scale_ETM_ISI_L,
                          plant.scale_OSEM_L, plant.scale_OSEM_P])
        nfreq = self.block_n // 2 + 1
        freq = np.linspace(0, fs // 2, nfreq)
        T_batch = self.block_n / fs
        self.norm = 0.5 * (T_batch) ** 0.5                       # = 0.5*(1/delta_freq)**0.5
        self.nfreq = nfreq
        # tst: precompute scale * rootPSD * TF (complex) per component
        self.shape_tst = np.zeros((6, nfreq), complex)
        for k in range(6):
            j = _II[k]
            rpsd = np.interp(freq, plant.ns[j]['ff'], plant.ns[j]['rPSD'], left=0, right=0)
            tf = np.interp(freq, plant.tfs[j]['ff'], plant.tfs[j]['tf'], left=0, right=0)
            self.shape_tst[k] = scale[j] * rpsd * tf
        self.shape_tst[:, 0] = 0.0                               # DC = 0
        # optical power: rootPSD (scaled by scale_RIN)
        import numpy as _np
        power_psd = _np.genfromtxt('noise_inputs/O3_power_psd.csv', delimiter=',')
        self.shape_pow = np.interp(freq, power_psd[:, 0], self.scale_RIN * power_psd[:, 1],
                                   left=0, right=0)
        self.shape_pow[0] = 0.0

    def _white(self, rng):
        re = rng.normal(0, self.norm, self.nfreq)
        im = rng.normal(0, self.norm, self.nfreq)
        return re + 1j * im

    def block(self, seed):
        """Return (tst_itm(n,), tst_etm(n,), power(n,)) for one block, RNG=seed.

        Draw order (power, then 6 tst) is identical across regimes, so regimes
        built with the same seed are coherent."""
        rng = np.random.RandomState(seed)
        wp = self._white(rng)
        power = (1.0 + np.fft.irfft(wp * self.shape_pow, n=self.block_n) * self.fs) * self.P
        comps = np.empty((6, self.block_n))
        for k in range(6):
            w = self._white(rng)
            comps[k] = np.fft.irfft(w * self.shape_tst[k], n=self.block_n) * self.fs
        itm = comps[:3].sum(0); etm = comps[3:].sum(0)
        return itm, etm, power


def build_env_and_W(horizon, period, seed, fs=FS, **env_kwargs):
    """RegimeEnv + per-second weights W (3, S+1) and severity/sens for the horizon.

    env_kwargs forwards to RegimeEnv (e.g. spike_rate, spike_attack, spike_hold,
    spike_decay, mu, drift_ramp, timing_std, exponent, hard_threshold)."""
    env = RegimeEnv(horizon, period=(period or None), seed=seed, **env_kwargs)
    S = int(np.ceil(horizon)) + 1
    tsec = np.arange(S)
    Wsec = np.array([env.get_weights(float(t)) for t in tsec]).T        # (3, S)
    return env, tsec, Wsec


def generate(horizon, out_dir, block=4096, period=0.0, seed=100, dtype='float32', force=False,
             **env_kwargs):
    """env_kwargs (spike_rate, spike_attack, spike_hold, spike_decay, mu, drift_ramp, ...)
    are forwarded to RegimeEnv and become part of the manifest cache key, so changing any
    of them (e.g. to lengthen R1 spikes) correctly triggers regeneration."""
    fs = FS
    N = int(round(horizon * fs))
    block_n = int(block * fs)
    n_blocks = int(np.ceil(N / block_n))
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.npz")
    env_key = json.dumps(env_kwargs, sort_keys=True)
    if os.path.exists(manifest_path) and not force:
        m = np.load(manifest_path, allow_pickle=True)
        if int(m['N']) == N and int(m['block_n']) == block_n and str(m['dtype']) == dtype \
                and int(m['seed']) == seed and float(m['period']) == float(period) \
                and str(m.get('env_kwargs', '{}')) == env_key:
            print(f"[cache] up-to-date at {out_dir} (N={N}, {n_blocks} blocks) -- skipping.")
            return out_dir
        print("[cache] manifest differs -> regenerating.")

    print(f"[cache] horizon={horizon:.0f}s  N={N:,}  block={block}s ({block_n:,} samp) "
          f"n_blocks={n_blocks}  dtype={dtype}")
    if env_kwargs:
        print(f"[cache] env overrides: {env_kwargs}")
    gb = 3 * N * np.dtype(dtype).itemsize / 1e9
    print(f"[cache] on-disk size ~ {gb:.2f} GB (sus_itm+sus_etm+power)")

    env, tsec, Wsec = build_env_and_W(horizon, period, seed, fs, **env_kwargs)
    sens_sec = np.array([SENS[0], SENS[1], SENS[2]]) @ Wsec                # (S,)
    gens = {R: RegimeNoiseGen(R, block_n, fs) for R in REGIMES}

    sus_itm = np.lib.format.open_memmap(os.path.join(out_dir, "sus_itm.npy"),
                                        mode='w+', dtype=dtype, shape=(N,))
    sus_etm = np.lib.format.open_memmap(os.path.join(out_dir, "sus_etm.npy"),
                                        mode='w+', dtype=dtype, shape=(N,))
    power = np.lib.format.open_memmap(os.path.join(out_dir, "power.npy"),
                                      mode='w+', dtype=dtype, shape=(N,))

    t0 = time.time()
    tsamp_all = np.arange(block_n) / fs
    for b in range(n_blocks):
        a0 = b * block_n; a1 = min(a0 + block_n, N); nb = a1 - a0
        # blend weights for this block (interp per-second W to samples)
        tsamp = (a0 / fs) + tsamp_all[:nb]
        W = np.vstack([np.interp(tsamp, tsec, Wsec[i]) for i in range(3)])   # (3, nb)
        bi = np.zeros(nb); be = np.zeros(nb); bp = np.zeros(nb)
        for R in REGIMES:
            itm, etm, pw = gens[R].block(seed + 1000 * b)     # same seed across regimes -> coherent
            bi += W[R] * itm[:nb]; be += W[R] * etm[:nb]; bp += W[R] * pw[:nb]
        sus_itm[a0:a1] = bi.astype(dtype); sus_etm[a0:a1] = be.astype(dtype)
        power[a0:a1] = bp.astype(dtype)
        if b % max(1, n_blocks // 20) == 0 or b == n_blocks - 1:
            el = time.time() - t0
            eta = el / (b + 1) * (n_blocks - b - 1)
            print(f"  block {b+1}/{n_blocks}  ({el:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    sus_itm.flush(); sus_etm.flush(); power.flush()

    np.savez(manifest_path, N=N, fs=fs, horizon=horizon, block_n=block_n, n_blocks=n_blocks,
             dtype=dtype, seed=seed, period=float(period), env_period=env.period,
             env_kwargs=env_key, Wsec=Wsec, tsec=tsec, sens_sec=sens_sec,
             n_spikes=len(env.intervals), n_drift=len(env.drift))
    print(f"[cache] done in {time.time()-t0:.0f}s -> {out_dir}")
    return out_dir


def load_cache(cache_dir):
    """Return dict with memmaps (read-only) + manifest arrays."""
    m = np.load(os.path.join(cache_dir, "manifest.npz"), allow_pickle=True)
    d = {k: m[k] for k in m.files}
    d['sus_itm'] = np.load(os.path.join(cache_dir, "sus_itm.npy"), mmap_mode='r')
    d['sus_etm'] = np.load(os.path.join(cache_dir, "sus_etm.npy"), mmap_mode='r')
    d['power'] = np.load(os.path.join(cache_dir, "power.npy"), mmap_mode='r')
    return d


def main():
    ap = argparse.ArgumentParser(description="Generate & store the long-run environment noise.")
    ap.add_argument("--horizon", type=float, required=True, help="horizon [s] (6 months = 15552000)")
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--block", type=float, default=4096, help="FFT block length [s]")
    ap.add_argument("--period", type=float, default=0.0, help="diurnal period [s] (0 = horizon/2.5)")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--spike-rate", type=float, default=None, help="R1 Poisson rate per diurnal period (RegimeEnv default 4.0)")
    ap.add_argument("--spike-attack", type=float, default=None, help="R1 mean ramp-up [s] (default 30.0)")
    ap.add_argument("--spike-hold", type=float, default=None, help="R1 mean plateau duration [s] (default 150.0)")
    ap.add_argument("--spike-decay", type=float, default=None, help="R1 mean ramp-down [s] (default 30.0)")
    args = ap.parse_args()
    env_kwargs = {}
    if args.spike_rate is not None: env_kwargs['spike_rate'] = args.spike_rate
    if args.spike_attack is not None: env_kwargs['spike_attack'] = args.spike_attack
    if args.spike_hold is not None: env_kwargs['spike_hold'] = args.spike_hold
    if args.spike_decay is not None: env_kwargs['spike_decay'] = args.spike_decay
    generate(args.horizon, args.out, block=args.block, period=args.period,
             seed=args.seed, dtype=args.dtype, force=args.force, **env_kwargs)


if __name__ == "__main__":
    main()
