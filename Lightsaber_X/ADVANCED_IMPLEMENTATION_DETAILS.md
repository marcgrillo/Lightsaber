# Advanced Implementation Details: Lightsaber_X

This document outlines the advanced architectural, physical, and computational mechanisms integrated into `Lightsaber_X`. These features transition the framework from a standard linear simulator into a highly optimized, dynamically non-stationary environment capable of resolving robust autonomous machine-learning policies (Bandits) for optomechanical control.

## 1. Batched Numba JIT Physics Kernel (`physics_fast.py`)
To accelerate evaluations from hours to seconds (achieving up to **~15,000x real-time speedups**), the core physical feedback loop was re-cast into a pre-compiled Numba JIT kernel (`run_fast_physics_kernel`). 

### State Persistence & Mechanical Continuity
When chopping a continuous simulation into discrete 1-second batches, naive state re-initialization destroys the memory of recursive filters. `Lightsaber_X` resolves this via an advanced bridging methodology:
- **`last_act` Latching**: The controller's physical actuation signal (`curr_act_sus`) is manually preserved and passed back into the JIT sequence as `last_act`. This prevents the abstract controller outputs from violently resolving to zero every batch boundary, which otherwise triggers punishing step-function mechanical resonances in the synthesized plant.
- **Controller Matrix Persistence**: `Lightsaber.py` applies `.astype(np.float64, copy=False)` dynamically during filter extraction (`get_sos_for_numba()`). Because Numpy's `astype` forces deep-copies by default, previous iterations wiped the internal ZKP (Zero-Pole-Gain) delay lines. Utilizing `copy=False` allows the JIT engine to recursively mutate the *exact original memory addresses* of the high-gain filter logic without artificially lobotomizing the controller.

## 2. Realistic Non-Stationary Environments (`noise_models.py`)
`Lightsaber_X` does not evaluate parameters under flat noise envelopes. It synthetically replicates genuine stochastic seismic anomalies combining diurnal mechanics with randomized spikes.

### The Sinusoidal-Poisson Event Generator
- **Daily Drift**: Simulated via a pseudo-random low-frequency sinusoid envelope.
- **Micro-Spikes**: Rare, intense physical bursts synthesized using a Poisson distribution that inject explicit `[Attack, Hold, Decay]` sequences over the environmental amplitude.

### 6-Stream Authentic Noise Assembly
Within the time boundaries predicted by the Poisson model, `StreamableNoise` blends **6 distinct colored noise traces** per batch:
1. `Seismic Base Disturbance` for ITM
2. `Seismic Base Disturbance` for ETM
3. `Optical Sensor Electro-Magnetic (OSEM) Translation Readout` for ITM
4. `OSEM Translation Readout` for ETM
5. `Pitch-coupled OSEM Force` for ITM
6. `Pitch-coupled OSEM Force` for ETM

The components are independently convolved through authentic optomechanical continuous Transfer Functions (e.g., modeling the top-mass isolating coupling down to the core test masses) to generate true-to-life interferometric noise floors.
Regime transitions explicitly harden when normalized probabilities breach thresholds (`> 0.75` or `< 0.25`), enforcing genuine stability periods mirroring realistic operational sequences.

## 3. Sidles-Sigg (SS) Adaptive Compensation
As interferometric circulating power fluctuates, `physics_fast.py` evaluates Sidles-Sigg cross-coupling phenomena continuously. 
To preserve stability, the active compensation feedback loop calculates dynamic angular optical gain matrices iteratively inside the JIT batch boundaries—locking angular eigenmodes based on the live running state of `P_cav` (Cavity Power), effectively mimicking active wave-front sensor loops.

## 4. Log-RMS Reward Topography (`reward_utils.py`)
The ultimate diagnostic metric for Bandit progression relies on calculating rigorous penalty evaluations of the sustained physical jitter across multiple critical topological frequency bands.
- **Physical SI Calibration**: The performance scores explicitly abstract non-linear Sigmoid bounds out of the physics loop, returning composite scalars purely driven by absolute 1.0 Radian RMS variations. 
- **Sub-Band Evaluation**: `AdvancedRewardCalculator` recursively breaks down independent `error` and `actuation` signals utilizing multi-pole Butterworth bandpass filters mapping spectrum domains from `0.05 Hz` to `100 Hz`.
- **2-Second Blocking Mode**: Employs mathematically weighted averages across finite overlapping subsets to stabilize evaluation profiles of transient Poisson spikes spanning outside simple stationary windows, anchoring the simulation's 'ideal' objective limit at approx **~167.5 to 169.0** log-variance bounds depending on specific regime characteristics.
