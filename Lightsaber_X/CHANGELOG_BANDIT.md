# Bandit & Environmental Noise Integration (O3 LIGO Model)

This document describes the changes made to `Lightsaber_X` to support online controller selection using the **DP-SKF** bandit algorithm and time-varying environmental noise.

## 1. New Features

### Dirichlet Process Switching Kalman Filter (DP-SKF)
The `algorithms.py` module introduces a Bayesian Bandit (Thompson Sampling) that handles non-stationary rewards.
- **Inference**: Uses a library of Kalman Filters (KFs) to track the performance of each arm (controller).
- **Dirichlet Process**: Allows for the automatic discovery of new "regimes" when the reward changes drastically.
- **Arm Selection**: Every hour (configurable), the bandit selects the best controller based on current state probabilities.

### Sinusoidal Poisson Environment
The `noise_models.py` module implements a realistic, time-varying mixing of three noise regimes:
- **Regime 0 (Nominal)**: Low background noise.
- **Regime 1 (High-Micro)**: Dominated by sudden Poisson-distributed spikes (e.g., storms).
- **Regime 2 (Extreme)**: High ambient noise levels driven by a 24-hour sinusoidal cycle.

### Performance Acceleration
To handle 12-hour (or longer) simulations, we introduced a **Numba-accelerated physics kernel** in `physics_fast.py`.
- **Benchmark**: Achieve >100x speedup relative to the standard simulation loop.
- **Batched Physics**: Physics runs in 4s batches (1024 samples) for speed, but integrates seamlessly with bandit intervals.

### Research-Standard Reward Aggregation
The system now implements a **300-second synchronization** logic:
- **Sample Buffering**: Physics data (Error and Actuation) is buffered over each `bandit_interval`.
- **Statistical Robustness**: A single, high-precision reward is calculated on the full 300s chunk. This ensures low-frequency frequency band features (down to 0.05Hz) are statistically valid.
- **Raw Z-Scoring**: Rewards are calculated as **Raw Z-Scores** (~150-300 range), matching the most robust LIGO research parameters for this model.

## 2. File Overview
- `algorithms.py`: `DP-SKF` and `BanditAlgorithm` base classes.
- `noise_models.py`: `SinusoidalPoissonEnv` and `StreamableNoise`.
- `physics_fast.py`: JIT-compiled Numba physics core.
- `simulate_bandit.py`: New entry point for runs involving the bandit.
- `generate_test_data.py`: Creates realistic, 12-hour high-fidelity noise profiles for validation.

## 3. Advanced Usage & Benchmarking

### Competitive Comparison
To evaluate the bandit against fixed baseline controllers (nominal vs high-micro), use the comparison script:
```bash
python compare_bandit_fixed.py --duration 3600 --output_dir results/test_run
```
This will sequentially run the **DPSKF** bandit and three **Fixed** controllers (C0, C1, C2) and generate a `realtime_cumulative_reward.png` plot showing the bandit's relative advantage.

### Diagnostic Visualizations
Every simulation run now automatically generates a `bandit_log_*.png` file. This 4-panel plot includes:
1. **Regime Evolution**: Tracking the environment weights ($W_0, W_1, W_2$).
2. **Controller Picks**: Visualizing when and why the bandit switches variants.
3. **Reward History**: Raw and moving-average rewards.
4. **Kalman States**: Tracking the internal Log-Likelihood and RMS error estimates.

## 4. Hyperparameter Guide

### Environmental ($\text{SinusoidalPoissonEnv}$)
- `--mu` (Default: 30.0): Frequency of environmental non-stationarity. Higher values create faster regime shifts.
- `--r1_rate` (Default: 3.0): Frequency of Poisson "micro-seismic" spikes in Regime 1.
- `--exponent` (Default: 0.33): Controls the "peakiness" of the transitions.

### Bandit ($\text{DPSKF}$)
- `--alpha` (Default: 1e-05): Dirichlet Process concentration. Controls how aggressively the bandit creates new internal models for unseen environments.
- `--W` (Default: 300): History window (in decision steps) used for Bayesian likelihood evaluation.
- `--Q` (Default: 0.1): Process noise. Controls how quickly the bandit "forgets" old reward averages for a state (drift rate).
- `--R` (Default: 0.04): Measurement noise. High values make the bandit more robust to transient noise but slower to react to real performance changes.
- `--bandit_interval` (Default: 300): Number of seconds between controller re-evaluations.

## 5. Prerequisites & Data
Before running any bandit simulations, you **MUST** generate the realistic noise profiles:
```bash
python generate_test_data.py
```
This creates high-fidelity `.csv` and `.npz` files in `data/temporary_test_data/` (excluded from git) which are required for the sinusoidal mixing models.

## 6. Maintenance / Modifications
If you need to add a new controller:
1. Define the filter coefficients in `configuration/config.yaml` under `PITCH_CONTROL`.
2. Add the name (e.g., `C3_new_design`) to the `CONTROLLER_MAP` in `simulate_bandit.py`.
3. The Bandit's `num_arms` parameter in the initialization will need to be increased to match.

> [!TIP]
> If you change the reward weights, update the `bandit_reward_weights_rel2base.npz` file in `data/`. The calculation is handled in `reward_utils.py`.
