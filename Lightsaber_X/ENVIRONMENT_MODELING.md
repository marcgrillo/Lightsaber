# Non-Stationary Environmental Physics Model

This document outlines the rigorous mathematical and physical modeling implemented in `Lightsaber_X` to simulate a dynamically non-stationary environment. The primary goal is to provide a testing ground for Bandit algorithms (e.g., DP-SKF).

## 1. Physical Scale Modulation

The core mechanic driving the regime transitions is **not just a change in spectrum shape**, but a massive scalar shift in the physical geometries of the interferometers. As the environment transitions from a `"Nominal"` state to an `"Extreme"` microseismic state (simulating a severe storm), the following base multipliers are applied dynamically:

| Parameter | C0 (Nominal) | C1 (High-Micro) | C2 (Extreme) |
| :--- | :--- | :--- | :--- |
| **Input Power Scale** | `1.0` | `0.95` | `0.90` |
| **Sensing Noise Scale** | `120.0` | `12.0` | `0.25` |
| **Seismic Noise Scale** | `1.0` | `6.0` | `20.0` |

> [!NOTE]
> **Why are these scales necessary?**
> A simple modification of the ASD spectral shape (e.g., adding a resonance peak) is physically insufficient to drive deep performance disparities between controllers. The massive 20x injection of seismic noise forces the baseline `C0_nominal` controller to fail, while the extreme reduction in high-frequency sensing noise (from 120x down to 0.25x) provides a clean sensor read that specifically favors the aggressive low-frequency rejection profile of `C2_high_micro2`.

## 2. Dynamic Streaming

These parameters are not hard-switched. The `SinusoidalPoissonEnv` maintains a time-dependent, continuous trajectory of mixing weights $W = [W_0(t), W_1(t), W_2(t)]$. 

At every simulation step $t$, the system computes:
$S_{power}(t) = W \cdot \vec{S}_{power}$
$S_{sens}(t) = W \cdot \vec{S}_{sens}$
$S_{seis}(t) = W \cdot \vec{S}_{seis}$

### Kernel Injection
Unlike traditional static-batch simulations, `Lightsaber_X` utilizes a fully compiled Numba fast-physics kernel. 
- The **Sensing Noise** (`sn_s`, `sn_h`) arrays are dynamically generated and geometrically scaled by $S_{sens}(t)$ per 4-second sub-block.
- The **Input Power** is passed as a continuous time-series stream $P_{in}(t) = P_{base} \cdot S_{power}(t)$.
- The **Seismic Noise** is generated pre-kernel via continuous `ColoredNoiseFIR` generators, where the base ASD spectra point-wise values are scaled by $S_{seis}(t)$ *before* filter generation.

## 3. Hardware Controller Syncing

The bandit algorithm makes an exploration decision every 300 seconds ($T_{interval}$). When the bandit selects a new `arm_idx`, the simulated physical hardware immediately loads the corresponding State-Space variables (SOS Matrices) for the new variant (`C0`, `C1`, or `C2`).

**Crucially, the SOS internal states are reset to zero upon switching.** This zeroing eliminates "ringing" from the previous controller's phase delays and allows the new controller an honest, mathematically clean sub-block evaluation for the next log-RMS reward accumulation. (In a real system this is not the case, the controller is always running and the noise is always present. When evaluating real deployment, the controller states should not be reset upon switching. This is a simplification for simulation purposes.)

## 4. Benchmark Performance Target

With the physical scales fully synchronized, the system expects to recover the following absolute baseline rewards (tested using pure SI Radians) in a 100% `Nominal [1.0, 0.0, 0.0]` environment:

- **C0_nominal**: `~172.90` (Optimal)
- **C1_high_micro**: `~169.35`
- **C2_high_micro2**: `~166.35`

This `~6.55` point disparity demonstrates that the controllers are uniquely and physically disjoint. As the environment drifts toward `W_2 -> 1.0`, `C0`'s reward will structurally collapse due to the 20x seismic scaling, triggering the active algorithmic switch.
