# --- repo path bootstrap (this script was moved into dev/; run it as
#     `python dev/<name>.py` from the Lightsaber_X/ directory). Makes the
#     Lightsaber_X package root, the bandit/ package, and this dev/ folder
#     importable, and anchors the working dir to the package root so the
#     relative data/output paths (noise_inputs/, *.npz, bandit_runs/) resolve. ---
import os as _os, sys as _sys
_DEV = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_DEV)
for _p in (_ROOT, _os.path.join(_ROOT, 'bandit'), _DEV):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import bandit_rewards
import argparse

# Add current directory to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bandit'))

# Import Lightsaber components
from Lightsaber import Plant, Sensors, SS_compensation, Controller, Postprocessing

# Import configuration from bandit_simulation (to ensure consistency)
from bandit_simulation import (
    DEFAULT_FLAGS, REGIME_POWER_SCALE, REGIME_SENSING_SCALE, 
    REGIME_OSEM_SCALE, REGIME_SEISMIC_SCALE, REGIME_FILES, 
    CONTROLLER_CHOICES, T_FFT, FS, SEGMENT_DURATION
)

def run_continuous_case(regime_key, controller_key, total_duration_s, feat_names, w_weights, seed=1):
    """
    Runs a continuous simulation for a specific Regime and Controller.
    Returns list of ALL raw scores (every 2s).
    """
    # 1. SETUP PARAMETERS
    regime_name, regime_file = REGIME_FILES[str(regime_key)]
    controller_name = CONTROLLER_CHOICES[int(controller_key)]
    
    flags = DEFAULT_FLAGS.copy()
    osem_scale = REGIME_OSEM_SCALE.get(regime_name, 1.0)
    flags['scale_OSEM_L'] = osem_scale
    flags['scale_OSEM_P'] = osem_scale
    power_scale = REGIME_POWER_SCALE.get(regime_name, 1.0)
    flags['P'] = flags['P'] * power_scale
    seis_scale = REGIME_SEISMIC_SCALE.get(regime_name, 1.0)
    flags['scale_ITM_ISI_L'] = seis_scale
    flags['scale_ETM_ISI_L'] = seis_scale
    sens_scale = REGIME_SENSING_SCALE.get(regime_name, 1.0)
    flags['n_hard'] = flags['n_hard'] * sens_scale
    flags['n_soft'] = flags['n_soft'] * sens_scale
    
    data = {'sampling_frequency': flags['fs'], 'duration_batch': total_duration_s, 'duration_fft': T_FFT}
    physics = {'P': flags['P'], 'L': flags['L'], 'R_ITM': flags['R_ITM'], 'R_ETM': flags['R_ETM'], 't_ITM': flags['t_ITM']}
    sensing = {'noise_hard_mode': flags['n_hard'], 'noise_soft_mode': flags['n_soft']}
    parameters = {
        'scale_ITM_ISI_L': flags['scale_ITM_ISI_L'], 'scale_ETM_ISI_L': flags['scale_ETM_ISI_L'], 
        'scale_OSEM_L': flags['scale_OSEM_L'], 'scale_OSEM_P': flags['scale_OSEM_P'], 'scale_RIN': flags['scale_RIN']
    }
    
    noise_files = [regime_file, regime_file, 'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt', 'noise_inputs/O3_power_psd.csv']
    transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'

    asc_plant = Plant(physics, data, parameters, plot_dir=None, noise_files=noise_files, reference_data_file=reference_data_file, transfer_files=transfer_files, seed=seed)
    asc_sensors = Sensors(sensing, physics, data, seed=seed)
    asc_SS_compensation = SS_compensation(data, physics, asc_plant, plot_dir=None)
    asc_controller = Controller(data, physics, plot_dir=None, controller_name=controller_name)
    
    L_c = 3994.5
    R_ITM_c = 1934
    R_ETM_c = 2245
    g1 = 1 - L_c / R_ITM_c
    g2 = 1 - L_c / R_ETM_c
    r_val = 0.5 * (g1 - g2 + np.sqrt((g1 - g2) ** 2 + 4))
    local2eigen = np.array([[1, r_val], [-r_val, 1]]) / (1 + r_val ** 2)
    
    N = int(total_duration_s * flags['fs'])
    BANDIT_STEP_S = 2
    BANDIT_STEP_SAMPLES = int(BANDIT_STEP_S * flags['fs'])
    
    tstP_t = np.zeros((N, 2))
    control_t = np.zeros((N, 2))
    readout_t = np.zeros((N, 2))
    SS_compensation_t = np.zeros((N, 2))
    cavity_power_t = np.zeros((N, 1))
    
    raw_scores = []
    
    desc = f"R{regime_key}|C{controller_key}"
    # tqdm per case
    for k in range(N-1):
        tstP_t[k+1, :], _, cavity_power_t[k+1, :], _, _, _ = asc_plant.propagate(pum_input_signal=-control_t[k, :], SS_comp = SS_compensation_t[k, :])
        
        if not np.isfinite(tstP_t[k+1, :]).all():
            print(f"NaN detected in {desc} at step {k}")
            break
            
        readout_t[k+1, :] = asc_sensors.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power=cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        
    # --- Post-Simulation Processing (Continuous Filtering) ---
    # Convert whole trace to Eigen
    ang_eig = (local2eigen @ tstP_t.T).T
    ctrl_eig = (local2eigen @ control_t.T).T
    
    err_h = ang_eig[:, 1]
    u_h = ctrl_eig[:, 1]
    
    # Compute features on the FULL trace
    # band_feats_blocks handles 'to_blocks' internal splitting
    feats = {}
    feats.update(bandit_rewards.band_feats_blocks(err_h, bandit_rewards.ERR_BANDS))
    feats.update(bandit_rewards.band_feats_blocks(u_h, bandit_rewards.U_BANDS))
    
    # Calculate Z for all blocks
    n_blocks = len(next(iter(feats.values())))
    Z = np.zeros(n_blocks)
    for wi, fn in zip(w_weights, feat_names):
        if fn in feats:
            Z += wi * np.log(feats[fn] + 1e-30)
            
    raw_scores = -Z
            
    return raw_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Number of 200s averaged samples to collect")
    args = parser.parse_args()
    
    n_averages = args.samples
    window_duration = 200 # seconds
    block_steps = window_duration // 2 # 100 discrete 2s rewards
    
    total_case_duration = n_averages * window_duration
    
    print(f"Goal: Collect {n_averages} samples per case.")
    print(f"Each sample is mean of {block_steps} raw rewards (over {window_duration}s).")
    print(f"Total Sim Time per Case: {total_case_duration}s.")
    
    try:
        feat_names, w_weights = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    regimes = [0, 1, 2]
    controllers = [0, 1, 2]
    
    results = {}
    
    total_ops = len(regimes) * len(controllers)
    pbar_main = tqdm(total=total_ops, desc="Combined Cases")

    for r in regimes:
        for c in controllers:
            pbar_main.set_description(f"R{r}-C{c}")
            
            # Run one long continuous simulation
            # Note: seed=1 is constant, but time evolution provides variance
            raw_scores = run_continuous_case(r, c, total_case_duration, feat_names, w_weights, seed=1)
            
            # Chunk and Average
            raw_array = np.array(raw_scores)
            
            # Ensure we have enough data (handle potential dropouts)
            max_len = n_averages * block_steps
            if len(raw_array) < max_len:
                print(f"Warning: R{r}C{c} yielded fewer scores ({len(raw_array)}) than expected ({max_len})")
                raw_array = np.pad(raw_array, (0, max_len - len(raw_array)), 'edge')
            else:
                raw_array = raw_array[:max_len]
                
            # Reshape (n_averages, block_steps)
            chunks = raw_array.reshape(n_averages, block_steps)
            
            # Average over block_steps (axis 1)
            averaged_samples = np.mean(chunks, axis=1)
            
            results[(r, c)] = averaged_samples
            pbar_main.update(1)
            
    pbar_main.close()
    
    # Plotting Overlaid Histograms
    print("Plotting results...")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    all_vals = [s for r_scores in results.values() for s in r_scores]
    if all_vals:
         g_min, g_max = min(all_vals), max(all_vals)
         span = g_max - g_min if g_max != g_min else 1.0
         ov_min = g_min - 0.05 * span
         ov_max = g_max + 0.05 * span
    else:
         ov_min, ov_max = 0, 1

    for r_idx, r in enumerate(regimes):
        ax = axes[r_idx]
        ax.set_title(f"Regime {r} (Averaged 200s Samples)")
        ax.set_ylabel("Density")
        
        for c in controllers:
            scores = results[(r, c)]
            if len(scores) > 0:
                mu = np.mean(scores)
                # Overlaid
                ax.hist(scores, bins='auto', range=(ov_min, ov_max), density=True, 
                        histtype='step', linewidth=2, color=colors[c],
                        label=f"C{c} ($\mu$={mu:.1f})")
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Mean Reward (over 200s)")
    plt.tight_layout()
    plt.savefig("characterize_averaged_overlay.png")
    print("Saved plot to characterize_averaged_overlay.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
