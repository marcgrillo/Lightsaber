import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import bandit_rewards

# Add current directory to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bandit'))

# Import Lightsaber components
from Lightsaber import Plant, Sensors, SS_compensation, Controller, Postprocessing

# Import configuration from bandit_simulation (to ensure consistency)
# We assume bandit_simulation.py is in the same directory and has these dicts
from bandit_simulation import (
    DEFAULT_FLAGS, REGIME_POWER_SCALE, REGIME_SENSING_SCALE, 
    REGIME_OSEM_SCALE, REGIME_SEISMIC_SCALE, REGIME_FILES, 
    CONTROLLER_CHOICES, T_FFT, FS, SEGMENT_DURATION
)

def run_case(regime_key, controller_key, duration_s, feat_names, w_weights, seed=1):
    """
    Runs a simulation for a specific Regime and Controller for a given duration.
    Returns list of raw scores.
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
    
    data = {'sampling_frequency': flags['fs'], 'duration_batch': duration_s, 'duration_fft': T_FFT}
    physics = {'P': flags['P'], 'L': flags['L'], 'R_ITM': flags['R_ITM'], 'R_ETM': flags['R_ETM'], 't_ITM': flags['t_ITM']}
    sensing = {'noise_hard_mode': flags['n_hard'], 'noise_soft_mode': flags['n_soft']}
    parameters = {
        'scale_ITM_ISI_L': flags['scale_ITM_ISI_L'], 'scale_ETM_ISI_L': flags['scale_ETM_ISI_L'], 
        'scale_OSEM_L': flags['scale_OSEM_L'], 'scale_OSEM_P': flags['scale_OSEM_P'], 'scale_RIN': flags['scale_RIN']
    }
    
    # Noise Config
    noise_files = [
        regime_file, regime_file, 
        'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt', 
        'noise_inputs/O3_power_psd.csv'
    ]
    transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'

    # Initialize Components
    asc_plant = Plant(physics, data, parameters, plot_dir=None, noise_files=noise_files, reference_data_file=reference_data_file, transfer_files=transfer_files, seed=seed)
    asc_sensors = Sensors(sensing, physics, data, seed=seed)
    asc_SS_compensation = SS_compensation(data, physics, asc_plant, plot_dir=None)
    asc_controller = Controller(data, physics, plot_dir=None, controller_name=controller_name)
    
    # Pre-generate noise (Plant does this in init, but we rely on its internal tst_noise_t)
    # The Plant constructor builds full duration noise.
    
    # Helpers for Eigen conversion
    L_c = 3994.5
    R_ITM_c = 1934
    R_ETM_c = 2245
    g1 = 1 - L_c / R_ITM_c
    g2 = 1 - L_c / R_ETM_c
    r_val = 0.5 * (g1 - g2 + np.sqrt((g1 - g2) ** 2 + 4))
    local2eigen = np.array([[1, r_val], [-r_val, 1]]) / (1 + r_val ** 2)
    
    # Simulation Loop
    N = int(duration_s * flags['fs'])
    BANDIT_STEP_S = 2
    BANDIT_STEP_SAMPLES = int(BANDIT_STEP_S * flags['fs'])
    
    tstP_t = np.zeros((N, 2))
    control_t = np.zeros((N, 2))
    strain_noise_t = np.zeros((N, 1)) # Dummy
    readout_t = np.zeros((N, 2))
    SS_compensation_t = np.zeros((N, 2))
    cavity_power_t = np.zeros((N, 1))
    
    raw_scores = []
    
    desc = f"R{regime_key}|C{controller_key}"
    for k in range(N-1):
        # Propagate
        tstP_t[k+1, :], _, cavity_power_t[k+1, :], _, _, _ = asc_plant.propagate(pum_input_signal=-control_t[k, :], SS_comp = SS_compensation_t[k, :])
        
        if not np.isfinite(tstP_t[k+1, :]).all():
            print(f"NaN detected in {desc} at step {k}")
            break
            
        readout_t[k+1, :] = asc_sensors.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power=cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        
        # Bandit Step
        current_time_step = k + 1
        if current_time_step % BANDIT_STEP_SAMPLES == 0:
            start_idx = current_time_step - BANDIT_STEP_SAMPLES
            end_idx = current_time_step
            
            chunk_tstP = tstP_t[start_idx:end_idx]
            chunk_ctrl = control_t[start_idx:end_idx]
            
            chunk_ang_eig = (local2eigen @ chunk_tstP.T).T
            chunk_ctrl_eig = (local2eigen @ chunk_ctrl.T).T
            
            err_h = chunk_ang_eig[:, 1]
            u_h = chunk_ctrl_eig[:, 1]
            
            feats = {}
            feats.update(bandit_rewards.band_feats_blocks(err_h, bandit_rewards.ERR_BANDS))
            feats.update(bandit_rewards.band_feats_blocks(u_h, bandit_rewards.U_BANDS))
            
            Z = 0.0
            for wi, fn in zip(w_weights, feat_names):
                if fn in feats:
                    Z += wi * np.log(feats[fn] + 1e-30)
            
            raw_scores.append(-np.sum(Z))
            
    return raw_scores

def main():
    # Load weights
    try:
        feat_names, w_weights = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
        print("Loaded weights successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    regimes = [0, 1, 2]
    controllers = [0, 1, 2]
    duration = 200 # 200 seconds per case -> 100 samples/case
    
    results = {}
    
    print(f"Running Characterization (Duration={duration}s per case, Total ~{len(regimes)*len(controllers)*duration/60:.1f} min)...")
    
    pbar = tqdm(total=len(regimes)*len(controllers))
    
    for r in regimes:
        for c in controllers:
            pbar.set_description(f"Simulating R{r}-C{c}")
            scores = run_case(r, c, duration, feat_names, w_weights)
            results[(r, c)] = scores
            pbar.update(1)
    
    pbar.close()
    
    # Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    
    # Global range for histograms
    all_vals = [s for r_scores in results.values() for s in r_scores]
    if not all_vals:
        print("No scores collected!")
        return
        
    g_min, g_max = min(all_vals), max(all_vals)
    bins = np.linspace(g_min, g_max, 20)
    
    for r in regimes:
        for c in controllers:
            ax = axes[r, c]
            scores = results[(r, c)]
            mu = np.mean(scores) if scores else 0
            std = np.std(scores) if scores else 0
            
            ax.hist(scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f"Regime {r} | Controller {c}\n$\mu={mu:.1f}, \sigma={std:.1f}$")
            
            # Highlight optimum
            if r == c:
                ax.set_facecolor('#f0fff0') # Light green for diagonal
                
            ax.grid(True, alpha=0.3)
            
            if r == 2: ax.set_xlabel("Raw Score")
            if c == 0: ax.set_ylabel("Count")

    plt.suptitle("Raw Reward Score Distributions (Unmapped) for all (R, C) Pairs", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("characterize_rewards_9x9.png")
    print("Saved plot to characterize_rewards_9x9.png")
    plt.close(fig)

    # --- Plot 2: Overlaid Histograms (Rows=Regimes, Colors=Controllers) ---
    print("Plotting overlaid histograms...")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Use global range derived above
    # Pads it slightly
    span = g_max - g_min if g_max != g_min else 1.0
    ov_min = g_min - 0.05 * span
    ov_max = g_max + 0.05 * span
    
    for r_idx, r in enumerate(regimes):
        ax = axes2[r_idx]
        ax.set_title(f"Regime {r}")
        ax.set_ylabel("Density")
        
        for c in controllers:
            scores = results[(r, c)]
            if scores:
                mu = np.mean(scores)
                # Overlaid
                ax.hist(scores, bins='auto', range=(ov_min, ov_max), density=True, 
                        histtype='step', linewidth=2, color=colors[c],
                        label=f"C{c} ($\mu$={mu:.1f})")
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes2[-1].set_xlabel("Raw Reward Score")
    plt.tight_layout()
    plt.savefig("characterize_rewards_overlay.png")
    print("Saved plot to characterize_rewards_overlay.png")
    plt.close(fig2)

if __name__ == "__main__":
    main()
