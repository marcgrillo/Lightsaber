import os
import shutil
import numpy as np
import sys
import argparse
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

# Add current directory to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bandit'))

# Import from Lightsaber
from Lightsaber import Plant, Sensors, SS_compensation, Controller, FilterLP, Postprocessing, plot_psd, plot_hoft, plot_diff_disp_noise
from bandit.adswitch import AdSwitch
import bandit_rewards

# --- Configuration Constants ---
FS = 256
SEGMENT_DURATION = 32  # Seconds per bandit step
T_FFT = 4

# Parameter dictionaries
DEFAULT_FLAGS = {
    'fs': 256,
    'P': 705, 'L': 3994.5, 'R_ITM': 1934, 'R_ETM': 2245, 't_ITM': 0.014,
    'n_hard': 3e-14, 'n_soft': 1e-13,
    'f_pass': 10, 'f_stop': 30, 'min_att': 20,
    'scale_ITM_ISI_L': 1, 'scale_ETM_ISI_L': 1,
    'scale_OSEM_L': 1, 'scale_OSEM_P': 1,
    'scale_RIN': 1,
}

REGIME_POWER_SCALE = { "R0_nominal": 1.0, "R1_high_micro": 0.95, "R2_high_micro2": 0.9 }
REGIME_SENSING_SCALE = { "R0_nominal": 120.0, "R1_high_micro": 12.0, "R2_high_micro2": 0.25 }
REGIME_OSEM_SCALE = { "R0_nominal": 1.0, "R1_high_micro": 1.0, "R2_high_micro2": 1.0 }
REGIME_SEISMIC_SCALE = { "R0_nominal": 1.0, "R1_high_micro": 6.0, "R2_high_micro2": 20.0 }

REGIME_FILES = {
    "0": ("R0_nominal", "noise_inputs/ASD_R0_nominal.csv"),
    "1": ("R1_high_micro", "noise_inputs/ASD_R1_high_micro.csv"),
    "2": ("R2_high_micro2",   "noise_inputs/ASD_R2_high_micro2.csv"),
}

CONTROLLER_CHOICES = {
    0: "C0_nominal",
    1: "C1_high_micro",
    2: "C2_high_micro2",
}

def patched_closed_loop_run(asc_plant, asc_sensing, asc_SS_compensation, asc_controller, postprocessing, data, plot_dir, reference_data_file):
    """
    Patched version of closed_loop_run that respects data['duration_batch'].
    """
    asc_plant.reset_counters()

    N = int(data['duration_batch']*data['sampling_frequency']) # Ensure int

    # SIGNALS
    strain_noise_t = np.zeros((N, 1))
    readout_t = np.zeros((N, 2))
    SS_compensation_t = np.zeros((N, 2))
    control_t = np.zeros((N, 2))
    cavity_power_t = np.zeros((N, 1))

    # AUXILIARY
    tstP_t = np.zeros((N, 2))
    tstBS_t = np.zeros((N, 2))
    asc_noise_t = np.zeros((N, 1))
    whitened_strain_t = np.zeros((N, 1))
    torque_noise_t = np.zeros((N, 2))
    
    for k in tqdm(range(N-1), desc="Simulating"):
        tstP_t[k+1, :], tstBS_t[k+1, :], cavity_power_t[k+1, :], asc_noise_t[k+1, :], strain_noise_t[k+1, :], torque_noise_t[k+1, :] = asc_plant.propagate(pum_input_signal=-control_t[k, :], SS_comp = SS_compensation_t[k, :])

        if (not np.isfinite(tstP_t[k+1, :]).all() or
            not np.isfinite(cavity_power_t[k+1, :]).all() or
            not np.isfinite(control_t[k, :]).all()):
            print("NaN/Inf detected at step", k+1)
            sys.exit(0)

        readout_t[k+1, :] = asc_sensing.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power = cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        whitened_strain_t[k+1, :] = postprocessing.sample(strain_noise = strain_noise_t[k+1, :])
        
        if np.any(np.abs(tstP_t[k+1, :]) > 1) or np.any(~np.isfinite(tstP_t[k+1, :])):
            print('Diverging time series at', np.round(100.*k/N),'%')
            sys.exit(0)

    # --- Plotting/Saving ---
    L = 3994.5
    R_ITM = 1934
    R_ETM = 2245
    dL_2_strain = np.sqrt(2.)/L
    g1 = 1 - L / R_ITM
    g2 = 1 - L / R_ETM
    r = 0.5 * (g1 - g2 + np.sqrt((g1 - g2) ** 2 + 4))
    local2eigen = np.array([[1, r], [-r, 1]]) / (1 + r ** 2)
    
    # FIX: Use correct N from duration
    # N is already correct here from top of function
    
    angles_eigen = np.zeros((N, 2))
    for k in range(N):
        angles_eigen[k, :] = local2eigen @ tstP_t[k, :]

    if plot_dir is not None:
        labels1 = ['ASC noise', 'total noise', 'total whitened noise']
        strain_noises = np.hstack((asc_noise_t, strain_noise_t, whitened_strain_t))

        # plot detector data as GW strain noise
        plot_hoft(strain_noises, data['duration_fft'], data['sampling_frequency'],
                    reference_data_file=reference_data_file, label = labels1,
                    filename=os.path.join(plot_dir, f'StrainNoise.png'))

        # plot detector data as displacement noise
        plot_diff_disp_noise(strain_noises/dL_2_strain, data['duration_fft'], data['sampling_frequency'], label = labels1,
                    filename=os.path.join(plot_dir, f'Diff_disp_noise.png'))
        
        control_eigen = np.zeros((N, 2))
        for k in range(N):
            control_eigen[k, :] = local2eigen @ control_t[k, :]

        # plot of TST pitch motion and output of linear controller
        for col_idx, plot_name in [(0,"ITM"), (1, "ETM")]:
            plot_psd(control_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                    ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_output_closed_loop_{plot_name}.png'))

        # plot of pitch motion as seen by the soft-mode and hard-mode sensors
        for col_idx, plot_name in [(0,"soft"), (1, "hard")]:
            plot_psd(angles_eigen[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                    ylabel='TST P [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'tstP_closed_loop_{plot_name}.png'))   
            plot_psd(readout_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                    ylabel='Control input [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_input_closed_loop_{plot_name}.png'))
            plot_psd(control_eigen[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                    ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_output_closed_loop_{plot_name}.png'))
        
        # Save all data
        def save_csv(name, arr):
            np.savetxt(os.path.join(plot_dir, name), arr, delimiter=" ")

        save_csv("torque_noise_t.csv", torque_noise_t)
        save_csv("asc_noise.csv", asc_noise_t)
        save_csv("strain_noise.csv", strain_noise_t)
        save_csv("whitened_strain_noise.csv", whitened_strain_t)
        save_csv("cavity_power.csv", cavity_power_t)
        save_csv("control_ITM_ETM.csv", control_t)
        save_csv("SS_compensation.csv", SS_compensation_t)
        save_csv("BSM_ITM_ETM.csv", tstBS_t)
        save_csv("ITM_ETM_angle.csv", tstP_t)
        save_csv("angles_eigen.csv", angles_eigen)
        save_csv("control_input.csv", readout_t)

    return tstP_t, readout_t, control_t, strain_noise_t


def run_simulation_segment(regime_key, controller_key, duration, plot_dir, seed=None):
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
    
    data = {'sampling_frequency': flags['fs'], 'duration_batch': duration, 'duration_fft': T_FFT}
    physics = {'P': flags['P'], 'L': flags['L'], 'R_ITM': flags['R_ITM'], 'R_ETM': flags['R_ETM'], 't_ITM': flags['t_ITM']}
    sensing = {'noise_hard_mode': flags['n_hard'], 'noise_soft_mode': flags['n_soft']}
    parameters = {
        'scale_ITM_ISI_L': flags['scale_ITM_ISI_L'], 
        'scale_ETM_ISI_L': flags['scale_ETM_ISI_L'], 
        'scale_OSEM_L': flags['scale_OSEM_L'], 
        'scale_OSEM_P': flags['scale_OSEM_P'], 
        'scale_RIN': flags['scale_RIN']
    }
    
    os.makedirs(plot_dir, exist_ok=True)
    
    noise_files = [
        regime_file, regime_file,
        'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt',
        'noise_inputs/O3_power_psd.csv'
    ]
    transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'

    # Pass seed to Plants and Sensors if provided
    asc_plant = Plant(physics, data, parameters, plot_dir=None, noise_files=noise_files, reference_data_file=reference_data_file, transfer_files=transfer_files, seed=seed)
    asc_sensors = Sensors(sensing, physics, data, seed=seed)
    asc_SS_compensation = SS_compensation(data, physics, asc_plant, plot_dir=None)
    asc_controller = Controller(data, physics, plot_dir=None, controller_name=controller_name)
    postprocessing = Postprocessing(data)

    # Use patched runner
    patched_closed_loop_run(asc_plant, asc_sensors, asc_SS_compensation, asc_controller, postprocessing, data, plot_dir, reference_data_file)
    
    return True

def plot_reward_distributions(history, step_num, plot_dir):
    """
    Plots the distribution of rewards for each regime and controller.
    Rows: Regimes (0, 1, 2)
    Cols: Single column with histograms for C0, C1, C2 overlaid or side-by-side?
    User request: "three histograms of rewards in each subplot, one for each controller"
    """
    regimes = ["0", "1", "2"]
    controllers = [0, 1, 2]
    # User request: better colors. Using Tableau 10 standard.
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    if step_num == 0: # Handle single subplot case if len(regimes)==1 (unlikely) or just safety
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
        
    all_rewards = [h['reward'] for h in history]
    if not all_rewards:
        min_r, max_r = 0, 1
    else:
        min_r, max_r = min(all_rewards), max(all_rewards)
        # Add some padding
        span = max_r - min_r if max_r != min_r else 1.0
        min_r -= 0.1 * span
        max_r += 0.1 * span

    for i, r_key in enumerate(regimes):
        ax = axes[i]
        ax.set_title(f"Regime {r_key}")
        ax.set_ylabel("Density")
        
        # Filter history for this regime
        regime_hist = [h for h in history if h['regime'] == r_key]
        
        for c_idx in controllers:
            # rewards for this controller in this regime
            rs = [h['reward'] for h in regime_hist if h['chosen_arm'] == c_idx]
            
            if rs:
                # User request: normalize density, histtype step, auto bins
                mean_r = np.mean(rs)
                ax.hist(rs, bins='auto', range=(min_r, max_r), density=True, histtype='step', linewidth=2, label=f"C{c_idx} (N={len(rs)}, $\mu$={mean_r:.3f})", color=colors[c_idx])
            else:
                pass
        
        # Only show legend if there are labels
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Reward")
    plt.tight_layout()
    # User request: overwrite plot at each step
    plt.savefig(os.path.join(plot_dir, "reward_distribution.png"))
    plt.close(fig)

def generate_stitched_noise(total_duration, regime_schedule_list, segment_duration, seed):
    """
    Generates composite noise arrays (tst_noise_t, input_power) by running Plant generation
    for each regime type and stitching them according to the schedule.
    
    regime_schedule_list: list of regime keys ['0', '0', ... '1', '1', ...] per segment.
    """
    
    # We need to instantiate plants for each regime type to get their noise
    # This is expensive but correct.
    # To save time, we can instantiate just 3 plants (R0, R1, R2) with full duration? 
    # No, that's 3x memory.
    # Better: Instantiate 3 plants with SHORT duration just to read scalings? 
    # Actually, Lightsaber generates noise in `create_tst_noise_from_top` using FFT batches.
    # We can generate the full duration for ONE plant, but modifying the 'noise_file' inputs 
    # or scaling factors is tricky because `Plant` does it all in `__init__`.
    
    # Workaround:
    # 1. Calculate N samples per segment.
    # 2. For each distinct regime in sequence (R0, R1, R2), we assume they are grouped.
    #    (The schedule is 0...0, 1...1, 2...2).
    # 3. We can instantiate Plant_R0 for T/3, Plant_R1 for T/3, Plant_R2 for T/3.
    # 4. Concatenate their noises.
    # 5. Handle transitions carefully? Lightsaber batch noise is independent.
    
    # Determine split points
    # regime_schedule_list is like ['0', ..., '0', '1', ..., '1', '2', ..., '2']
    # We just need to know how long each regime lasts.
    
    counts = {}
    for r in regime_schedule_list:
        counts[r] = counts.get(r, 0) + 1
    
    # durations for R0, R1, R2
    # We assume standard 0->1->2 ordering for simplicity of implementation here, 
    # matching the main loop logic.
    
    stitched_tst_noise = []
    stitched_input_power = []
    
    # We need a dummy Data dict to pass to Plant
    # FS is constant
    
    # Helper to create a partial plant and extract noise
    def get_noise_for_regime(r_key, dur, r_seed):
        regime_name, regime_file = REGIME_FILES[str(r_key)]
        
        flags = DEFAULT_FLAGS.copy()
        osem_scale = REGIME_OSEM_SCALE.get(regime_name, 1.0)
        flags['scale_OSEM_L'] = osem_scale
        flags['scale_OSEM_P'] = osem_scale
        power_scale = REGIME_POWER_SCALE.get(regime_name, 1.0)
        flags['P'] = flags['P'] * power_scale
        seis_scale = REGIME_SEISMIC_SCALE.get(regime_name, 1.0)
        flags['scale_ITM_ISI_L'] = seis_scale
        flags['scale_ETM_ISI_L'] = seis_scale
        
        # We don't care about sensing/control here, just Noise generation
        
        data = {'sampling_frequency': flags['fs'], 'duration_batch': dur, 'duration_fft': T_FFT}
        physics = {'P': flags['P'], 'L': flags['L'], 'R_ITM': flags['R_ITM'], 'R_ETM': flags['R_ETM'], 't_ITM': flags['t_ITM']}
        parameters = {
            'scale_ITM_ISI_L': flags['scale_ITM_ISI_L'], 
            'scale_ETM_ISI_L': flags['scale_ETM_ISI_L'], 
            'scale_OSEM_L': flags['scale_OSEM_L'], 
            'scale_OSEM_P': flags['scale_OSEM_P'], 
            'scale_RIN': flags['scale_RIN']
        }
        
        noise_files = [
            regime_file, regime_file,
            'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt',
            'noise_inputs/O3_power_psd.csv'
        ]
        transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
        reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'
        
        # Instantiate Plant
        # Note: We must ensure seed varies if we want variance, 
        # but here we want the noise for this specific chunk.
        # We can use the master seed + some offset for regime chunks?
        # Or just pass the seed.
        p = Plant(physics, data, parameters, plot_dir=None, noise_files=noise_files, reference_data_file=reference_data_file, transfer_files=transfer_files, seed=r_seed)
        return p.tst_noise_t, p.input_power

    # Generate chunks
    # We iterate 0, 1, 2.
    current_seed = seed
    for r_key in ["0", "1", "2"]:
        if r_key not in counts: continue
        
        dur = counts[r_key] * segment_duration
        if dur == 0: continue
        
        print(f"Generating noise for Regime {r_key} ({dur}s)...")
        # We update seed for each chunk to avoid repeating patterns if R0 and R1 have same params (unlikely but good practice)
        # Using current_seed
        chunk_noise, chunk_power = get_noise_for_regime(r_key, dur, current_seed)
        
        stitched_tst_noise.append(chunk_noise)
        stitched_input_power.append(chunk_power)
        
        current_seed += 1 # Increment seed for next chunk
        
    # Concatenate
    # tst_noise_t shape is (2, N) I think? Lightsaber: np.zeros((2*(len(frequencies)-1), 6)) -> sum -> (2, N)
    # let's check shape.
    # Plant.read_optical_noise -> self.input_power length N
    # Plant.create_tst_noise_from_top -> self.tst_noise_t shape (2, N)
    
    full_noise = np.concatenate(stitched_tst_noise, axis=1)
    full_power = np.concatenate(stitched_input_power)
    
    return full_noise, full_power

def run_interactive_simulation(total_duration, regime_schedule, regime_block_duration, seed, bandit, feat_names, w_weights, plot_dir, bandit_window_duration=100):
    """
    Runs one continuous simulation loop.
    """
    
    # 1. SETUP MASTER PLANT (Regime 0 Init)
    # We start with Regime 0 parameters, but we will overwrite noise.
    start_regime = regime_schedule[0]
    regime_name, regime_file = REGIME_FILES[start_regime]
    
    flags = DEFAULT_FLAGS.copy()
    # Initial scalings (R0)
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
    
    data = {'sampling_frequency': flags['fs'], 'duration_batch': total_duration, 'duration_fft': T_FFT}
    physics = {'P': flags['P'], 'L': flags['L'], 'R_ITM': flags['R_ITM'], 'R_ETM': flags['R_ETM'], 't_ITM': flags['t_ITM']}
    sensing = {'noise_hard_mode': flags['n_hard'], 'noise_soft_mode': flags['n_soft']}
    parameters = {
        'scale_ITM_ISI_L': flags['scale_ITM_ISI_L'], 'scale_ETM_ISI_L': flags['scale_ETM_ISI_L'], 
        'scale_OSEM_L': flags['scale_OSEM_L'], 'scale_OSEM_P': flags['scale_OSEM_P'], 'scale_RIN': flags['scale_RIN']
    }

    noise_files = [regime_file, regime_file, 'noise_inputs/n_osem_L.txt', 'noise_inputs/n_osem_P.txt', 'noise_inputs/O3_power_psd.csv']
    transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'

    print("Initializing Main Plant...")
    asc_plant = Plant(physics, data, parameters, plot_dir=None, noise_files=noise_files, reference_data_file=reference_data_file, transfer_files=transfer_files, seed=seed)
    asc_sensors = Sensors(sensing, physics, data, seed=seed)
    asc_SS_compensation = SS_compensation(data, physics, asc_plant, plot_dir=None)
    
    # 2. INJECT COMPOSITE NOISE
    print("Generating and Injecting Stitched Noise...")
    # Regimes are distinct. We pass just the block duration.
    # The generate_stitched_noise function iterates based on logic "counts * segment_duration".
    # Here we have regime_schedule = ["0", "1", "2"] and each has count 1. 
    # So effectively efficient.
    full_noise, full_power = generate_stitched_noise(total_duration, regime_schedule, regime_block_duration, seed)
    
    # Verify lengths match (accounting for potential small rounding if total_duration vs segments mismatch)
    # asc_plant.tst_noise_t length is based on data['duration_batch'].
    # We should trim or pad full_noise to match asc_plant expectations if needed.
    target_N = asc_plant.tst_noise_t.shape[1]
    
    if full_noise.shape[1] > target_N:
        full_noise = full_noise[:, :target_N]
        full_power = full_power[:target_N]
    elif full_noise.shape[1] < target_N:
        # Pad with zeros? Or just error.
        print(f"Warning: Stitched noise shorter than Plant duration ({full_noise.shape[1]} < {target_N}). Padding.")
        pad_amt = target_N - full_noise.shape[1]
        full_noise = np.pad(full_noise, ((0,0), (0, pad_amt)))
        full_power = np.pad(full_power, (0, pad_amt))
        
    asc_plant.tst_noise_t = full_noise
    asc_plant.input_power = full_power
    
    # 3. INITIALIZE CONTROLLER (Interactive)
    # Start with random arm or C0? Let's ask bandit for first arm.
    arm_idx = bandit.select_arm()
    controller_name = CONTROLLER_CHOICES[arm_idx]
    print(f"Initial Bandit Choice: C{arm_idx}")
    asc_controller = Controller(data, physics, plot_dir=None, controller_name=controller_name)
    
    postprocessing = Postprocessing(data)
    
    # 4. MAIN LOOP
    FS = data['sampling_frequency']
    N = target_N
    
    # Buffers
    tstP_t = np.zeros((N, 2))
    control_t = np.zeros((N, 2))
    strain_noise_t = np.zeros((N, 1))
    # ... other buffers needed for step ...
    readout_t = np.zeros((N, 2))
    SS_compensation_t = np.zeros((N, 2))
    cavity_power_t = np.zeros((N, 1))
    
    asc_plant.reset_counters()
    
    # Helpers for Eigen conversion (for Reward)
    L_c = 3994.5
    R_ITM_c = 1934
    R_ETM_c = 2245
    g1 = 1 - L_c / R_ITM_c
    g2 = 1 - L_c / R_ETM_c
    r_val = 0.5 * (g1 - g2 + np.sqrt((g1 - g2) ** 2 + 4))
    local2eigen = np.array([[1, r_val], [-r_val, 1]]) / (1 + r_val ** 2)

    # Bandit Settings
    SUB_BLOCK_S = 2   # Feature calculation block
    UPDATE_WINDOW_S = bandit_window_duration # Bandit Update Interval (Tunable)
    UPDATE_WINDOW_SAMPLES = int(UPDATE_WINDOW_S * FS)
    
    # Segment Length for Regime Switch (in samples)
    REGIME_STEP_SAMPLES = int(regime_block_duration * FS) 
    
    history = []
    current_regime_idx = 0
    
    print("\nStarting Interactive Loop...")
    
    pbar = tqdm(range(N-1), desc="Simulating")
    for k in pbar:
        
        # A. CHECK REGIME CHANGE
        # regime_schedule is based on SEGMENT_DURATION (32s).
        # k corresponds to time k/FS.
        # current segment index = k // REGIME_STEP_SAMPLES
        seg_idx = k // REGIME_STEP_SAMPLES
        if seg_idx < len(regime_schedule):
            expected_regime = regime_schedule[seg_idx]
            
            # If we just crossed boundary or it's step 0 (already set R0), check if update needed.
            # We track current *active* regime key to detect changes.
            if k > 0 and (k % REGIME_STEP_SAMPLES == 0):
                if expected_regime != start_regime: # Only if diff from 'current' (tracked via var)
                    # Update Plant Parameters!
                    # Note: We already updated the NOISE arrays.
                    # We just need to update P, n_hard, n_soft.
                    
                    r_name = REGIME_FILES[expected_regime][0]
                    p_scale = REGIME_POWER_SCALE.get(r_name, 1.0)
                    s_scale = REGIME_SENSING_SCALE.get(r_name, 1.0)
                    
                    # Update Plant P
                    asc_plant.P = 705.0 * p_scale
                    
                    # Update Sensor Noise
                    asc_sensors.n_hard = 3e-14 * s_scale
                    asc_sensors.n_soft = 1e-13 * s_scale
                    
                    # Log
                    # print(f"[t={k/FS:.1f}s] Switching to Regime {expected_regime} (P={asc_plant.P:.1f}, n_hard={asc_sensors.n_hard:.2e})")

        # B. PROPAGATE
        tstP_t[k+1, :], _, cavity_power_t[k+1, :], _, strain_noise_t[k+1, :], _ = asc_plant.propagate(pum_input_signal=-control_t[k, :], SS_comp = SS_compensation_t[k, :])

        if not np.isfinite(tstP_t[k+1, :]).all():
            print("NaN detected!")
            break

        readout_t[k+1, :] = asc_sensors.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power = cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        
        # C. CONTROLLER STEP
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        
        # D. BANDIT STEP (Every UPDATE_WINDOW_S)
        current_time_step = k + 1
        if current_time_step % UPDATE_WINDOW_SAMPLES == 0:
             # 1. EXTRACT DATA for last Window
             start_idx = current_time_step - UPDATE_WINDOW_SAMPLES
             end_idx = current_time_step
             
             # Need angles_eigen (Hard Error) and control_eigen (Hard Control)
             # Slices
             chunk_tstP = tstP_t[start_idx:end_idx]
             chunk_ctrl = control_t[start_idx:end_idx]
             
             # Convert to Eigen
             chunk_ang_eig = (local2eigen @ chunk_tstP.T).T
             chunk_ctrl_eig = (local2eigen @ chunk_ctrl.T).T
             
             err_h = chunk_ang_eig[:, 1]
             u_h = chunk_ctrl_eig[:, 1]
             
             # 2. COMPUTE SCORES (per 2s sub-block)
             # band_feats_blocks returns features for all 2s blocks within the window
             feats = {}
             feats.update(bandit_rewards.band_feats_blocks(err_h, bandit_rewards.ERR_BANDS))
             feats.update(bandit_rewards.band_feats_blocks(u_h, bandit_rewards.U_BANDS))
             
             # Compute Z for each block
             # Check length
             n_blocks = len(next(iter(feats.values())))
             Z = np.zeros(n_blocks)
             
             for wi, fn in zip(w_weights, feat_names):
                if fn in feats:
                    Z += wi * np.log(feats[fn] + 1e-30)
             
             # Raw Scores per block
             raw_scores = -Z
             
             # 3. AVERAGE RAW SCORES
             mean_raw_score = np.mean(raw_scores)
             
             # Normalize (Sigmoid Mapping on the MEAN)
             # Center = 167.5, Scale = 2.5
             # Maps [165, 170] -> [0.27, 0.73]
             norm_reward = 1.0 / (1.0 + np.exp(-(mean_raw_score - 167.5) / 2.5))
             
             # 3. UPDATE BANDIT
             bandit.update(arm_idx, norm_reward)
             
             # --- LOGGING & HISTORY (Done BEFORE updating arm_idx to new arm) ---
             current_regime = regime_schedule[min(seg_idx, len(regime_schedule)-1)]
             step_idx = current_time_step // UPDATE_WINDOW_SAMPLES
             
             with open("bandit_log.txt", "a") as f:
                f.write(f"{step_idx:04d}    |   R{current_regime}   | C{arm_idx} | {norm_reward:.4f} | {mean_raw_score:.2f} (N={n_blocks})\n")
             
             history.append({
                "step": step_idx,
                "regime": current_regime,
                "chosen_arm": arm_idx,
                "reward": norm_reward,
                "raw_score": mean_raw_score
             })

             pbar.set_postfix({"BanditStep": step_idx, "Regime": current_regime, "Arm": arm_idx, "Rew": f"{norm_reward:.2f}"})

             # 4. SELECT NEW ARM
             new_arm_idx = bandit.select_arm()
             
             # 5. SWITCH CONTROLLER (Hot Swap)
             if new_arm_idx != arm_idx:
                 # Bumpless handover: cross-fade to the new hard controller over 2s (no kick).
                 # (Calling set_feedback_filter_hard directly would cold-reset the loop state.)
                 asc_controller.switch_hard(CONTROLLER_CHOICES[new_arm_idx], ramp_s=2.0)
                 arm_idx = new_arm_idx

             pbar.set_postfix({"BanditStep": step_idx, "Regime": current_regime, "Arm": arm_idx, "Rew": f"{norm_reward:.2f}"})
             
             # Plot Update
             # User asked "Save the histogram once every 30 bandit steps"
             if step_idx % 30 == 0:
                 plot_reward_distributions(history, step_idx, plot_dir)


    print("\nSimulation Complete.")
    # Plot final state regardless of step count
    if history:
        plot_reward_distributions(history, history[-1]["step"], plot_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Interactive Bandit Simulation")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--duration", type=int, default=300, help="Total duration [s]")
    parser.add_argument("--C1", type=float, default=4.0, help="AdSwitch eviction threshold constant C1")
    parser.add_argument("--window", type=int, default=100, help="Bandit reward averaging window [s]")
    args = parser.parse_args()

    master_seed = args.seed
    total_time = args.duration
    c1_param = args.C1
    window_s = args.window
    
    np.random.seed(master_seed)
    
    try:
        feat_names, w_weights = bandit_rewards.load_reward_weights("bandit_reward_weights_rel2base.npz")
    except Exception as e:
        print(f"Failed to load reward weights: {e}")
        return

    num_arms = 3
    horizon = total_time // window_s # Approx horizon
    bandit_log_file = "bandit_log.txt"
    bandit = AdSwitch(num_arms=num_arms, horizon=horizon, C1=c1_param, log_file=bandit_log_file)
    
    # User request: "the regime should change only three times: every 1/3 of the total duration"
    regime_block_duration = total_time // 3
    num_regime_blocks = 3
    
    regime_schedule = ["0", "1", "2"]
    
    # We pass the block duration to the noise generator, so it generates T/3 for each.
    # Note: run_interactive_simulation expects regime_schedule to map to 'segments'?
    # No, our new run_interactive_simulation takes regime_schedule as a list of keys 
    # and calculates noise for each distinct key based on its count? 
    # Let's check generate_stitched_noise.
    
    # To keep it compatible with existing generate_stitched_noise which uses counts * segment_duration,
    # we can just pass [0, 1, 2] and use `regime_block_duration` as the 'segment_duration' arg.
    # But wait, run_interactive_simulation uses SEGMENT_DURATION (32s) for some internal logic?
    # REGIME_STEP_SAMPLES currently uses SEGMENT_DURATION.
    # We must update run_interactive_simulation loop to check regime changes based on `regime_block_duration`.
    # Let's update `run_interactive_simulation` signature or logic.
    
    # It's better to update the run_interactive_simulation to take `regime_block_duration` as an argument.
    
    print(f"Starting Interactive Bandit Simulation: {total_time}s, Seed={master_seed}")
    print(f"Structure: 3 Regime Blocks of {regime_block_duration}s each.")
    print(f"Control: Bandit updates every {window_s}s (Window Average).")
    
    base_plot_dir = "bandit_runs"
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # Initialize Log
    with open(bandit_log_file, "a") as f:
        f.write(f"\n=== New Interactive Run (Seed={master_seed}, Duration={total_time}s, Window={window_s}s) ===\n")
        f.write(f"Step({window_s}s) | Regime | Arm | Reward | Mean Raw Score (N)\n")

    run_interactive_simulation(total_time, regime_schedule, regime_block_duration, master_seed, bandit, feat_names, w_weights, base_plot_dir, bandit_window_duration=window_s)

if __name__ == "__main__":
    main()
