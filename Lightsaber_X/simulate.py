"""Main simulator runner for LIGO Lightsaber with Bandit Controller Switching."""

import json
import os
import time
import pathlib
import sys
import copy

from absl import app

import numpy as np
import yaml
import tqdm
import matplotlib.pyplot as plt

from Lightsaber import *
from adswitch import AdSwitch
import utils
import plotting

import pathlib
from typing import Any, Dict, Optional, Union

PathLike = Union[str, pathlib.Path]

# --- Configuration Constants ---
NUM_STEPS = 10 # Total simulation steps (batches)
REGIME_SWITCH_PROB = 0.2 
SWITCH_INTERVAL = 50 

REGIMES = {
    0: {'name': 'R0_nominal', 'file': 'ASD_R0_nominal.csv'},
    1: {'name': 'R1_high_micro', 'file': 'ASD_R1_high_micro.csv'},
    2: {'name': 'R2_high_micro2', 'file': 'ASD_R2_high_micro2.csv'}
}

CONTROLLERS = {
    0: 'C0_nominal',
    1: 'C1_high_micro',
    2: 'C2_high_micro2'
}

# --- Reward Parameters ---
# Tuned Weights for Diagonal Optimality (R0->C0, R1->C1, R2->C2)
ALPHA_RMS = 1.0e9
BETA_BUMP = 1.0e10

def calculate_reward(pitch_timeseries, fs, offset):
    """
    Calculates reward based on Total RMS and Bump RMS (4.5-7.5 Hz).
    Reward = - (alpha * RMS_total + beta * RMS_bump)
    """
    # Use utils.compute_psd to get PSD
    # Note: utils.compute_psd computes RMS for whole valid range (usually defined by window)
    # We will use the PSD and frequencies to compute specific bands.
    
    t_fft = 64 # From config, reasonable default
    ff, psd, rms_total = utils.compute_psd(pitch_timeseries, t_fft, fs)
    
    # Total RMS (0.1 - 100 Hz) - already returned as rms_total by compute_psd usually if filtered?
    # Actually utils.compute_psd returns sqrt(integral(PSD)), but let's re-integrate for safety over specific band
    
    f_total_idx = np.logical_and(ff >= 0.1, ff <= 100)
    rms_total_calc = np.sqrt(np.sum(psd[f_total_idx]) * (ff[1]-ff[0]))
    
    # Bump RMS (4.5 - 7.5 Hz)
    f_bump_idx = np.logical_and(ff >= 4.5, ff <= 7.5)
    rms_bump = np.sqrt(np.sum(psd[f_bump_idx]) * (ff[1]-ff[0]))
    
    cost = ALPHA_RMS * rms_total_calc + BETA_BUMP * rms_bump
    norm_cost = 1 / ( 1 + np.exp(-cost + offset) )
    norm_rew = 1 - norm_cost
    return norm_rew, cost, rms_total_calc, rms_bump

def save_results_as_txt(results, simulation, directory, step, regime_name, controller_name):
    """
    Saves the results as txt files and plots of the spectral densities of the time series.
    Modified to include step info in filename.
    """

    fs = eval(str(simulation['simulation_sampling_frequency']))
    t_fft = eval(str(simulation['duration_fft']))

    directory = pathlib.Path(directory)
    # Only save every few steps to avoid spamming disk? Or overwrite? 
    # Let's save specific interesting steps or just overwrite 'current'.
    # For now, save with step index.
    
    step_dir = directory / f"step_{step}_{regime_name}_{controller_name}"
    os.makedirs(step_dir, exist_ok=True)
    
    for result in results:
        # result: [data, unit, name, title]
        filename = os.path.join(step_dir, f'{result[2]}.csv')
        np.savetxt(filename, result[0], delimiter=' ')

        filename = os.path.join(step_dir, f'{result[2]}.png')
        plotting.plot_psd(result[0], t_fft, fs, filename, ylabel=result[3]+' ['+result[1]+'/Hz$^{1/2}$]')


def create_system(config, simulation, plot_dir=False):
    """
    Creates the system from config.
    """
    seed = eval(str(simulation['seed_for_random']))

    system = System()

    # We need to ensure config is clean/deep copied if reused, but here we pass a fresh modifies config each time
    for comp_name in config.keys():
        comp_config = config[comp_name]
        comp_config['name'] = comp_name
        comp_config['simulation_sampling_frequency'] = simulation['simulation_sampling_frequency']
        comp_config['duration_batch'] = simulation['duration_batch']
        comp_config['duration_fft'] = simulation['duration_fft']
        
        # Instantiate component
        # Note: eval() is used to get the class from string name
        # We need to make sure classes are in scope (from Lightsaber import *)
        system.append(eval(comp_config['type'])(comp_config, seed, plot_dir))

    for beam in system.get_by_type('Beam'):
        mirrors = system.get_by_type('Mirror')
        laser = system.get_by_type('Laser')
        beam.set_parameters(mirrors[0], mirrors[1], laser[0].wavelength)

    return system


def link_components(system):
    for comp in system.components:
        for comp_to_be_linked in system.components:
            comp_to_be_linked.substitute_names_by_variables(comp)


def main(argv):
    del argv  # Unused.

    # 1. Load Base Config
    with open('configuration/config.yaml') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    simulation = {}
    output = {}
    
    # Extract Simulation and Output blocks to clean up component config
    components_config = copy.deepcopy(base_config)
    keys_to_delete = []
    
    for key in base_config.keys():
        if base_config[key]['type'] == 'Simulation':
            simulation = base_config[key]
            keys_to_delete.append(key)
        elif base_config[key]['type'] == 'Output':
            output = base_config[key]
            keys_to_delete.append(key)
            
    for k in keys_to_delete:
        del components_config[k]
        
    # Ensure output directory exists
    os.makedirs(output['out_directory'], exist_ok=True)

    # SPEED UP: Override duration for faster simulation (64s instead of 1024s)
    # This allows checking long-term behavior in reasonable time.
    components_config['ITM']['duration_batch'] = 128
    components_config['ETM']['duration_batch'] = 128
    components_config['PITCH_CONTROL']['duration_batch'] = 128
    simulation['duration_batch'] = 128 

    # 2. Initialize Bandit
    # 3 arms (C0, C1, C2)
    # Horizon = NUM_STEPS (batches)
    bandit = AdSwitch(num_arms=3, horizon=NUM_STEPS, C1=4.0)

    # Tracking Data
    history = {
        'step': [],
        'regime': [],
        'controller': [],
        'reward': [],
        'rms_total': [],
        'rms_bump': []
    }
    
    # NEW: Bandit Internal Stats Tracking
    bandit_history = {
        'step': [],
        'episode_start': [],
        'arm0_mean': [], 'arm0_n': [], 'arm0_good': [],
        'arm1_mean': [], 'arm1_n': [], 'arm1_good': [],
        'arm2_mean': [], 'arm2_n': [], 'arm2_good': []
    }
    
    # Optimal mapping for Regret calculation
    # R0 -> C0, R1 -> C1, R2 -> C2
    OPTIMAL_CONTROLLER = {0: 0, 1: 1, 2: 2}

    current_regime_idx = 0
    
    print(f"Starting Simulation Loop: {NUM_STEPS} steps")
    
    for step in tqdm.tqdm(range(NUM_STEPS)):
        
        # --- A. Regime Switching ---
        # Switch every SWITCH_INTERVAL steps, or randomly
        if step % SWITCH_INTERVAL == 0:
            current_regime_idx = np.random.choice(list(REGIMES.keys()))
            # print(f"Step {step}: Switched to Regime {REGIMES[current_regime_idx]['name']}")
            
        regime_info = REGIMES[current_regime_idx]
        regime_file = os.path.join('noise_inputs', regime_info['file'])
        
        # --- B. Bandit Selection ---
        selected_arm = bandit.select_arm()
        controller_name = CONTROLLERS[selected_arm]
        
        # --- C. Update Configuration ---
        # Create a step-specific config
        current_config = copy.deepcopy(components_config)
        
        # Update Input Noise based on Regime (ITM and ETM sus_noise)
        # Config structure: ITM -> sus_noise -> list of [file, tf]
        # We assume the first element of sus_noise is the seismic input we want to swap
        # "noise_inputs/ITM_SEI_LIGO_O3.csv" is usually at index 0
        
        # Update ITM
        current_config['ITM']['sus_noise'][0][0] = regime_file
        # Update ETM (assuming symmetric input or same file)
        current_config['ETM']['sus_noise'][0][0] = regime_file
        
        # Update Controller
        current_config['PITCH_CONTROL']['controller_name'] = controller_name
        
        # --- D. Create and Run System ---
        system = create_system(current_config, simulation, plot_dir=False)
        link_components(system)
        
        results = run(system, simulation)
        
        # --- E. Calculate Reward ---
        # results structure: 
        # 0: [pitch_tt, "rad", "pitch", "Pitch (ITM/ETM)"]
        # pitch_tt is (N, 2). Index 0 is ITM.
        
        pitch_data = results[0][0]
        
        # Convert to Eigenmode (Soft/Hard)
        beam = system.get_by_type('Beam')[0]
        local_to_eigen = beam.local_to_eigen
        
        # pitch_data is (N, 2). Transform to (N, 2) eigen.
        # eigen = local @ matrix.T
        eigen_data = pitch_data @ local_to_eigen.T
        
        # Select Hard Component (Index 1)
        hard_pitch = eigen_data[:, 1]
        
        # --- F. Update Bandit ---
        # Use simple RMS reciprocal or similar
        itm_pitch = hard_pitch # Variable name reuse for minimal diff, or rename to pitch_for_reward
        fs = eval(str(simulation['simulation_sampling_frequency']))
        
        reward, cost, rms_tot, rms_bump = calculate_reward(itm_pitch, fs, offset=0)
        #print(f"Orig. Reward: {reward}")
        

        bandit.update(selected_arm, reward)
        
        # Log to console for user visibility
        tqdm.tqdm.write(f"Step {step:03d}: {regime_info['name']:<15} | Sel: {controller_name:<15} | Rw: {reward:.4f} | Cost: {cost:.2e} | RMS: {rms_tot:.2e} | BUMP: {rms_bump:.2e}")
        
        # --- G. Log ---
        optimal_arm = OPTIMAL_CONTROLLER[current_regime_idx]
        # regret = optimal_reward - actual_reward (Hard to know optimal reward without running optimal)
        # We will track just correctness for now
        
        history['step'].append(step)
        history['regime'].append(regime_info['name'])
        history['controller'].append(controller_name)
        history['reward'].append(reward)
        history['rms_total'].append(rms_tot)
        history['rms_bump'].append(rms_bump)
        
        # --- H. Log Bandit Stats ---
        bandit_history['step'].append(step)
        bandit_history['episode_start'].append(bandit.t_l)
        
        for arm_idx in range(3):
            # Get stats for current episode [t_l, t]
            mu, n = bandit._get_stats(arm_idx, bandit.t_l, bandit.t)
            is_good = arm_idx in bandit.good_arms
            
            bandit_history[f'arm{arm_idx}_mean'].append(mu)
            bandit_history[f'arm{arm_idx}_n'].append(n)
            bandit_history[f'arm{arm_idx}_good'].append(is_good)

        # Save last step detailed data
        if step == NUM_STEPS - 1:
            save_results_as_txt(results, simulation, output['out_directory'], step, regime_info['name'], controller_name)

    # --- Plotting Summary ---
    plot_summary(history)
    save_bandit_stats(bandit_history)

def plot_summary(history):
    steps = history['step']
    rewards = history['reward']
    regimes = history['regime']
    controllers = history['controller']
    
    # Convert string labels to integers for plotting
    regime_map = {name: i for i, name in enumerate(['R0_nominal', 'R1_high_micro', 'R2_high_micro2'])}
    controller_map = {name: i for i, name in enumerate(['C0_nominal', 'C1_high_micro', 'C2_high_micro2'])}
    
    reg_vals = [regime_map[r] for r in regimes]
    ctrl_vals = [controller_map[c] for c in controllers]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Regimes vs Controllers
    ax1.plot(steps, reg_vals, 'k--', label='Regime', alpha=0.5)
    ax1.scatter(steps, ctrl_vals, c=rewards, cmap='viridis', label='Controller Choice')
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Nominal', 'HighMicro', 'HighMicro2'])
    ax1.set_ylabel('Mode')
    ax1.legend()
    ax1.set_title('Regime vs Controller Selection')
    
    # Plot 2: Rewards
    ax2.plot(steps, rewards, 'b-', label='Reward')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Evolution')
    
    plt.tight_layout()
    plt.savefig('data_output/simulation_summary.png')
    print("Simulation Complete. Summary plot saved to data_output/simulation_summary.png")

    save_stats_and_plots(history)

def save_bandit_stats(bandit_history):
    """
    Saves and plots the internal belief state of the bandit.
    """
    import pandas as pd
    
    df = pd.DataFrame(bandit_history)
    stats_file = 'data_output/bandit_internal_stats.csv'
    df.to_csv(stats_file, index=False)
    print(f"Bandit internal stats saved to {stats_file}")
    
    # Plot Estimated Means
    plt.figure(figsize=(12, 6))
    steps = bandit_history['step']
    
    colors = ['blue', 'orange', 'green']
    for arm in range(3):
        means = bandit_history[f'arm{arm}_mean']
        is_good = bandit_history[f'arm{arm}_good']
        
        # Plot mean trace
        plt.plot(steps, means, label=f'Arm {arm} Mean', color=colors[arm], alpha=0.7)
        
        # Scatter for Good/Bad status (X if bad)
        # We find indices where arm was BAD
        bad_indices = [i for i, good in enumerate(is_good) if not good]
        if bad_indices:
            plt.scatter([steps[i] for i in bad_indices], [means[i] for i in bad_indices], 
                        marker='x', color=colors[arm], s=20)

    plt.title("Bandit Estimated Means (Cross 'x' = Evicted/Bad Arm)")
    plt.xlabel("Step")
    plt.ylabel("Estimated Mean Reward (Current Episode)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data_output/bandit_estimated_means.png')
    print("Bandit means plot saved to data_output/bandit_estimated_means.png")

def save_stats_and_plots(history):
    """
    Groups rewards by (Regime, Controller) and saves/plots statistics.
    """
    import pandas as pd
    import seaborn as sns
    
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(history)
    
    # 1. Save Statistics to Text File
    stats_file = 'data_output/simulation_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("--- Simulation Statistics (Grouped by Regime and Controller) ---\n\n")
        
        groups = df.groupby(['regime', 'controller'])
        for (regime, controller), group in groups:
            rewards = group['reward']
            cnt = len(rewards)
            avg = rewards.mean()
            std = rewards.std()
            f.write(f"Regime: {regime:<20} | Controller: {controller:<20} | N={cnt:3d} | Mean Rw: {avg:.4f} | Std: {std:.4f}\n")
    
    print(f"Statistics saved to {stats_file}")

    # 2. Histogram Plots
    # Create a figure with one subplot per Regime
    unique_regimes = sorted(df['regime'].unique())
    n_regimes = len(unique_regimes)
    
    if n_regimes > 0:
        fig, axes = plt.subplots(n_regimes, 1, figsize=(10, 5 * n_regimes), squeeze=False)
        axes = axes.flatten()
        
        for idx, regime in enumerate(unique_regimes):
            ax = axes[idx]
            subset = df[df['regime'] == regime]
            
            # Plot histograms for each controller in this regime
            # Using simple loop to control colors/alpha
            for controller in sorted(subset['controller'].unique()):
                c_data = subset[subset['controller'] == controller]['reward']
                ax.hist(c_data, bins=15, alpha=0.5, label=f"{controller} (N={len(c_data)})")
            
            ax.set_title(f"Reward Distribution in {regime}")
            ax.set_xlabel("Reward")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        hist_file = 'data_output/simulation_histograms.png'
        plt.savefig(hist_file)
        print(f"Histograms saved to {hist_file}")

if __name__ == '__main__':
    start_time = time.time()
    app.run(main)
    print("--- %s seconds ---" % np.round(time.time() - start_time))
