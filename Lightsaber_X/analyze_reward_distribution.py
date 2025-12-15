"""
Script to analyze the distribution of rewards for each controller in a FIXED regime.
Demonstrates the variance due to stochastic noise.
"""

import os
import sys
import copy
import numpy as np
import yaml
import tqdm
import matplotlib.pyplot as plt
from Lightsaber import *
import utils

# --- COPIED FROM SIMULATE.PY ---
# Tuned Weights for Diagonal Optimality
ALPHA_RMS = 1.0e9
BETA_BUMP = 1.0e10

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

def calculate_reward(pitch_timeseries, fs):
    t_fft = 64 
    ff, psd, rms_total = utils.compute_psd(pitch_timeseries, t_fft, fs)
    
    # Total RMS (0.1 - 100 Hz)
    f_total_idx = np.logical_and(ff >= 0.1, ff <= 100)
    rms_total_calc = np.sqrt(np.sum(psd[f_total_idx]) * (ff[1]-ff[0]))
    
    # Bump RMS (4.5 - 7.5 Hz)
    f_bump_idx = np.logical_and(ff >= 4.5, ff <= 7.5)
    rms_bump = np.sqrt(np.sum(psd[f_bump_idx]) * (ff[1]-ff[0]))
    
    # Cost
    cost = ALPHA_RMS * rms_total_calc + BETA_BUMP * rms_bump
    # Sigmoid Reward
    reward = 1.0 / (1.0 + cost)
    
    return reward

def create_system(config, simulation, plot_dir=False):
    # Use None for seed to get randomness
    seed = None 
    system = System()
    for comp_name in config.keys():
        comp_config = config[comp_name]
        comp_config['name'] = comp_name
        comp_config['simulation_sampling_frequency'] = simulation['simulation_sampling_frequency']
        comp_config['duration_batch'] = simulation['duration_batch']
        comp_config['duration_fft'] = simulation['duration_fft']
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

def main():
    print("Loading Configuration...")
    with open('configuration/config.yaml') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    simulation = {}
    components_config = copy.deepcopy(base_config)
    keys_to_delete = []
    
    for key in base_config.keys():
        if base_config[key]['type'] == 'Simulation':
            simulation = base_config[key]
            keys_to_delete.append(key)
        elif base_config[key]['type'] == 'Output':
            keys_to_delete.append(key)
            
    for k in keys_to_delete:
        del components_config[k]
        
    # SPEED UP
    components_config['ITM']['duration_batch'] = 64
    components_config['ETM']['duration_batch'] = 64
    components_config['PITCH_CONTROL']['duration_batch'] = 64
    simulation['duration_batch'] = 64 

    # --- PARAMETERS ---
    # Target Regime: R2 (The most chaotic one)
    TARGET_REGIME_IDX = 2 
    regime_info = REGIMES[TARGET_REGIME_IDX]
    regime_file = os.path.join('noise_inputs', regime_info['file'])
    
    print(f"Analyzing Reward Distribution for Regime: {regime_info['name']}")
    print(f"File: {regime_file}")
    
    N_SAMPLES = 50
    print(f"Running {N_SAMPLES} iterations per controller...")
    
    results = {0: [], 1: [], 2: []}
    
    # Update Regime Config ONCE
    components_config['ITM']['sus_noise'][0][0] = regime_file
    components_config['ETM']['sus_noise'][0][0] = regime_file
    
    for c_idx, c_name in CONTROLLERS.items():
        print(f"Sampling Controller {c_name}...")
        
        # Update Controller
        current_config = copy.deepcopy(components_config)
        current_config['PITCH_CONTROL']['controller_name'] = c_name
        
        for _ in tqdm.tqdm(range(N_SAMPLES)):
            system = create_system(current_config, simulation)
            link_components(system)
            run_output = run(system, simulation)
            
            pitch_data = run_output[0][0]
            
            # Convert to Eigenmode
            beam = system.get_by_type('Beam')[0]
            local_to_eigen = beam.local_to_eigen
            eigen_data = pitch_data @ local_to_eigen.T
            hard_pitch = eigen_data[:, 1]
            
            itm_pitch = hard_pitch
            fs = eval(str(simulation['simulation_sampling_frequency']))
            
            r = calculate_reward(itm_pitch, fs)
            results[c_idx].append(r)
            
    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green']
    for c_idx, rewards in results.items():
        c_name = CONTROLLERS[c_idx]
        plt.hist(rewards, bins=15, alpha=0.5, label=f"{c_name} (Mean: {np.mean(rewards):.4f})", color=colors[c_idx])
        plt.axvline(np.mean(rewards), color=colors[c_idx], linestyle='--', linewidth=2)
        
    plt.title(f"Reward Distribution in {regime_info['name']} (N={N_SAMPLES})")
    plt.xlabel("Reward (Sigmoid)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = 'data_output/reward_distribution.png'
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    
    # Print Stats
    print("\n--- STATISTICS ---")
    for c_idx, rewards in results.items():
        mean = np.mean(rewards)
        std = np.std(rewards)
        print(f"{CONTROLLERS[c_idx]}: Mean={mean:.4f}, Std={std:.4f} => CV={std/mean:.1%}")

if __name__ == '__main__':
    main()
