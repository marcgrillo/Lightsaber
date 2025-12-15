"""Verification script to check Ground Truth controller performance."""

import os
import sys
import copy
import numpy as np
import yaml
import tqdm
import matplotlib.pyplot as plt

from Lightsaber import *
import utils
import plotting

# Reuse configuration logic/constants from simulate.py (conceptually)
# To avoid circular imports or refactoring, I'll copy the necessary setup code.

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

# --- Reward Parameters (MUST MATCH simulate.py) ---
# Tuned Weights for Diagonal Optimality
ALPHA_RMS = 1.0e9
BETA_BUMP = 1.0e10

def calculate_reward(pitch_timeseries, fs, offset):
    t_fft = 64 
    ff, psd, rms_total = utils.compute_psd(pitch_timeseries, t_fft, fs)
    
    f_total_idx = np.logical_and(ff >= 0.1, ff <= 100)
    rms_total_calc = np.sqrt(np.sum(psd[f_total_idx]) * (ff[1]-ff[0]))
    
    # Bump RMS (4.5 - 7.5 Hz)
    f_bump_idx = np.logical_and(ff >= 4.5, ff <= 7.5)
    rms_bump = np.sqrt(np.sum(psd[f_bump_idx]) * (ff[1]-ff[0]))
    
    cost = ALPHA_RMS * rms_total_calc + BETA_BUMP * rms_bump
    norm_cost = 1 / ( 1 + np.exp(-cost + offset) )
    norm_rew = 1 - norm_cost
    return norm_rew, cost, rms_total_calc, rms_bump

def create_system(config, simulation, plot_dir=False):
    seed = eval(str(simulation['seed_for_random']))
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
    output = {}
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
        
    # FORCE SPEED UP
    components_config['ITM']['duration_batch'] = 1024
    components_config['ETM']['duration_batch'] = 1024
    components_config['PITCH_CONTROL']['duration_batch'] = 1024
    simulation['duration_batch'] = 1024 

    print("\nStarting Exhaustive Verification (3 Regimes x 3 Controllers)...")
    print(f"{'Regime':<20} | {'Controller':<20} | {'Reward':<10} | {'RMS':<10} | {'BUMP':<10}")
    print("-" * 80)

    results_matrix = np.zeros((3, 3)) # Rows: Regime, Cols: Controller

    for r_idx, r_data in REGIMES.items():
        regime_file = os.path.join('noise_inputs', r_data['file'])
        
        for c_idx, c_name in CONTROLLERS.items():
            
            # Setup Config
            current_config = copy.deepcopy(components_config)
            current_config['ITM']['sus_noise'][0][0] = regime_file
            current_config['ETM']['sus_noise'][0][0] = regime_file
            current_config['PITCH_CONTROL']['controller_name'] = c_name
            
            # Run multiple iterations to estimate variance
            N_runs = 3
            rewards = []
            costs = []
            rms_tots = []
            rms_bumps = []
            offset = 2
            
            for i in range(N_runs):
                simulation['seed_for_random'] = 42 + i
                
                system = create_system(current_config, simulation, plot_dir=False)
                link_components(system)
                results_single = run(system, simulation)
                
                pitch_data_s = results_single[0][0]
                #print(pitch_data_s.shape)
                
                # Convert to Eigenmode (Soft/Hard)
                beam = system.get_by_type('Beam')[0]
                local_to_eigen = beam.local_to_eigen
                eigen_data_s = pitch_data_s @ local_to_eigen.T

                # Select Hard Component (Index 1)
                hard_pitch_s = eigen_data_s[:, 1]
                
                itm_pitch_s = hard_pitch_s 
                fs_s = eval(str(simulation['simulation_sampling_frequency']))
                
                # These are the raw physical values (Radians)
                reward, cost, val_rms_tot, val_rms_bump = calculate_reward(itm_pitch_s, fs_s, offset)
        
                
                rewards.append(reward)
                costs.append(cost)
                rms_tots.append(val_rms_tot)
                rms_bumps.append(val_rms_bump)
            
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_cost = np.mean(costs)
            mean_rms = np.mean(rms_tots)
            mean_bump = np.mean(rms_bumps)
            
            results_matrix[r_idx, c_idx] = mean_reward
            
            # Print Stats with RAW RMS
            print(f"{r_data['name']:<15} | {c_name:<15} | Rw: {mean_reward:.4f} | Cost: {mean_cost:.2e} | RMS: {mean_rms:.2e} | BUMP: {mean_bump:.2e}")

    print("\nSummary Matrix (Rows: R0-R2, Cols: C0-C2):")
    print(results_matrix)
    
    print("\nOptimality Check:")
    # Partial Optimality Expectations: R0->C1, R1->C1, R2->C2
    expected_map = {0: 1, 1: 1, 2: 2}
    
    for r in range(3):
        best_c = np.argmax(results_matrix[r, :])
        expected_c = expected_map[r]
        is_correct = (best_c == expected_c)
        star = "*" if is_correct else "X"
        print(f"Regime {REGIMES[r]['name']}: Best Controller is {CONTROLLERS[best_c]} (Expected {CONTROLLERS[expected_c]}) [{star}]")

if __name__ == '__main__':
    main()
