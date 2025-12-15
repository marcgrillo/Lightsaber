"""
Script to collect raw physics metrics (Total RMS, Bump RMS) for all 9 combinations.
Used to tune Alpha/Beta weights.
"""

import os
import sys
import copy
import numpy as np
import yaml
import tqdm
from Lightsaber import *
import utils
import json

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

def get_metrics(pitch_timeseries, fs):
    t_fft = 64 
    ff, psd, rms_total = utils.compute_psd(pitch_timeseries, t_fft, fs)
    
    # Total RMS (0.1 - 100 Hz)
    f_total_idx = np.logical_and(ff >= 0.1, ff <= 100)
    rms_total_calc = np.sqrt(np.sum(psd[f_total_idx]) * (ff[1]-ff[0]))
    
    # Bump RMS (4.5 - 7.5 Hz)
    f_bump_idx = np.logical_and(ff >= 4.5, ff <= 7.5)
    rms_bump = np.sqrt(np.sum(psd[f_bump_idx]) * (ff[1]-ff[0]))
    
    return rms_total_calc, rms_bump

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

def measure_readout_rms(fs):
    # Load Sensor Noise file to compare magnitude
    try:
        data = np.genfromtxt('noise_inputs/SENSOR_PITCH_HARD.csv', delimiter=',')
        # Approximate RMS from PSD
        import scipy.integrate
        rms = np.sqrt(scipy.integrate.simpson(data[:,1]**2, data[:,0]))
        print(f"DEBUG: Approximate Sensor Noise RMS: {rms:.2e}")
        return rms
    except:
        print("DEBUG: Could not measure sensor RMS")
        return 0

def main():
    with open('configuration/config.yaml') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
        
    measure_readout_rms(256) # Debug check

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
        
    # FORCE SPEED UP
    components_config['ITM']['duration_batch'] = 64
    components_config['ETM']['duration_batch'] = 64
    components_config['PITCH_CONTROL']['duration_batch'] = 64
    simulation['duration_batch'] = 64 

    print("Collecting physics data...")
    # Rows: Regimes, Cols: Controllers
    raw_data = {} 

    # We need robust stats, so run N times
    N_runs = 5

    for r_idx, r_data in REGIMES.items():
        raw_data[r_idx] = {}
        regime_file = os.path.join('noise_inputs', r_data['file'])
        
        for c_idx, c_name in CONTROLLERS.items():
            
            # Setup Config
            current_config = copy.deepcopy(components_config)
            current_config['ITM']['sus_noise'][0][0] = regime_file
            current_config['ETM']['sus_noise'][0][0] = regime_file
            current_config['PITCH_CONTROL']['controller_name'] = c_name
            
            t_vals = []
            b_vals = []

            for i in range(N_runs):
                simulation['seed_for_random'] = 42 + i
                system = create_system(current_config, simulation, plot_dir=False)
                link_components(system)
                results = run(system, simulation)
                
                pitch_data = results[0][0]
                
                # Convert to Eigenmode
                beam = system.get_by_type('Beam')[0]
                local_to_eigen = beam.local_to_eigen
                eigen_data = pitch_data @ local_to_eigen.T
                hard_pitch = eigen_data[:, 1]
                
                itm_pitch = hard_pitch
                fs = eval(str(simulation['simulation_sampling_frequency']))
                
                t, b = get_metrics(itm_pitch, fs)
                t_vals.append(t)
                b_vals.append(b)
            
            # Avg
            mean_t = np.mean(t_vals)
            mean_b = np.mean(b_vals)
            
            raw_data[r_idx][c_idx] = {'T': mean_t, 'B': mean_b}
            print(f"R{r_idx}_{r_data['name']} | C{c_idx}_{c_name} -> T={mean_t:.2e}, B={mean_b:.2e}")

    print("\n--- RESULTS PYTHON DICT ---")
    print(raw_data)
    
    with open('physics_results.json', 'w') as f:
        json.dump(raw_data, f, indent=4)
        print("Results saved to physics_results.json")

if __name__ == '__main__':
    main()
