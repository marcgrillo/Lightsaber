import os
import sys
import copy
import numpy as np
import yaml
import matplotlib.pyplot as plt
from Lightsaber import *
import utils

CONTROLLERS = {
    0: 'C0_nominal',
    1: 'C1_high_micro',
    2: 'C2_high_micro2'
}

REGIMES = {
    2: {'name': 'R2_high_micro2', 'file': 'ASD_R2_tuned.csv'}
}

def create_system(config, simulation, plot_dir=False):
    seed = 42
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
        
    components_config['ITM']['duration_batch'] = 64
    components_config['ETM']['duration_batch'] = 64
    components_config['PITCH_CONTROL']['duration_batch'] = 64
    simulation['duration_batch'] = 64 

    # Target Regime: R2 (The most chaotic one)
    TARGET_REGIME_IDX = 2 
    regime_info = REGIMES[TARGET_REGIME_IDX]
    regime_file = os.path.join('noise_inputs', regime_info['file'])
    
    print(f"Diagnosing Spectra for Regime: {regime_info['name']}")
    print(f"File: {regime_file}")
    
    psds = {}
    
    # Update Regime Config
    components_config['ITM']['sus_noise'][0][0] = regime_file
    components_config['ETM']['sus_noise'][0][0] = regime_file
    
    for c_idx, c_name in CONTROLLERS.items():
        print(f"Running {c_name}...")
        current_config = copy.deepcopy(components_config)
        current_config['PITCH_CONTROL']['controller_name'] = c_name
        
        system = create_system(current_config, simulation)
        link_components(system)
        
        # Run
        results = run(system, simulation)
        pitch_data = results[0][0]
        itm_pitch = pitch_data[:, 0]
        fs = eval(str(simulation['simulation_sampling_frequency']))
        
        ff, psd, rms = utils.compute_psd(itm_pitch, 64, fs)
        psds[c_idx] = (ff, psd)

    # Plot
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'red']
    for c_idx, (ff, psd) in psds.items():
        c_name = CONTROLLERS[c_idx]
        plt.loglog(ff, psd, label=c_name, color=colors[c_idx], linewidth=1.5 if c_idx==0 else 1.0)
        
    plt.title(f"Output Pitch Spectra in {regime_info['name']}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [rad^2/Hz]")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('diagnosis_R2_spectra.png')
    print("Plot saved to diagnosis_R2_spectra.png")

if __name__ == '__main__':
    main()
