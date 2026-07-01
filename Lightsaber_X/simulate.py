"""Main simulator runner for LIGO Lightsaber."""

import os
import json
import time
import random

from absl import app
from absl import flags
import numpy as np
from Lightsaber import *


flags.DEFINE_integer('fs', 256, 'Sampling frequency [Hz]')
flags.DEFINE_integer('T_batch', 1024, 'Duration of time-series of input noises [s]')
flags.DEFINE_integer('T_fft', 64, 'Duration of FFT segment for plots [s]')
# Plant parameters.
flags.DEFINE_float('P', 705, 'Input power to the arm cavities [W]')  # 705, 200
flags.DEFINE_float('L', 3994.5, 'Arm length [m]')
flags.DEFINE_float('R_ITM', 1934, 'Radius of input test mass [m]')
flags.DEFINE_float('R_ETM', 2245, 'Radius of end test mass [m]')
flags.DEFINE_float('t_ITM', 0.014, 'Power transmission of input test mass')
# Sensor
flags.DEFINE_float('n_hard', 3e-14, 'Spectral density of hard-mode readout noise [rad/rtHz]')
flags.DEFINE_float('n_soft', 1e-13, 'Spectral density of soft-mode readout noise [rad/rtHz]')
# Low-pass filter
flags.DEFINE_float('f_pass', 10, 'Edge of the pass band [Hz]')
flags.DEFINE_float('f_stop', 30, 'Edge of the stop band [Hz]')
flags.DEFINE_float('min_att', 20, 'Minimum attenuation [dB] in stop band of low-pass filter')
# parameters to scale the input noises
flags.DEFINE_float('scale_ITM_ISI_L', 1, 'Parameter to scale seismic noise input for ITM ISI L')
flags.DEFINE_float('scale_ETM_ISI_L', 1, 'Parameter to scale seismic noise input for ETM ISI L')
flags.DEFINE_float('scale_OSEM_L', 1, 'Parameter to scale damping loop noise OSEM L')
flags.DEFINE_float('scale_OSEM_P', 1, 'Parameter to scale damping loop noise OSEM P')
flags.DEFINE_float('scale_RIN', 1, 'Parameter to scale relative input power fluctations')

flags.DEFINE_integer('seed', 1, 'RNG seed for deterministic noise realizations')

flags.DEFINE_string('plot_dir', 'sos_plots', 'Directory to save plots in.')

FLAGS = flags.FLAGS

# Regime-dependent optical power scaling
REGIME_POWER_SCALE = {
    "R0_nominal": 1.0,     # 200 kW
    "R1_high_micro": 0.95,  #  kW
    "R2_high_micro2": 0.9,  #  kW
}

REGIME_SENSING_SCALE = {
    "R0_nominal": 120.0,      # much noisier readout (penalize high gain)
    "R1_high_micro": 12.0,    # moderately noisy readout
    "R2_high_micro2": 0.25,   # clean readout (allow high gain)
}

# Regime-dependent OSEM scaling (applies to BOTH OSEM_L and OSEM_P)
REGIME_OSEM_SCALE = {
    "R0_nominal": 1.0,
    "R1_high_micro": 1.0,
    "R2_high_micro2": 1.0,
}


REGIME_FILES = {
    "0": ("R0_nominal", "noise_inputs/ASD_R0_nominal.csv"),
    "1": ("R1_high_micro", "noise_inputs/ASD_R1_high_micro.csv"),
    "2": ("R2_high_micro2",   "noise_inputs/ASD_R2_high_micro2.csv"),
}

CONTROLLER_CHOICES = {
    "0": "C0_nominal",
    "1": "C1_high_micro",
    "2": "C2_high_micro2",
}


def choose_regime_and_controller():
    print("\n=== Seismic regime selection ===")
    print("  [0] R0_nominal")
    print("  [1] R1_high_micro")
    print("  [2] R2_high_micro2")
    regime_key = input("Choose seismic regime [0–2]: ").strip()
    if regime_key not in REGIME_FILES:
        print("Invalid choice, defaulting to [0] R0_nominal")
        regime_key = "0"

    regime_name, regime_file = REGIME_FILES[regime_key]


    print("\n=== Hard-mode controller selection ===")
    print("  [0] C0_nominal")
    print("  [1] C1_high_micro")
    print("  [2] C2_high_micro2")
    ctrl_key = input("Choose controller [0–2]: ").strip()
    if ctrl_key not in CONTROLLER_CHOICES:
        print("Invalid choice, defaulting to [0] C0_nominal")
        ctrl_key = "0"

    controller_name = CONTROLLER_CHOICES[ctrl_key]

    print(f"\nUsing seismic regime: {regime_name} ({regime_file})")
    print(f"Using hard-mode controller: {controller_name}\n")


    return regime_name, regime_file, controller_name



def main(argv):
    del argv  # Unused.
    # ---- deterministic runs ----
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
    print(f"[simulate.py] Using seed = {FLAGS.seed}")


    #plot_dir = FLAGS.plot_dir
    root_plot_dir = FLAGS.plot_dir  # e.g. 'sos_plots'

    data = {'sampling_frequency': FLAGS.fs, 'duration_batch': FLAGS.T_batch, 'duration_fft': FLAGS.T_fft}
    physics = {'P': FLAGS.P, 'L': FLAGS.L, 'R_ITM': FLAGS.R_ITM, 'R_ETM': FLAGS.R_ETM, 't_ITM': FLAGS.t_ITM}
    sensing = {'noise_hard_mode': FLAGS.n_hard, 'noise_soft_mode': FLAGS.n_soft}
    low_pass = {'pass-band_edge': FLAGS.f_pass, 'stop-band_edge': FLAGS.f_stop, 'minimum_attenuation': FLAGS.min_att}
    parameters = {'scale_ITM_ISI_L': FLAGS.scale_ITM_ISI_L, 'scale_ETM_ISI_L': FLAGS.scale_ETM_ISI_L, 'scale_OSEM_L': FLAGS.scale_OSEM_L, 'scale_OSEM_P': FLAGS.scale_OSEM_P, 'scale_RIN': FLAGS.scale_RIN}
    print(json.dumps(FLAGS.flag_values_dict(), indent=2))

    # Ask user which seismic regime and which controller to use
    regime_name, regime_file, controller_name = choose_regime_and_controller()

    osem_scale = REGIME_OSEM_SCALE.get(regime_name, 1.0)
    print(f"Applying OSEM scale factor: {osem_scale}x for {regime_name}")

    parameters = dict(parameters)  # ensure local copy
    parameters['scale_OSEM_L'] = osem_scale
    parameters['scale_OSEM_P'] = osem_scale


    # -----------------------------
    # Regime-dependent power scaling
    # -----------------------------
    power_scale = REGIME_POWER_SCALE.get(regime_name, 1.0)

    physics = dict(physics)  # make local copy
    physics['P'] = FLAGS.P * power_scale

    print(
        f"Applying optical power for {regime_name}: "
        f"P = {physics['P']:.1f} W "
        f"(scale ×{power_scale})"
    )

    # -----------------------------
    # Regime-dependent seismic scaling
    # -----------------------------
    REGIME_SCALE = {
        "R0_nominal": 1.0,
        "R1_high_micro": 6.0,
        "R2_high_micro2": 20.0,
    }

    seis_scale = REGIME_SCALE.get(regime_name, 1.0)
    print(f"Applying seismic scale factor: {seis_scale}x for {regime_name}")

    # Override ONLY ISI seismic scaling
    parameters = dict(parameters)  # ensure local copy
    parameters['scale_ITM_ISI_L'] = seis_scale
    parameters['scale_ETM_ISI_L'] = seis_scale

    # -----------------------------
    # Regime-dependent sensing noise scaling
    # -----------------------------
    sensing = dict(sensing)  # local copy
    sens_scale = REGIME_SENSING_SCALE.get(regime_name, 1.0)
    sensing['noise_hard_mode'] = FLAGS.n_hard * sens_scale
    sensing['noise_soft_mode'] = FLAGS.n_soft * sens_scale   # optional; can keep soft fixed if you want
    print(
        f"Applying sensing noise scale for {regime_name}: "
        f"n_hard = {sensing['noise_hard_mode']:.3e} (scale ×{sens_scale})"
    )

    # -----------------------------
    # Build per-run plot directory
    # -----------------------------
    run_plot_dir = os.path.join(root_plot_dir, f"{regime_name}__{controller_name}")
    os.makedirs(run_plot_dir, exist_ok=True)
    print(f"Saving all plots to: {run_plot_dir}")


    # Same seismic ASD for ITM and ETM, depending on chosen regime
    noise_files = [
        regime_file,                     # ITM_ISI_L (chosen regime)
        regime_file,                     # ETM_ISI_L (same regime)
        'noise_inputs/n_osem_L.txt',     # OSEM L
        'noise_inputs/n_osem_P.txt',     # OSEM P
        'noise_inputs/O3_power_psd.csv'  # RIN / power PSD
    ]
    transfer_files = ['transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topL_2_tstP.txt', 'transfer_functions/tf_topNL_2_tstP.txt', 'transfer_functions/tf_topNP_2_tstP.txt']
    reference_data_file = 'noise_inputs/aicReferenceData_Aplus.txt'


    asc_plant = Plant(
        physics, data, parameters,
        plot_dir=run_plot_dir,
        noise_files=noise_files,
        reference_data_file=reference_data_file,
        transfer_files=transfer_files,
        seed=FLAGS.seed,
    )
    asc_sensors = Sensors(sensing, physics, data, seed=FLAGS.seed)
    asc_SS_compensation = SS_compensation(data, physics, asc_plant, plot_dir=run_plot_dir)
    asc_controller = Controller(data, physics, plot_dir=run_plot_dir,
                                controller_name=controller_name)
    low_pass_filter = FilterLP(data, low_pass)
    postprocessing = Postprocessing(data)


    sim_open_loop = False
    sim_closed_loop = True

    # open-loop simulation
    if sim_open_loop:
        open_loop_run(asc_plant, asc_sensors, asc_SS_compensation, asc_controller, low_pass_filter, data, run_plot_dir)

    # closed-loop simulation
    if sim_closed_loop:
        closed_loop_run(asc_plant, asc_sensors, asc_SS_compensation, asc_controller, postprocessing, data, run_plot_dir, reference_data_file)


if __name__ == '__main__':
    start_time = time.time()
    app.run(main)
    print("--- %s seconds ---" % np.round(time.time() - start_time))
