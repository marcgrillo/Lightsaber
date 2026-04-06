import os
import yaml
import numpy as np
import time
from absl import app, flags
from Lightsaber import System, Laser, Mirror, Beam, Sensor, Controller
from algorithms import DPSKF, FixedBandit
from noise_models import SinusoidalPoissonEnv, StreamableNoise
from physics_fast import run_fast_physics_kernel
from plotting import plot_bandit_log
from reward_utils import AdvancedRewardCalculator
from ss_compensation import SS_compensation
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configuration/config.yaml', 'Path to config file')
flags.DEFINE_string('output_dir', 'data_output/bandit_test', 'Output directory')
flags.DEFINE_integer('duration', 43200, 'Total simulation duration in seconds (12h = 43200)')
flags.DEFINE_integer('batch_size', 1024, 'Processing batch size in samples')
flags.DEFINE_integer('bandit_interval', 300, 'Interval between bandit decisions in seconds (B)')
flags.DEFINE_string('bandit_type', 'DPSKF', 'Type of bandit to use: DPSKF, Fixed0, Fixed1, Fixed2')
flags.DEFINE_string('noise_cache', '', 'Path to a .npz file to load/save environmental noise arrays (e.g. data/noise_cache.npz)')

# Environmental Hyperparameters (SinusoidalPoissonEnv)
flags.DEFINE_float('mu', 30.0, 'Environmental non-stationarity frequency')
flags.DEFINE_float('std_dev', 3600.0, 'Standard deviation for drift events')
flags.DEFINE_float('trans_mean', 60.0, 'Mean duration of transitions')
flags.DEFINE_float('exponent', 0.33, 'Exponent for sinusoidal shape')
flags.DEFINE_float('r1_rate', 3.0, 'Rate of Poisson spikes (Regime 1)')
flags.DEFINE_float('r1_attack', 60.0, 'Attack duration for Poisson events')
flags.DEFINE_float('r1_hold', 1200.0, 'Hold duration for Poisson events')
flags.DEFINE_float('r1_decay', 60.0, 'Decay duration for Poisson events')

# Bandit Hyperparameters (DPSKF) - Restored to User Optimal Values
flags.DEFINE_float('alpha', 1e-05, 'Dirichlet Process concentration (DPa)')
flags.DEFINE_integer('W', 300, 'History window for likelihood evaluation')
flags.DEFINE_float('Q', 0.1, 'Kalman filter process noise')
flags.DEFINE_float('R', 0.04, 'Kalman filter measurement noise')
flags.DEFINE_string('weights_path', 'data/bandit_reward_weights_rel2base.npz', 'Path to reward weights')

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_system_compat(config, plot_dir=False):
    """Helper to initialize system as per simulate.py logic."""
    sim_config = config['SIM']
    # Filter out SIM and OUTPUT from main config for create_system
    plant_config = {k: v for k, v in config.items() if k not in ['SIM', 'OUTPUT']}
    
    # Matching simulate.create_system logic
    from Lightsaber import System, Laser, Mirror, Beam, Sensor, Controller
    system = System()
    seed = eval(str(sim_config.get('seed_for_random', 'None')))
    
    for comp_name, comp_data in plant_config.items():
        comp_data['name'] = comp_name
        comp_data['simulation_sampling_frequency'] = sim_config['simulation_sampling_frequency']
        comp_data['duration_batch'] = sim_config['duration_batch']
        comp_data['duration_fft'] = sim_config['duration_fft']
        # Instantiate by type name
        cls = globals()[comp_data['type']]
        system.append(cls(comp_data, seed, plot_dir))

    # Link up mirrors to beam
    for beam in system.get_by_type('Beam'):
        mirrors = system.get_by_type('Mirror')
        laser = system.get_by_type('Laser')
        beam.set_parameters(mirrors[0], mirrors[1], laser[0].wavelength)
        
    # Link names to variables
    for comp in system.components:
        for comp_to_be_linked in system.components:
            comp_to_be_linked.substitute_names_by_variables(comp)
            
    # Resolve matrices
    for comp in system.components:
        if hasattr(comp, 'matrix'):
            if isinstance(comp.matrix, str):
                comp.matrix = eval(comp.matrix, {'self': comp, 'np': np})
            elif isinstance(comp.matrix, list):
                new_m = np.zeros((len(comp.matrix), len(comp.matrix[0])), dtype=object)
                for i in range(len(comp.matrix)):
                    for j in range(len(comp.matrix[i])):
                        if isinstance(comp.matrix[i][j], str):
                            new_m[i,j] = eval(comp.matrix[i][j], {'self': comp, 'np': np})
                        else:
                            new_m[i,j] = comp.matrix[i][j]
                comp.matrix = new_m.astype(float)
    return system

class BanditLogger:
    def __init__(self, output_dir, name="DPSKF"):
        self.log_path = os.path.join(output_dir, f"bandit_log_{name}.txt") # Fixed name for easier comparison
        self.file = open(self.log_path, 'w')
        # Header for readability
        self.file.write("Step | Regime | Controller | Reward | Raw | Err | NumClusters | PickedCluster\n")
        self.file.flush()

    def log_step(self, step, regime, controller_name, reward, raw=0, err=0, n_clusters=0, picked_cluster=0):
        line = f"{step} | {regime} | {controller_name} | {reward:.6e} | {raw:.6e} | {err:.6e} | {n_clusters} | {picked_cluster}\n"
        self.file.write(line)
        self.file.flush()

    def close(self):
        self.file.close()

def main(_):
    config = load_config(FLAGS.config)
    
    # Separate SIM and OUTPUT
    sim_config = config['SIM']
    
    fs = sim_config['simulation_sampling_frequency']
    total_samples = int(FLAGS.duration * fs)
    batch_n = FLAGS.batch_size
    bandit_n = int(FLAGS.bandit_interval * fs)
    
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # 1. Initialize Lightsaber System & Advanced Reward Calculator
    print("Initializing Lightsaber system...")
    sys = create_system_compat(config)
    reward_calc = AdvancedRewardCalculator(FLAGS.weights_path, fs=fs)
    
    # Correct component lookup by name
    itm = sys.get_by_name('ITM')[0]
    etm = sys.get_by_name('ETM')[0]
    beam = sys.get_by_name('ARM_BEAM_X')[0]
    ctrl = sys.get_by_name('PITCH_CONTROL')[0]
    sensor = sys.get_by_name('PITCH_SENSOR')[0]
    
    # 2. Initialize Environmental Noise & Bandit
    print(f"Setting up Environment and {FLAGS.bandit_type}...")
    env = SinusoidalPoissonEnv(
        FLAGS.duration, mu=FLAGS.mu, std_dev=FLAGS.std_dev, 
        trans_mean=FLAGS.trans_mean, exponent=FLAGS.exponent, 
        r1_rate=FLAGS.r1_rate, r1_attack=FLAGS.r1_attack, 
        r1_hold=FLAGS.r1_hold, r1_decay=FLAGS.r1_decay
    )
    
    if FLAGS.bandit_type == 'DPSKF':
        bandit = DPSKF(
            num_arms=3, dp_alpha=FLAGS.alpha, window_size=FLAGS.W, 
            Q=FLAGS.Q, R=FLAGS.R, prior_mu=167.5, prior_var=10.0 # Standard research defaults
        )
    elif FLAGS.bandit_type.startswith('Fixed'):
        arm_idx = int(FLAGS.bandit_type.replace('Fixed', ''))
        bandit = FixedBandit(num_arms=3, arm_index=arm_idx)
    else:
        raise ValueError(f"Unknown bandit_type: {FLAGS.bandit_type}")
    
    # Controller choices (Mapping index -> variant name in config)
    # Note: These names should ideally exist in config.yaml under PITCH_CONTROL
    CONTROLLER_MAP = {0: 'C0_nominal', 1: 'C1_high_micro', 2: 'C2_high_micro2'}
    
    # 3. Setup Streamable Noises (Realistic Segmented Paths)
    # Using the realistic and authentic data directly from 6th_clean
    data_path = "../../6th_clean/data"
    
    # ITM/ETM base noise (now correctly populated from Environment with physical scalars)
    regen_needed = True
    if FLAGS.noise_cache and os.path.exists(FLAGS.noise_cache):
        try:
            print(f"Loading cached environmental noise from {FLAGS.noise_cache}...")
            cached_data = np.load(FLAGS.noise_cache)
            sus_noise_itm = cached_data['sus_noise_itm']
            sus_noise_etm = cached_data['sus_noise_etm']
            p_scale_tt = cached_data['p_scale_tt']
            s_scale_tt = cached_data['s_scale_tt']
            
            if len(s_scale_tt) >= total_samples:
                print("Cache valid for current duration.")
                regen_needed = False
            else:
                print(f"Cache duration mismatch ({len(s_scale_tt)} < {total_samples}). Regenerating...")
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating...")

    if regen_needed:
        print("Generating fresh environmental noise streams...")
        sus_noise_itm, sus_noise_etm, p_scale_tt, s_scale_tt = env.get_noise_streams(data_path, fs)
        if FLAGS.noise_cache:
            print(f"Saving generated noise to cache at {FLAGS.noise_cache}...")
            # Trim or pad if needed, but get_noise_streams should follow env.total_duration
            np.savez_compressed(FLAGS.noise_cache, 
                                sus_noise_itm=sus_noise_itm, 
                                sus_noise_etm=sus_noise_etm, 
                                p_scale_tt=p_scale_tt, 
                                s_scale_tt=s_scale_tt)
                                
    input_power_tt = p_scale_tt * config['INPUT']['power']
    
    # 4. PRE-FLIGHT: Prepare Numba Kernel Pointers & Fixed Constants
    P_av_ptr = np.array([beam.P_av], dtype=np.float64)
    N_ptr = np.array([float(beam.N)], dtype=np.float64)
    last_pitch = np.zeros(2, dtype=np.float64)
    
    # Extract constants
    m1_R, m1_T = itm.R, itm.T
    m2_R, m2_T = etm.R, etm.T
    lambda0 = float(sim_config['wave_length'] if 'wave_length' in sim_config else config['INPUT']['wave_length'])
    BS_offset = np.array(beam.BS_offset, dtype=np.float64)
    angle_to_bs = np.array(beam.angle_to_bs).astype(np.float64)
    
    # Advanced mathematical scaling matrices (transforms mechanical geometry to abstract eigenbasis)
    _L = 3994.5
    _g1 = 1 - _L / 1934.0
    _g2 = 1 - _L / 2245.0
    _r = 0.5 * (_g1 - _g2 + np.sqrt((_g1 - _g2)**2 + 4))
    advanced_eigen_matrix = np.array([[1, _r], [-_r, 1]]) / (1 + _r**2)
    advanced_local_matrix = np.array([[1, -_r], [_r, 1]])
    
    local_to_eigen = np.ascontiguousarray(advanced_eigen_matrix.astype(np.float64))
    eigen_to_local = np.ascontiguousarray(advanced_local_matrix.astype(np.float64))
    
    # Pre-extract plant SOS (Fixed)
    hp_sos = np.array(beam.high_pass_sos).astype(np.float64)
    hp_zi = np.zeros((hp_sos.shape[0], 2))
    
    # 5. MAIN LOOP
    # 5. MAIN LOOP
    print(f"Starting simulation: {FLAGS.duration}s ({total_samples} samples)")
    start_time = time.time()
    
    rewards = []
    decisions = []
    
    t_samples = 0
    logger = BanditLogger(FLAGS.output_dir, name=FLAGS.bandit_type)
    
    # Initialization of SS Compensation logic
    ss_comp = SS_compensation(beam.L, itm.RoC, etm.RoC, fs)
    pav_ptr = np.array([45000.0], dtype=np.float64) # Seed realistic power to prevent startup ringing
    n_pav_ptr = np.array([1000.0], dtype=np.float64)
    prev_ss = np.zeros(2, dtype=np.float64)
    # The SS_compensation internal zpk state tracker
    ss_zi = np.zeros((2, 4, 2), dtype=np.float64)
    
    # Accumulation buffers for accurate batched Log-RMS scoring.
    # Advanced logic: We accumulate the microscopic physical state frames (pitch traces and actuation signals) 
    # to evaluate the full macroscopic continuous reward scalar precisely matched with the original logic basis.
    acc_pitch = []
    acc_actuation = []
    
    # ADVANCED FEATURE: Physical Continuity Latches
    # These tracking arrays are deliberately pulled outside the JIT loop to permit exact, micro-second
    # continuation of mechanical and control states across Python's larger "Time Chunk" breaks.
    last_pitch = np.zeros(2)
    last_act = np.zeros(2)
    
    # 5.1 BOOTSTRAP: Pick initial arm
    current_arm = bandit.select_arm()
    decisions.append(current_arm)
    ctrl.set_controller_variant(config['PITCH_CONTROL'], CONTROLLER_MAP[current_arm])
    
    # Setup SS initialization unit arrays
    ss_s_sos, ss_h_sos = ss_comp.get_ss_sos_unit()
    ss_s_sos = np.ascontiguousarray(ss_s_sos.astype(np.float64))
    ss_h_sos = np.ascontiguousarray(ss_h_sos.astype(np.float64))
    ss_eigen2local = np.ascontiguousarray(ss_comp.eigen2local.astype(np.float64))
    
    print(f"Starting simulation loop for {FLAGS.bandit_type}...", flush=True)
    pbar = tqdm(total=total_samples, desc=f"Simulating {FLAGS.bandit_type}", unit="sample")
    
    while t_samples < total_samples:
        # -- KERNEL EXECUTION (Fast physics in batches) --
        n_block = min(batch_n, total_samples - t_samples)
        
        # Get latest controller SOS
        ctrl_sos_s, ctrl_zi_s, ctrl_sos_h, ctrl_zi_h = ctrl.get_sos_for_numba()
        
        # Authentic dynamic sensor noise scaled per-sample like 6th_clean
        scale_s_base = (fs / 2.0)**0.5 * 1e-13
        scale_h_base = (fs / 2.0)**0.5 * 3e-14
        block_s_scale = s_scale_tt[t_samples:t_samples + n_block]
        
        sn_s = np.random.normal(0, 1.0, n_block) * scale_s_base * block_s_scale
        sn_h = np.random.normal(0, 1.0, n_block) * scale_h_base * block_s_scale
        
        # NOTE: ss_s_sos and ss_h_sos are static "unit" representations passed from outside now.

        readout, actuation, pitch, last_p, act_latch = run_fast_physics_kernel(
            n_block, t_samples,
            sus_noise_itm, sus_noise_etm,
            input_power_tt,
            sn_s, sn_h,
            m1_R, m1_T, m2_R, m2_T,
            lambda0, BS_offset, angle_to_bs,
            hp_sos, hp_zi,
            itm.rad_to_angle_sos, itm.rad_to_angle_sos_state,
            etm.rad_to_angle_sos, etm.rad_to_angle_sos_state,
            itm.act_to_angle_sos, itm.act_to_angle_sos_state,
            etm.act_to_angle_sos, etm.act_to_angle_sos_state,
            ctrl_sos_s, ctrl_zi_s,
            ctrl_sos_h, ctrl_zi_h,
            eigen_to_local,
            local_to_eigen,
            ss_s_sos, ss_h_sos, ss_zi, ss_eigen2local, prev_ss,
            ss_comp.dydth_soft, ss_comp.dydth_hard, ss_comp.kk_lp, ss_comp.P_const,
            pav_ptr, n_pav_ptr,
            last_pitch,
            last_act
        )
        
        # Store results
        acc_pitch.append(pitch)
        acc_actuation.append(actuation)
        
        # Update loop pointers
        t_samples += n_block
        pbar.update(n_block)
        
        # Latch
        last_pitch = last_p
        last_act = act_latch
        
        # -- BANDIT UPDATE & DECISION (Every 300s) --
        if t_samples % bandit_n == 0 or t_samples == total_samples:
            # Aggregate the 300s window
            agg_pitch = np.concatenate(acc_pitch)
            agg_actuation = np.concatenate(acc_actuation)
            
            # 5.2 LEARN: Update with aggregated reward
            # 6th_clean transforms arrays to Eigen space before reward calculation
            agg_pitch_eig = (research_eigen_matrix @ agg_pitch.T).T
            agg_actuation_eig = (research_eigen_matrix @ agg_actuation.T).T
            
            # Extract ONLY Hard Mode (Index 1) for research-standard calculation
            reward, raw_score = reward_calc.calculate_reward(agg_pitch_eig[:, 1], agg_actuation_eig[:, 1])
            bandit.update(current_arm, reward)
            
            # 5.3 LOG: Log state BEFORE picking next arm
            w0, w1, w2 = env.get_weights(t_samples / fs)
            regime_str = f"M_{w0:.2f}_{w1:.2f}_{w2:.2f}"
            logger.log_step(
                int(t_samples/fs), regime_str, CONTROLLER_MAP[current_arm], reward,
                raw=raw_score, err=np.std(agg_pitch), 
                n_clusters=bandit.num_clusters, picked_cluster=bandit.c[current_arm]
            )
            
            # 5.4 ACT: Pick arm for NEXT interval
            if t_samples < total_samples:
                current_arm = bandit.select_arm()
                decisions.append(current_arm)
                ctrl.set_controller_variant(config['PITCH_CONTROL'], CONTROLLER_MAP[current_arm])
            
            # 5.5 RESET: Clear intervals
            acc_pitch = []
            acc_actuation = []

    pbar.close()
    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s (Speedup: {FLAGS.duration/duration:.1f}x)")
    
    # Save results
    np.save(os.path.join(FLAGS.output_dir, "decisions.npy"), np.array(decisions))
    logger.close()
    
    # Auto-generate diagnostic plot
    print("Generating diagnostic plots...")
    plot_bandit_log(logger.log_path)
    
    print(f"Results saved to {FLAGS.output_dir}")

if __name__ == '__main__':
    app.run(main)
