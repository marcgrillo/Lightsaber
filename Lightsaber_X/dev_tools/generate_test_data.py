import os
import numpy as np
import pandas as pd
from scipy import signal

def generate_realistic_asd(f, base_level=1e-12, slope=-2):
    """Generates a realistic noise ASD following a power law with some resonances."""
    asd = base_level * (f + 0.1)**slope
    # Add some resonances (simulating suspension peaks)
    resonances = [1.2, 3.5, 10.5, 45.0]
    for res_f in resonances:
        asd += 1e-11 * np.exp(-(f - res_f)**2 / (0.1 * res_f))
    return asd

def save_asd(filename, ff, asd):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for freq, val in zip(ff, asd):
            f.write(f"{freq:.18e} {val:.18e}\n")

def save_tf(filename, ff, tf):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for freq, val in zip(ff, tf):
            # Format as complex string: (real+imagj)
            f.write(f"{freq:.18e} {val.real:+.18e}{val.imag:+.18e}j\n")

def main():
    fs = 256
    duration_s = 12 * 3600  # 12 hours
    n_samples = int(duration_s * fs)
    output_dir = "data/temporary_test_data"
    
    print(f"Generating realistic test data for {duration_s/3600:.1f} hours...")
    
    ff = np.linspace(0, fs//2, 8193)
    
    # 1. Generate Regime ASDs
    # R0: Nominal
    asd_r0 = generate_realistic_asd(ff, base_level=1e-12)
    save_asd(os.path.join(output_dir, "ASD_R0_nominal.csv"), ff, asd_r0)
    
    # R1: High Micro (more noise at low frequencies)
    asd_r1 = generate_realistic_asd(ff, base_level=5e-12) + 1e-10 * np.exp(-ff/2.0)
    save_asd(os.path.join(output_dir, "ASD_R1_high_micro.csv"), ff, asd_r1)
    
    # R2: Extreme
    asd_r2 = generate_realistic_asd(ff, base_level=2e-11)
    save_asd(os.path.join(output_dir, "ASD_R2_high_micro2.csv"), ff, asd_r2)
    
    # 2. Others (OSEM, RIN, A+)
    asd_osem = generate_realistic_asd(ff, base_level=1e-13, slope=-1)
    save_asd(os.path.join(output_dir, "n_osem_L.txt"), ff, asd_osem)
    save_asd(os.path.join(output_dir, "n_osem_P.txt"), ff, asd_osem)
    
    # RIN (Input Power)
    asd_rin = 1e-8 * np.ones_like(ff)
    # Save as CSV with comma for genfromtxt compat
    df_rin = pd.DataFrame({'f': ff, 'asd': asd_rin})
    df_rin.to_csv(os.path.join(output_dir, "O3_power_psd.csv"), index=False, header=False)
    
    # A+ curve
    asd_aplus = 1e-23 * np.ones_like(ff) # Simplification for test
    with open(os.path.join(output_dir, "aicReferenceData_Aplus.txt"), 'w') as f:
        for freq, val in zip(ff, asd_aplus):
            f.write(f"{freq} {val*0.1} {val*0.2} {val*0.3} {val}\n")
            
    # 3. Transfer Functions
    # Simplified PUM -> TST TF
    z, p, k = signal.butter(2, 2.0, btype='low', analog=True, output='zpk')
    w, h = signal.freqs_zpk(z, p, k, 2 * np.pi * ff)
    save_tf(os.path.join(output_dir, "tf_topL_2_tstP.txt"), ff, h)
    save_tf(os.path.join(output_dir, "tf_topNL_2_tstP.txt"), ff, h)
    save_tf(os.path.join(output_dir, "tf_topNP_2_tstP.txt"), ff, h)

    print(f"Realistic ASDs and TFs generated in {output_dir}")
    
    # 4. Generate some "current" traces for the full test if needed
    # (Actually the simulator will generate these from ASDs if configured correctly)
    
    print("Done.")

if __name__ == "__main__":
    main()
