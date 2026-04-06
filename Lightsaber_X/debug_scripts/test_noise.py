import numpy as np
import matplotlib.pyplot as plt
from noise_models import SinusoidalPoissonEnv

def test_environmental_weights():
    duration = 86400  # 24 Hours
    env = SinusoidalPoissonEnv(duration, seed=42)
    
    t_eval = np.linspace(0, duration, 1000)
    w0_list, w1_list, w2_list = [], [], []
    
    print("Evaluating environmental weights over 24 hours...")
    for t in t_eval:
        w = env.get_weights(t)
        w0_list.append(w[0])
        w1_list.append(w[1])
        w2_list.append(w[2])
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval/3600, w0_list, label='Regime 0 (Nominal)')
    plt.plot(t_eval/3600, w1_list, label='Regime 1 (Spikes/Storms)', alpha=0.8)
    plt.plot(t_eval/3600, w2_list, label='Regime 2 (Extreme Drift)', linestyle='--')
    
    plt.title('Time-Varying Environmental Weights (24-Hour Cycle)')
    plt.xlabel('Time [Hours]')
    plt.ylabel('Mixing Weight')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = 'environmental_weights_test.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    test_environmental_weights()
