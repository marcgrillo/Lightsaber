import os
import subprocess
import argparse
import time
from plotting import plot_comparison

def main():
    parser = argparse.ArgumentParser(description="Competitive Benchmarking: Bandit vs Fixed Controllers")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    parser.add_argument("--output_dir", type=str, default="results/test_run", help="Directory for all results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    algorithms = ["DPSKF", "Fixed0", "Fixed1", "Fixed2"]
    
    print(f"Starting competitive benchmark for {args.duration}s...")
    
    for algo in algorithms:
        print(f"\n>>> Running {algo}...")
        cmd = [
            "python", "simulate_bandit.py",
            f"--duration={args.duration}",
            f"--output_dir={args.output_dir}",
            f"--bandit_type={algo}",
            "--bandit_interval=300",
            "--noise_cache=data/cached_compare_noise.npz"
        ]
        
        try:
            # Use subprocess to ensure clean state for each run
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {algo}: {e}")
            continue

    print("\n>>> All simulations complete. Generating comparison plots...")
    comparison_plot = os.path.join(args.output_dir, "realtime_cumulative_reward.png")
    plot_comparison(args.output_dir, comparison_plot)
    
    print(f"\n[SUCCESS] Benchmark complete. Results and plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
