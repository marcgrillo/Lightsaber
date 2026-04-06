import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_bandit_log(log_abspath):
    """
    Parses a bandit log file into a DataFrame.
    Format: Step | Regime | Controller | Reward | Raw | Err | NumClusters | PickedCluster
    """
    data = []
    if not os.path.exists(log_abspath): return pd.DataFrame()

    with open(log_abspath, 'r') as f:
        for line in f:
            if "|" not in line: continue
            parts = [p.strip() for p in line.split('|')]
            
            try:
                step = int(parts[0])
                regime = parts[1]
                controller_str = parts[2]
                
                # Robust extraction of C0, C1, C2 index
                c_idx = -1
                for i in range(10): # support up to 10 controllers
                    if f"C{i}" in controller_str:
                        c_idx = i
                        break
                        
                reward = float(parts[3])
                
                raw_score = np.nan
                rms_err = np.nan
                num_clusters = np.nan
                picked_cluster = np.nan
                
                if len(parts) >= 5:
                    try: raw_score = float(parts[4])
                    except: pass
                if len(parts) >= 6:
                    try: rms_err = float(parts[5])
                    except: pass
                if len(parts) >= 7:
                    try: num_clusters = float(parts[6])
                    except: pass
                if len(parts) >= 8:
                    try: picked_cluster = float(parts[7])
                    except: pass

                data.append({
                    'Step': step,
                    'Regime': regime,
                    'Controller': c_idx,
                    'Reward': reward,
                    'raw_score': raw_score,
                    'rms_err': rms_err,
                    'NumClusters': num_clusters,
                    'PickedCluster': picked_cluster
                })
            except (ValueError, IndexError):
                continue
                
    return pd.DataFrame(data)

def plot_bandit_log(log_file, out_file=None):
    """
    Generates a 4-panel diagnostic plot for a bandit log.
    """
    df = parse_bandit_log(log_file)
    if df.empty:
        print(f"Warning: Empty log file {log_file}")
        return

    if out_file is None:
        out_file = log_file.replace(".txt", ".png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 1. Regime vs Time
    unique_regimes = sorted(df['Regime'].unique())
    regime_map = {r: i for i, r in enumerate(unique_regimes)}
    df['RegimeIdx'] = df['Regime'].map(regime_map)
    axes[0].step(df['Step'], df['RegimeIdx'], where='post', color='#1f77b4', linewidth=2)
    axes[0].set_ylabel('Regime')
    
    # Limit number of y-axis labels to 10
    indices = np.arange(len(unique_regimes))
    if len(indices) > 10:
        indices_to_show = np.linspace(0, len(indices) - 1, 10, dtype=int)
        axes[0].set_yticks(indices_to_show)
        axes[0].set_yticklabels([unique_regimes[i] for i in indices_to_show])
    else:
        axes[0].set_yticks(list(regime_map.values()))
        axes[0].set_yticklabels(list(regime_map.keys()))
        
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Bandit Diagnostic: {os.path.basename(log_file)}")

    # 2. Controller vs Time
    axes[1].scatter(df['Step'], df['Controller'], color='#ff7f0e', alpha=0.6, s=15, label='Picked Controller')
    axes[1].set_ylabel('Controller Index')
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['C0', 'C1', 'C2'])
    axes[1].set_ylim(-0.5, 2.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # 3. Reward vs Time
    axes[2].plot(df['Step'], df['Reward'], color='#2ca02c', alpha=0.4, marker='o', markersize=3, linestyle='None')
    window = min(50, len(df))
    if window > 5:
        axes[2].plot(df['Step'], df['Reward'].rolling(window=window, center=True).mean(), color='#2ca02c', linewidth=2, label=f'MA({window})')
    axes[2].set_ylabel('Reward')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    # 4. Score/Error vs Time
    ax4 = axes[3]
    if 'raw_score' in df.columns and not df['raw_score'].isna().all():
        ax4.plot(df['Step'], df['raw_score'], color='blue', label='Raw Score', alpha=0.7)
        ax4.set_ylabel('Score (Log-Lik)', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        
    if 'rms_err' in df.columns and not df['rms_err'].isna().all():
        ax4b = ax4.twinx()
        ax4b.plot(df['Step'], df['rms_err'], color='red', linestyle='--', label='RMS Err', alpha=0.7)
        ax4b.set_ylabel('RMS Error', color='red')
        ax4b.tick_params(axis='y', labelcolor='red')
        ax4b.set_yscale('log')
        
    ax4.set_xlabel('Time [s]')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved diagnostic plot to {out_file}")

def plot_comparison(run_folder, out_file):
    """
    Plots cumulative rewards and advantage vs a best-performing baseline.
    """
    log_files = glob.glob(os.path.join(run_folder, "bandit_log_*.txt"))
    if not log_files:
        print(f"No log files found in {run_folder}")
        return

    # Use a 2-panel plot for better visibility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    algo_data = {}

    for lf in sorted(log_files):
        name = os.path.basename(lf).replace("bandit_log_", "").replace(".txt", "")
        df = parse_bandit_log(lf)
        if df.empty: continue
        df['cum_reward'] = df['Reward'].cumsum()
        algo_data[name] = df

    if not algo_data: return

    # Identify best baseline at final step
    max_step = min([df['Step'].max() for df in algo_data.values()])
    best_algo = max(algo_data, key=lambda n: algo_data[n][algo_data[n]['Step'] <= max_step]['cum_reward'].iloc[-1])
    baseline = algo_data[best_algo][['Step', 'cum_reward']].rename(columns={'cum_reward': 'best_baseline'})

    colors = plt.cm.tab10.colors
    for i, (name, df) in enumerate(algo_data.items()):
        df = df[df['Step'] <= max_step]
        merged = pd.merge(df, baseline, on='Step', how='left').ffill()
        advantage = merged['cum_reward'] - merged['best_baseline']
        
        c = colors[i % len(colors)]
        
        # 1. Cumulative Reward Advantage
        ax1.plot(df['Step'], advantage, label=name, color=c, linewidth=2)
        
        # 2. Moving Average Reward
        ma_rew = df['Reward'].rolling(window=min(100, len(df)), center=True).mean()
        ax2.plot(df['Step'], ma_rew, label=name, color=c, alpha=0.8)

    ax1.set_title("Cumulative Reward Advantage vs Best")
    ax1.set_ylabel("Advantage")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Moving Average Reward")
    ax2.set_ylabel("Reward")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {out_file}")
