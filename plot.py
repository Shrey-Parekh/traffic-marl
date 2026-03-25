"""
Generate IEEE-quality convergence plot for the paper.
Reads seed 1 CSVs and produces a clean PDF figure.

Usage:
    python plot_convergence.py

Output:
    figures/convergence.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# IEEE style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
})

# Model colors (consistent with paper)
COLORS = {
    'DQN':          '#404040',   # Dark gray
    'GNN-DQN':      '#1F77B4',   # Steel blue
    'GAT-DQN-Base': '#E36209',   # Burnt orange
    'GAT-DQN':      '#2CA02C',   # Forest green
}

# Baseline values (uniform scenario, 3 seeds)
BASELINES = {
    'Fixed-Time':  {'queue': 12.42, 'color': '#A8C8E8', 'dash': (4, 2)},
    'Webster':     {'queue': 13.45, 'color': '#D95F02', 'dash': (6, 2)},
    'MaxPressure': {'queue': 14.18, 'color': '#7570B3', 'dash': (2, 2)},
}

def ema(data, window=15):
    """Exponential moving average."""
    alpha = 2.0 / (window + 1)
    result = [data[0]]
    for v in data[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def main():
    # Load seed 1 CSVs from outputs directory
    # Adjust this path to where your CSVs are
    data_dir = Path("outputs")
    
    models = {}
    for name in ['DQN', 'GNN-DQN', 'GAT-DQN-Base', 'GAT-DQN']:
        csv_path = data_dir / f"{name}_3_uniform_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            models[name] = df
            print(f"Loaded {name}: {len(df)} episodes")
        else:
            print(f"WARNING: {csv_path} not found")

    if not models:
        print("No CSV files found. Check the data_dir path.")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))  # IEEE single column width
    
    # Plot each model
    for name in ['DQN', 'GNN-DQN', 'GAT-DQN-Base', 'GAT-DQN']:
        if name not in models:
            continue
        df = models[name]
        episodes = df['episode'].values
        queues = df['avg_queue'].values
        smoothed = ema(queues.tolist(), window=15)

        # Smoothed line
        lw = 0.75 if name == 'GAT-DQN' else 0.5
        ax.plot(episodes, smoothed, color=COLORS[name], linewidth=lw,
                label=name, zorder=3 if name == 'GAT-DQN' else 2)

    # Plot baseline reference lines
    for bname, bdata in BASELINES.items():
        ax.axhline(y=bdata['queue'], color=bdata['color'],
                   linestyle='--', linewidth=0.7, alpha=0.7, zorder=1)
        ax.text(5, bdata['queue'] + 0.15, f"{bname} ({bdata['queue']:.1f})",
                fontsize=6.5, color=bdata['color'], va='bottom')

    # Axis labels
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Queue Length (PCU)')
    
    # Axis limits
    ax.set_xlim(1, 250)
    ax.set_ylim(9, 15)
    
    # Grid
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    
    # Black border on all sides
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    # Ticks
    ax.tick_params(direction='in', length=3, width=0.6)
    
    # Legend (inside plot, upper right)
    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black',
                       fancybox=False, framealpha=0.9)
    legend.get_frame().set_linewidth(0.6)
    
    # Tight layout
    plt.tight_layout(pad=0.3)
    
    # Save
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    
    fig.savefig(out_dir / "convergence.pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_dir / "convergence.png", bbox_inches='tight', pad_inches=0.02)
    print(f"Saved to {out_dir / 'convergence.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()