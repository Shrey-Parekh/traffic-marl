# Mini Traffic MARL

A lightweight queue-based multi-agent reinforcement learning (MARL) traffic simulator with a shared-policy DQN controller, a fixed-time baseline, rich metrics, and an interactive Streamlit dashboard. Runs efficiently on CPU-only machines.

## What's Inside

- **Multi-Agent simulator**: N intersections, each an agent sharing one neural network policy
- **Learning controller**: DQN with replay buffer, target network and epsilon-greedy exploration
- **Fixed-time baseline**: Deterministic switching every 20 steps for apples-to-apples comparison
- **Interactive dashboard**: One click to run baseline + learning, visualize results and history
- **Metrics & reports**: JSON history, CSV, human-readable summary, final report and saved policy

## Requirements

- Python 3.9+ (3.10–3.12 tested)
- Windows/macOS/Linux, CPU only

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start (Dashboard)

Launch the UI:

```bash
streamlit run src/dashboard.py
```

In the sidebar:
- Set the number of intersections (N), the random seed, and the number of learning episodes
- The baseline's switch period is fixed at 20 steps
- Each episode runs for a fixed number of steps (shown in the UI)

Click "Run Simulation". The dashboard will:
1) Simulate the fixed-time baseline
2) Train the shared-policy MARL controller for the requested episodes
3) Update metrics files and refresh visualizations automatically

## How the System Works

- **Environment** (`src/env.py`): queue-based roads; every step simulates arrivals, departures and light phase actions
- **Agent** (`src/agent.py`): DQN + replay buffer; shared weights across all intersections
- **Training** (`src/train.py`): runs episodes, optimizes DQN, and writes metrics after every episode
- **Dashboard** (`src/dashboard.py`): orchestrates baseline + training runs and renders:
  - History table sourced from `outputs/metrics.json` (latest first)
  - Learning curves (avg queue, throughput, travel time)
  - Per-intersection analysis from the latest episode
- **Config** (`src/config.py`): centralized configuration for all hyperparameters and paths

## Command Line Usage

### Training Script

Train the MARL controller from the command line:

```bash
python -m src.train
```

**With custom parameters:**

```bash
python -m src.train --episodes 50 --N 4 --max_steps 300 --seed 42
```

**All available parameters:**

```bash
python -m src.train \
  --episodes 50              # Number of training episodes (default: 50)
  --N 4                      # Number of intersections (default: 2)
  --max_steps 300            # Steps per episode (default: 300)
  --lr 0.001                 # Learning rate (default: 0.001)
  --batch_size 64            # Batch size for training (default: 64)
  --gamma 0.99               # Discount factor (default: 0.99)
  --replay_capacity 20000    # Replay buffer size (default: 20000)
  --epsilon_start 1.0        # Initial exploration rate (default: 1.0)
  --epsilon_end 0.05         # Final exploration rate (default: 0.05)
  --epsilon_decay_steps 5000 # Epsilon decay steps (default: 5000)
  --update_target_steps 200  # Target network update frequency (default: 200)
  --min_buffer_size 1000     # Minimum buffer size before training (default: 1000)
  --seed 123                 # Random seed (default: 123)
  --save_dir outputs         # Output directory (default: outputs)
  --neighbor_obs             # Enable neighbor observations (optional flag)
```

**Examples:**

```bash
# Quick test
python -m src.train --episodes 10 --N 2 --seed 42

# Longer training
python -m src.train --episodes 100 --N 6 --max_steps 500 --seed 42

# With neighbor observations
python -m src.train --episodes 50 --N 4 --neighbor_obs --seed 42
```

### Baseline Script

Run the fixed-time baseline for comparison:

```bash
python -m src.baseline
```

**With custom parameters:**

```bash
python -m src.baseline --episodes 10 --N 4 --switch_period 20 --seed 42
```

**All available parameters:**

```bash
python -m src.baseline \
  --episodes 10              # Number of episodes (default: 10)
  --N 4                      # Number of intersections (default: 2)
  --max_steps 300            # Steps per episode (default: 300)
  --switch_period 20         # Switch period in steps (default: 20)
  --seed 123                 # Random seed (default: 123)
  --save_dir outputs         # Output directory (default: outputs)
```

### Generate Comprehensive Baseline

Generate baseline data across multiple switch periods and seeds:

```bash
python -m src.generate_baseline
```

**With custom parameters:**

```bash
python -m src.generate_baseline \
  --episodes 20 \
  --N 6 \
  --max_steps 300 \
  --switch_periods "10,15,20,25,30" \
  --seeds "1,2,3,4,5" \
  --save_dir outputs
```

### Run Multiple Scenarios

Run training across multiple seeds and network sizes:

```bash
python -m src.scenarios
```

**With custom parameters:**

```bash
python -m src.scenarios \
  --total_episodes 100 \
  --seeds "1,2,3,4,5" \
  --Ns "2,4,6"
```

## Outputs (in `outputs/`)

- `metrics.json`: full per-episode history used by the dashboard's Run History
- `metrics.csv`: spreadsheet-friendly export of the same
- `live_metrics.json`: latest episode summary (ingested by the dashboard)
- `summary.txt`: rolling human-readable status (updates each episode)
- `final_report.json`: aggregate stats at the end of a run
- `policy_final.pth`: saved DQN weights
- `baseline_metrics.json`: Baseline results (when running baseline scripts)

The project preserves previous results; new runs append to the history rather than wiping it.

## Run History – Column Guide

- **Episode**: 1-based episode index
- **Agents**: Number of learning agents (equals intersections `N`)
- **Epsilon**: Exploration rate used for that episode
- **Avg Queue**: Average cars waiting per intersection (lower is better)
- **Throughput**: Vehicles that finished during the episode (higher is better)
- **Avg Travel Time (s)**: Average time spent in the network (lower is better)
- **Loss**: Average DQN training loss during the episode
- **Updates**: Gradient updates performed in the episode

## Project Layout

```
traffic-marl/
├── src/
│   ├── agent.py          # DQN model and replay buffer
│   ├── env.py            # Queue-based multi-intersection environment
│   ├── train.py          # Training loop, logging and file outputs
│   ├── dashboard.py      # Streamlit UI to run and visualize experiments
│   ├── baseline.py       # Fixed-time controller utilities
│   ├── scenarios.py      # Batch runner for multiple seeds/sizes
│   ├── generate_baseline.py  # Comprehensive baseline generator
│   └── config.py         # Centralized configuration and hyperparameters
├── outputs/              # Results written here after runs
├── requirements.txt
└── README.md
```

## Recommended Workflow

### For Quick Testing:
```bash
# Start the dashboard
streamlit run src/dashboard.py
```
Then use the web interface with default settings.

### For Custom Training:
```bash
# Train with your parameters
python -m src.train --episodes 50 --N 4 --seed 42

# View results in dashboard
streamlit run src/dashboard.py
```

### For Comparison Studies:
```bash
# 1. Run baseline
python -m src.baseline --episodes 20 --N 4 --seed 42

# 2. Run AI training
python -m src.train --episodes 20 --N 4 --seed 42

# 3. Compare results in dashboard
streamlit run src/dashboard.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running from the project root directory:
```bash
cd /path/to/traffic-marl
python -m src.train
```

### Issue: "streamlit: command not found"
**Solution**: Install Streamlit:
```bash
pip install streamlit
```

### Issue: Dashboard won't start
**Solution**: 
- Check if port 8501 is already in use
- Try: `streamlit run src/dashboard.py --server.port 8502`

### Issue: Training is slow
**Solution**: 
- Reduce number of episodes: `--episodes 10`
- Reduce intersections: `--N 2`
- Reduce steps per episode: `--max_steps 150`

### Issue: Out of memory errors
**Solution**:
- Reduce replay buffer: `--replay_capacity 10000`
- Reduce batch size: `--batch_size 32`
- Reduce number of intersections: `--N 2`

## Tips

1. **Start small**: Begin with `--N 2` and `--episodes 10` to test quickly
2. **Use seeds**: Always specify `--seed` for reproducible results
3. **Check outputs**: Look at `outputs/summary.txt` for quick status updates
4. **Dashboard auto-refresh**: The dashboard refreshes every 5 seconds (configurable in sidebar)
5. **Compare fairly**: Use the same seed for baseline and AI training for fair comparison

## Key Improvements

The project includes several improvements for stability and accuracy:

1. **Observation Normalization**: Queue lengths and time values are normalized for better neural network training
2. **Weight Initialization**: Xavier/Glorot initialization for improved convergence
3. **Training Stability**: Minimum buffer size (warm-up period) prevents unstable early training
4. **Safety Checks**: Queue serving and travel time calculations include safety validations
5. **Logging**: All output uses Python's logging module for better control and debugging
6. **Path Management**: Uses `pathlib.Path` for robust cross-platform path handling
7. **Centralized Config**: All hyperparameters centralized in `src/config.py`

## Example Commands Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (dashboard)
streamlit run src/dashboard.py

# Quick test (command line)
python -m src.train --episodes 10 --N 2 --seed 42

# Standard training
python -m src.train --episodes 50 --N 4 --seed 42

# Baseline comparison
python -m src.baseline --episodes 20 --N 4 --seed 42

# Comprehensive baseline study
python -m src.generate_baseline --episodes 20 --N 6 --seeds "1,2,3"

# Multiple scenarios
python -m src.scenarios --total_episodes 100 --seeds "1,2,3" --Ns "2,4,6"
```

## License

Open source; see the license file for details.
