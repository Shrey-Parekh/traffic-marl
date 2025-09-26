## Mini Traffic MARL

A lightweight queue-based multi-agent reinforcement learning (MARL) traffic simulator featuring a shared-policy Deep Q-Network (DQN) controller, fixed-time baseline comparison, comprehensive evaluation metrics, and an interactive Streamlit dashboard. Designed to run efficiently on CPU-only systems without internet connectivity.

### 🚦 Key Features

- **Multi-Agent Traffic Simulation**: Queue-based simulator with N intersections, each having two phases (North-South/East-West)
- **Intelligent Traffic Control**: Shared-policy DQN across all traffic light agents with parameter sharing for efficient learning
- **Realistic Traffic Modeling**: Poisson arrival distributions, fixed departure capacity, and simple routing logic
- **Comprehensive Evaluation**: Fixed-time baseline controller and multi-scenario baseline generator for fair comparison
- **Rich Analytics**: Detailed metrics logging (JSON, CSV, summary reports) with per-intersection analysis
- **Interactive Dashboard**: Real-time Streamlit dashboard with live metrics, training progress, and baseline comparisons
- **Reproducible Results**: Seed-based reproducibility for consistent experimental results

### 📋 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: CPU-only (no GPU required)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: ~500MB for dependencies and outputs

### 🔧 Installation

#### Option 1: Automated Installation (Windows)

**For Command Prompt:**
```cmd
install.bat
```

**For PowerShell:**
```powershell
.\install.ps1
```

#### Option 2: Manual Installation (All Platforms)

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.\.venv\Scripts\activate.bat
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 📦 Dependencies

The project uses the following core libraries:
- **numpy** (≥1.21.0): Numerical computing and array operations
- **torch** (≥1.9.0): Deep learning framework for DQN implementation
- **tqdm** (≥4.62.0): Progress bars for training visualization
- **streamlit** (≥1.28.0): Interactive web dashboard framework
- **matplotlib** (≥3.5.0): Static plotting and visualization
- **pandas** (≥1.3.0): Data manipulation and analysis
- **plotly** (≥5.0.0): Interactive charts and graphs

### 🚀 Quick Start Guide

#### 1. Basic Training (3-minute demo)
```bash
# Train AI controller with 5 episodes
python -m src.train --episodes 5 --N 4

# Run fixed-time baseline for comparison
python -m src.baseline --episodes 5 --N 4 --switch_period 20

# Launch interactive dashboard
streamlit run src/dashboard.py
```

#### 2. Extended Training & Evaluation
```bash
# Train with more episodes for better performance
python -m src.train --episodes 50 --N 6 --max_steps 500

# Generate comprehensive baseline data
python -m src.generate_baseline --episodes 20 --N 6 --switch_periods 10,15,20,25,30 --seeds 1,2,3,4,5

# Run multi-scenario evaluation
python -m src.scenarios --total_episodes 100 --seeds 1,2,3,4,5 --Ns 2,4,6
```

#### 3. Dashboard Features
After running training, launch the dashboard to explore:
```bash
streamlit run src/dashboard.py
```

**Dashboard Capabilities:**
- 📊 Real-time training progress visualization
- 🚦 Per-intersection performance analysis
- 📈 AI vs Fixed-time baseline comparison
- ⚡ Interactive simulation runs
- 📋 Exportable metrics and reports


### ⚙️ Configuration Options

#### Training Parameters (`python -m src.train`)
```bash
--episodes 50              # Number of training episodes
--N 2                      # Number of intersections
--max_steps 300            # Steps per episode
--lr 1e-3                  # Learning rate
--batch_size 64            # Batch size for training
--gamma 0.99               # Discount factor
--replay_capacity 20000    # Replay buffer size
--epsilon_start 1.0        # Initial exploration rate
--epsilon_end 0.05         # Final exploration rate
--epsilon_decay_steps 5000 # Steps to decay exploration
--update_target_steps 200  # Target network update frequency
--seed 123                 # Random seed for reproducibility
--neighbor_obs             # Include neighbor observations
--save_dir outputs         # Output directory
```

#### Baseline Parameters (`python -m src.baseline`)
```bash
--episodes 10              # Number of episodes to run
--N 2                      # Number of intersections
--max_steps 300            # Steps per episode
--switch_period 20         # Fixed switching period
--seed 123                 # Random seed
--save_dir outputs         # Output directory
```

#### Comprehensive Baseline (`python -m src.generate_baseline`)
```bash
--episodes 20              # Episodes per configuration
--N 6                      # Number of intersections
--max_steps 300            # Steps per episode
--switch_periods "10,15,20,25,30"  # Periods to test
--seeds "1,2,3,4,5"        # Seeds for statistical analysis
--save_dir outputs         # Output directory
```

#### Multi-Scenario Evaluation (`python -m src.scenarios`)
```bash
--total_episodes 100       # Total episodes across all runs
--seeds "1,2,3,4,5"        # Seeds to test
--Ns "2,4,6"               # Network sizes to test
```

### 🧠 How the System Works

#### Simple Explanation
- You simulate a line of intersections. Every few seconds (a "step"), cars arrive randomly, the controller decides which direction is green (NS or EW) at each intersection, and cars either exit (NS green) or move to the next intersection (EW green).
- Each intersection is an agent, but all agents share one neural network (shared policy). That one policy maps each intersection’s local observation to an action (keep phase or switch), enabling fast and consistent learning.
- The controller gets a reward that is the negative of the queues (fewer waiting cars is better) and learns from many episodes to reduce queues and travel time while keeping throughput high.

### Key terms you’ll see
- Episode: one complete simulation run (reset → `max_steps`). Think “one day” of traffic.
- Step: one small time slice (default ~2 s). Cars arrive, lights act, cars move.
- Observation: for each intersection `[NS queue, EW queue, current phase, time since switch]` (+ optional neighbor EW).
- Action: for each intersection `0=keep` or `1=switch` (respects min_green).
- Reward: for each intersection `-(NS queue + EW queue)`; fewer cars waiting is better.
- Seed: fixes random arrivals so runs are reproducible and comparable.
- Baseline (fixed-time): switches phase every K steps (e.g., 20) without learning.

### Baseline: how we compare fairly
- Baseline controller: simple timer that toggles NS↔EW every `--switch_period` steps (subject to `min_green`).
- Same environment, same `--N`, same `--max_steps`, and ideally the same `--seed` as training → apples-to-apples.
- Comprehensive baseline: `python -m src.generate_baseline ...` sweeps multiple periods and seeds and reports the best fixed-time performance with averages and standard deviations.

### Dashboard guide
Launch:
```bash
streamlit run src/dashboard.py
```
What you’ll see:
- Latest episode cards: Episode, Avg Queue (lower better), Throughput (higher better), Avg Travel Time (lower better)
- Training progress charts (Plotly): trends across episodes
- Per-intersection table + bar charts: spot bottlenecks quickly
- Baseline comparison: AI vs fixed-time with deltas and an overall verdict
- Summary/CSV/Final Averages: quick human-readable summary and data export

Interactive runs (no files needed):
- In the sidebar, use “Run a one-off simulation” to run either:
  - Baseline: choose switch period; see metrics instantly
  - Policy (greedy): uses `outputs/policy_final.pth`; see metrics instantly
- Use “Compare AI vs Fixed-Time (same seed, same N)” to run both back-to-back for your chosen inputs and see side-by-side comparison (deltas) immediately.

### Outputs
Training writes to `outputs/`:
- `metrics.json`: per-episode logs
- `metrics.csv`: same metrics in spreadsheet-friendly format
- `summary.txt`: latest human-readable summary (non-technical)
- `live_metrics.json`: last-episode summary for the dashboard
- `policy_final.pth`: final trained weights
- `final_report.json`: final averages and narrative
- `scenarios_report.json`: summary across multiple scenario runs

Baseline tools write:
- `baseline_metrics.json`: episode records (dashboard-ready)
- `baseline_detailed.json`: full statistical analysis across seeds and periods
- `baseline_summary.txt`: plain-English summary and best-strategy recommendation

### Reproducibility & performance
- Seeds are set for NumPy, Torch, and the environment to keep runs reproducible.
- Model is small (2×128) and the env is lightweight, so everything runs on CPU.
- If slow: reduce `--episodes` or `--N`; or shorten `--max_steps`.

### Demo script (3 minutes)
- Train a short run: `python -m src.train --episodes 5 --N 6`
- Generate a baseline: `python -m src.baseline --episodes 5 --N 6 --switch_period 20`
- Open the dashboard and show:
  - Latest metrics and curves
  - Per-intersection bottlenecks
  - Baseline comparison (explain deltas)
  - Final averages summary

### 📁 Project Structure

```
mini-traffic-marl/
├── src/                          # Source code
│   ├── agent.py                  # DQN agent and replay buffer
│   ├── env.py                    # Traffic environment simulation
│   ├── train.py                  # Training script and main loop
│   ├── baseline.py               # Fixed-time baseline controller
│   ├── dashboard.py              # Streamlit web dashboard
│   ├── scenarios.py              # Multi-scenario evaluation
│   └── generate_baseline.py      # Comprehensive baseline generation
├── outputs/                      # Generated results (created after first run)
│   ├── metrics.json              # Per-episode training metrics
│   ├── metrics.csv               # Spreadsheet-friendly metrics
│   ├── live_metrics.json         # Latest episode for dashboard
│   ├── policy_final.pth          # Trained neural network weights
│   ├── summary.txt               # Human-readable summary
│   └── final_report.json         # Comprehensive final report
├── requirements.txt              # Python dependencies
├── install.bat                   # Windows installation script
├── install.ps1                   # PowerShell installation script
└── README.md                     # This file
```

### 🔧 Troubleshooting

#### Common Issues and Solutions

**Installation Problems:**
- **Python not found**: Ensure Python 3.8+ is installed and added to PATH
- **Permission errors**: Run installation scripts as administrator on Windows
- **Package conflicts**: Use a fresh virtual environment

**Runtime Issues:**
- **ImportError on dashboard**: Always run `streamlit run src/dashboard.py` from the project root directory
- **Dashboard shows no data**: Run at least one training episode first to generate `outputs/live_metrics.json`
- **Interactive runs fail**: Complete initial training to create `outputs/policy_final.pth`
- **Slow performance**: Reduce `--N` (intersections), `--episodes`, or `--max_steps` parameters

**Training Issues:**
- **Poor convergence**: Increase `--episodes` or adjust learning rate with `--lr`
- **Memory errors**: Reduce `--replay_capacity` or `--batch_size`
- **Inconsistent results**: Ensure you're using the same `--seed` for reproducible comparisons

#### Performance Optimization

**For faster training:**
```bash
# Reduce problem size
python -m src.train --episodes 20 --N 3 --max_steps 200

# Smaller replay buffer
python -m src.train --replay_capacity 5000 --batch_size 32
```

**For better results:**
```bash
# Longer training with more exploration
python -m src.train --episodes 100 --epsilon_decay_steps 10000

# Larger network size (modify agent.py)
# Increase hidden layer size from 128 to 256
```

### 🎯 Use Cases and Applications

This project is ideal for:

- **Research & Education**: Understanding multi-agent reinforcement learning concepts
- **Traffic Engineering**: Prototyping adaptive traffic control algorithms  
- **Algorithm Development**: Testing new MARL approaches in a controlled environment
- **Benchmarking**: Comparing different traffic control strategies
- **Academic Projects**: Demonstrating AI applications in urban planning

### 🤝 Contributing

We welcome contributions! Here are some ways to help:

- **Bug Reports**: Submit issues with detailed reproduction steps
- **Feature Requests**: Suggest new functionality or improvements
- **Code Contributions**: Submit pull requests with enhancements
- **Documentation**: Improve README, add code comments, or create tutorials
- **Testing**: Help test on different platforms and configurations

### 📄 License

This project is open source. Please check the license file for specific terms and conditions.

### 🙏 Acknowledgments

- Built with PyTorch for deep learning capabilities
- Streamlit for the interactive dashboard
- Inspired by real-world traffic optimization challenges

### Version control and environment notes
- The repo is configured to ignore heavy/derived files via `.gitignore`.
- The `.venv/` directory is intentionally ignored; do not commit your virtual environment.
- Typical first-time Git steps:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
If you accidentally staged `.venv/` before the `.gitignore` was added, run:
```bash
git reset
git rm -r --cached .venv
```
Then add/commit again.
