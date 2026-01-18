# How to Run the Traffic MARL Project

This guide explains how to run all the different components of the project.

## 📋 Prerequisites

1. **Python 3.9 or higher** (Python 3.10-3.12 recommended)
2. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Quick Start (Recommended: Dashboard)

The easiest way to run the project is using the interactive dashboard:

```bash
streamlit run src/dashboard.py
```

This will:
- Open a web browser with the dashboard (usually at `http://localhost:8501`)
- Allow you to configure and run experiments through a user-friendly interface
- Show real-time visualizations and results

### Using the Dashboard:

1. **Configure parameters** in the sidebar:
   - **Intersections (N)**: Number of traffic intersections (default: 6)
   - **Seed**: Random seed for reproducibility (default: 42)
   - **Episodes**: Number of training episodes (default: 10)

2. **Click "Run Simulation"**:
   - The dashboard will run a fixed-time baseline first
   - Then train the AI controller
   - Results will appear automatically

3. **View results**:
   - Learning curves (queue length, throughput, travel time)
   - Comparison between AI and baseline
   - Per-intersection analysis
   - Training history table

---

## 🖥️ Command Line Options

### 1. Training Script (Main)

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

**Example - Quick training run:**

```bash
python -m src.train --episodes 10 --N 2 --seed 42
```

**Example - Longer training with more intersections:**

```bash
python -m src.train --episodes 100 --N 6 --max_steps 500 --seed 42
```

**Example - With neighbor observations enabled:**

```bash
python -m src.train --episodes 50 --N 4 --neighbor_obs --seed 42
```

---

### 2. Baseline Script (Fixed-Time Controller)

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

**Example:**

```bash
python -m src.baseline --episodes 20 --N 6 --switch_period 25 --seed 42
```

---

### 3. Generate Comprehensive Baseline

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

**All available parameters:**

```bash
python -m src.generate_baseline \
  --episodes 20                    # Episodes per run (default: 20)
  --N 6                            # Number of intersections (default: 6)
  --max_steps 300                  # Steps per episode (default: 300)
  --switch_periods "10,15,20,25,30" # Comma-separated switch periods (default: "10,15,20,25,30")
  --seeds "1,2,3,4,5"              # Comma-separated seeds (default: "1,2,3,4,5")
  --save_dir outputs               # Output directory (default: outputs)
```

---

### 4. Run Multiple Scenarios

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

**All available parameters:**

```bash
python -m src.scenarios \
  --total_episodes 100         # Total episodes across all scenarios (default: 100)
  --seeds "1,2,3,4,5"          # Comma-separated seeds (default: "1,2,3,4,5")
  --Ns "2,4,6"                 # Comma-separated intersection counts (default: "2,4,6")
```

---

## 📁 Output Files

All results are saved in the `outputs/` directory:

- **`metrics.json`**: Complete training history (JSON format)
- **`metrics.csv`**: Training history (CSV format, spreadsheet-friendly)
- **`live_metrics.json`**: Latest episode summary
- **`summary.txt`**: Human-readable training summary
- **`final_report.json`**: Aggregate statistics at end of training
- **`policy_final.pth`**: Saved neural network weights
- **`baseline_metrics.json`**: Baseline results (when running baseline scripts)

---

## 🎯 Recommended Workflow

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

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running from the project root directory:
```bash
cd C:\Users\Shrey\Documents\traffic-marl
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

---

## 💡 Tips

1. **Start small**: Begin with `--N 2` and `--episodes 10` to test quickly
2. **Use seeds**: Always specify `--seed` for reproducible results
3. **Check outputs**: Look at `outputs/summary.txt` for quick status updates
4. **Dashboard auto-refresh**: The dashboard refreshes every 5 seconds (configurable in sidebar)
5. **Compare fairly**: Use the same seed for baseline and AI training for fair comparison

---

## 📊 Example Commands Summary

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

---

For more details, see `README.md` and `IMPROVEMENTS.md`.


