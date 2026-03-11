# Quick Start Guide

## Heterogeneous Mixed-Traffic Signal Control using Graph Attention Networks
### A Case Study on Pune Urban Intersections

---

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. SUMO Installation (Required)

**Windows:**
```bash
# Download installer from: https://sumo.dlr.de/docs/Downloads.php
# Run the installer and add SUMO to PATH
# Verify installation:
sumo --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**
```bash
brew install sumo
```

**Verify SUMO is installed:**
```bash
sumo --version
netconvert --version
```

---

## Installation Steps

### Step 1: Clone/Navigate to Project
```bash
cd traffic-marl
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- torch (PyTorch for deep learning)
- numpy, scipy (numerical computing)
- streamlit (dashboard)
- plotly (visualization)
- traci, sumolib (SUMO Python interface)

### Step 3: Build SUMO Network
**This is critical - must be done before first run!**

```bash
netconvert -c sumo_config/pune_network.netccfg
```

**Expected output:**
```
Success.
```

This generates `pune_network.net.xml` from the node/edge definitions.

---

## Running the Project

### Option 1: Dashboard (Recommended for Beginners)

**Start the Streamlit dashboard:**
```bash
streamlit run src/dashboard.py
```

**What happens:**
1. Browser opens at `http://localhost:8501`
2. Dashboard shows 4 tabs:
   - **Training & Results**: Run RL training
   - **Traffic Analysis**: Visualize vehicle classes and queues
   - **Baselines Comparison**: Compare against traditional controllers
   - **Publication Stats**: Generate IEEE-ready tables

**Using the Dashboard:**
1. Check SUMO connection status (green dot = connected)
2. Configure parameters in sidebar:
   - Scenario: uniform / morning_peak / evening_peak
   - Seeds: Select 1 or more for statistical analysis
   - Episodes: 50 recommended for testing, 200+ for publication
   - Model: GAT-DQN (our novel contribution)
3. Click "🚀 Run Simulation"
4. Watch live training progress
5. View results in tabs

---

### Option 2: Command Line Training

**Single seed, uniform traffic:**
```bash
python src/train.py --model_type GAT-DQN --episodes 50 --scenario uniform --seed 42
```

**Multi-seed for statistics (publication-ready):**
```bash
python src/train.py --model_type GAT-DQN --episodes 100 --scenario morning_peak --seeds "1,2,3,4,5"
```

**All available arguments:**
```bash
python src/train.py --help
```

**Key arguments:**
- `--model_type`: DQN | GNN-DQN | PPO-GNN | GAT-DQN | GNN-A2C
- `--episodes`: Number of training episodes (default: 50)
- `--scenario`: uniform | morning_peak | evening_peak
- `--seeds`: Comma-separated seeds for multi-seed training
- `--max_steps`: SUMO simulation time in seconds (default: 600)
- `--N`: Number of intersections (fixed at 9 for 3×3 grid)

---

### Option 3: Run Baselines Only

**Test baseline controllers:**
```bash
python src/baseline.py --episodes 10 --scenario uniform
```

**Baselines included:**
- **FixedTime**: Simple cyclic controller (weakest)
- **Webster**: Adaptive timing using Webster's formula
- **MaxPressure**: Reactive pressure-based (strongest baseline)

---

## Expected Training Time

**Per Episode (SUMO simulation):**
- ~2-3 minutes on modern CPU
- ~1-2 minutes with GPU acceleration

**Full Training Run:**
- 50 episodes, 1 seed: ~2 hours
- 100 episodes, 5 seeds: ~10-15 hours
- Recommended: Start with 10 episodes for testing

**Tip:** Use `--episodes 10` for quick testing, then scale up for publication results.

---

## Output Files

All results saved to `outputs/` directory:

### Training Outputs
- `metrics.json` - Episode-by-episode metrics
- `metrics.csv` - CSV format for analysis
- `final_report.json` - Final performance summary
- `live_metrics.json` - Real-time training progress (dashboard)
- `model_*.pth` - Saved model weights

### Multi-Seed Outputs
- `metrics_all_seeds.json` - Combined results from all seeds
- `statistical_summary.json` - Mean, std, 95% CI for each metric

### Baseline Outputs
- `baseline_metrics.json` - Performance of FixedTime, Webster, MaxPressure

---

## Troubleshooting

### Issue: "SUMO not found"
**Solution:**
```bash
# Verify SUMO is in PATH
sumo --version

# If not found, add to PATH:
# Windows: Add C:\Program Files (x86)\Eclipse\Sumo\bin to PATH
# Linux/Mac: export PATH=$PATH:/usr/share/sumo/bin
```

### Issue: "pune_network.net.xml not found"
**Solution:**
```bash
# Build the network first
netconvert -c sumo_config/pune_network.netccfg
```

### Issue: "TraCI connection failed"
**Solution:**
- Close any running SUMO instances
- Check if port 8813 is available
- Restart training (uses random port on retry)

### Issue: Training is very slow
**Solutions:**
- Reduce `--max_steps` (e.g., 300 instead of 600)
- Reduce `--episodes` for testing
- Use simpler model (DQN instead of GAT-DQN)
- Check CPU usage (SUMO is CPU-intensive)

### Issue: Dashboard not updating
**Solution:**
- Check "Auto-refresh" is enabled
- Increase refresh interval to 30s
- Manually refresh browser

---

## Quick Test Run

**Verify everything works (5 minutes):**

```bash
# 1. Build network
netconvert -c sumo_config/pune_network.netccfg

# 2. Quick training test (2 episodes)
python src/train.py --model_type GAT-DQN --episodes 2 --scenario uniform --seed 42

# 3. Test baselines
python src/baseline.py --episodes 2 --scenario uniform

# 4. Launch dashboard
streamlit run src/dashboard.py
```

**Expected result:**
- Training completes in ~5 minutes
- `outputs/` contains metrics files
- Dashboard shows results in all 4 tabs

---

## Project Structure

```
traffic-marl/
├── src/
│   ├── agent.py           # RL models (GAT-DQN with VehicleClassAttention)
│   ├── baseline.py        # Traditional controllers
│   ├── config.py          # Configuration constants
│   ├── dashboard.py       # Streamlit dashboard (4 tabs)
│   ├── env_sumo.py        # SUMO environment wrapper
│   └── train.py           # Training script
├── sumo_config/
│   ├── pune_network.nod.xml    # Node definitions
│   ├── pune_network.edg.xml    # Edge definitions
│   ├── pune_network.typ.xml    # Road types
│   ├── pune_network.netccfg    # Network config
│   ├── pune_network.sumocfg    # SUMO config
│   └── pune_vehicles.rou.xml   # Vehicle routes
├── outputs/               # Results directory (auto-created)
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

---

## Research Features

### Novel Contributions
1. **VehicleClassAttention Module**: Learns attention weights for 4 vehicle classes
   - Two-wheelers (PCU: 0.5)
   - Auto-rickshaws (PCU: 0.75)
   - Cars (PCU: 1.0)
   - Pedestrian groups (PCU: 0.0)

2. **Mixed Traffic Modeling**: Non-lane-based flow with lane-splitting behavior

3. **Peak Hour Asymmetry**: Morning/evening peak scenarios with directional bias

### Metrics Tracked
- **Queue Length**: Raw count and PCU-weighted
- **Throughput**: Vehicles served per episode
- **Travel Time**: Average time in network
- **Episode Reward**: Cumulative RL reward

---

## Next Steps

1. **Quick Test**: Run 10-episode training to verify setup
2. **Baseline Comparison**: Run baselines to establish benchmarks
3. **Full Training**: 100+ episodes with 5 seeds for publication
4. **Analysis**: Use dashboard Tab 2 for traffic patterns
5. **Publication**: Generate LaTeX tables in Tab 4

---

## Support

**Documentation:**
- `PHASE1_COMPLETE.md` - SUMO network setup
- `PHASE2_COMPLETE.md` - Agent architecture and baselines
- `PHASE3_DASHBOARD_SPEC.md` - Dashboard specification
- `PROJECT_EXPLANATION.md` - Research context

**Common Issues:**
- Check SUMO installation: `sumo --version`
- Verify network built: `ls sumo_config/pune_network.net.xml`
- Check Python version: `python --version` (need 3.8+)

---

**Ready to start? Run the Quick Test above!** 🚀
