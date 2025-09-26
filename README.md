## Mini Traffic MARL

Lightweight queue-based multi-agent traffic simulator with a shared-policy DQN controller, fixed-time baseline, evaluation, and a Streamlit live dashboard. Designed to run CPU-only on a normal laptop with no internet at runtime.

### Features
- Queue-based simulator with N intersections, two phases (NS/EW)
- Poisson arrivals, fixed departure capacity, simple routing (EW -> next NS; NS exits)
- Shared-policy DQN across agents (parameter sharing)
- Fixed-time baseline and comprehensive baseline generator
- Evaluation and metrics logging (JSON, CSV, summary, final report)
- Streamlit dashboard with explainer, per-intersection tables, baseline comparison, and interactive runs

### Requirements
- Python 3.8+
- Libraries: numpy, torch (CPU), tqdm, streamlit, matplotlib, pandas, plotly

Install dependencies:
```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd)
.\.venv\Scripts\activate.bat
# macOS/Linux (bash/zsh)
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Quickstart

Train (short demo):
```bash
python -m src.train --episodes 3
```

Run fixed-time baseline:
```bash
python -m src.baseline --episodes 3
```

Run multi-scenario sweep (quick demo):
```bash
python -m src.scenarios --total_episodes 30 --seeds 1,2 --Ns 2,4
```

Launch dashboard:
```bash
streamlit run src/dashboard.py
```

Run 100+ diverse episodes (seeds and sizes):
```bash
python -m src.scenarios --total_episodes 100 --seeds 1,2,3,4,5 --Ns 2,4,6
```

Generate comprehensive baseline data:
```bash
python -m src.generate_baseline --episodes 20 --N 6 --switch_periods 10,15,20,25,30 --seeds 1,2,3,4,5
```


### Configuration
See `python -m src.train -h` for all hyperparameters, including:
- `--episodes`, `--N`, `--max_steps`, `--lr`, `--batch_size`, `--gamma`, `--replay_capacity`
- `--epsilon_start`, `--epsilon_end`, `--epsilon_decay_steps`, `--update_target_steps`
- `--seed`, `--neighbor_obs`

Multi-intersection and heterogeneous demand:
- Use `--N` to increase intersections (e.g., `--N 6` for a corridor).
- The env supports per-intersection arrival rates via `EnvConfig.arrival_rate_*_per_int` in code. You can customize before instantiating the env if you need varying demand per node.

### How the system works (plain English)
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

### Troubleshooting
- ImportError on dashboard: run `streamlit run src/dashboard.py` from the project root.
- Dashboard empty: run at least one training episode to create `outputs/live_metrics.json`.
- Policy (greedy) interactive run fails: train first so `outputs/policy_final.pth` exists.
- Slow training: decrease `--N`, `--episodes`, or `--max_steps`.

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
