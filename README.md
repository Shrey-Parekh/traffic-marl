# Mini Traffic MARL

A lightweight queue-based multi-agent reinforcement learning (MARL) traffic simulator with a shared-policy DQN controller, a fixed-time baseline, rich metrics, and an interactive Streamlit dashboard. Runs efficiently on CPU-only machines.

## What’s Inside

- **Multi‑Agent simulator**: N intersections, each an agent sharing one neural network policy
- **Learning controller**: DQN with replay buffer, target network and epsilon‑greedy exploration
- **Fixed‑time baseline**: Deterministic switching every 20 steps for apples‑to‑apples comparison
- **Interactive dashboard**: One click to run baseline + learning, visualize results and history
- **Metrics & reports**: JSON history, CSV, human‑readable summary, final report and saved policy

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
- The baseline’s switch period is fixed at 20 steps
- Each episode runs for a fixed number of steps (shown in the UI)

Click “Run Simulation”. The dashboard will:
1) Simulate the fixed‑time baseline
2) Train the shared‑policy MARL controller for the requested episodes
3) Update metrics files and refresh visualizations automatically

## How the System Works

- Environment (`src/env.py`): queue‑based roads; every step simulates arrivals, departures and light phase actions
- Agent (`src/agent.py`): DQN + replay buffer; shared weights across all intersections
- Training (`src/train.py`): runs episodes, optimizes DQN, and writes metrics after every episode
- Dashboard (`src/dashboard.py`): orchestrates baseline + training runs and renders:
  - History table sourced from `outputs/metrics.json` (latest first)
  - Learning curves (avg queue, throughput, travel time)
  - Per‑intersection analysis from the latest episode

## Outputs (in `outputs/`)

- `metrics.json`: full per‑episode history used by the dashboard’s Run History
- `metrics.csv`: spreadsheet‑friendly export of the same
- `live_metrics.json`: latest episode summary (ingested by the dashboard)
- `summary.txt`: rolling human‑readable status (updates each episode)
- `final_report.json`: aggregate stats at the end of a run
- `policy_final.pth`: saved DQN weights

The project preserves previous results; new runs append to the history rather than wiping it.

## Run History – Column Guide

- **Episode**: 1‑based episode index
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
│   ├── baseline.py       # (reference) fixed-time controller utilities
│   ├── scenarios.py      # Batch runner for multiple seeds/sizes
│   └── generate_baseline.py
├── outputs/              # Results written here after runs
├── requirements.txt
└── README.md
```

## Tips & Troubleshooting

- On some Windows consoles, non‑ASCII symbols can cause encoding errors; the trainer prints ASCII‑safe logs.
- If the dashboard doesn’t update instantly after training, wait a second; it auto‑refreshes.
- For faster experimentation, reduce `N` or the number of episodes.

## License

Open source; see the license file for details.

