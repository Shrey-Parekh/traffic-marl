````markdown
# Mini Traffic MARL

A lightweight queue-based multi-agent reinforcement learning (MARL) traffic simulator featuring a shared-policy Deep Q-Network (DQN) controller, fixed-time baseline comparison, comprehensive evaluation metrics, and an interactive Streamlit dashboard. Designed to run efficiently on CPU-only systems without internet connectivity.

## Key Features

- **Multi-Agent Traffic Simulation**: Queue-based simulator with N intersections, each having two phases (North-South/East-West)
- **Intelligent Traffic Control**: Shared-policy DQN across all traffic light agents with parameter sharing for efficient learning
- **Realistic Traffic Modeling**: Poisson arrival distributions, fixed departure capacity, and simple routing logic
- **Comprehensive Evaluation**: Built-in fixed-time baseline controller for fair comparison
- **Interactive Dashboard**: Input parameters directly in the dashboard, which will automatically run training, baseline evaluation, and comparisons
- **Rich Analytics**: Detailed metrics logging (JSON, CSV, reports) with per-intersection analysis
- **Reproducible Results**: Seed-based reproducibility for consistent experimental results

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: CPU-only (no GPU required)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: ~500MB for dependencies and outputs

## Installation

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
````

## Quick Start

Simply launch the dashboard:

```bash
streamlit run src/dashboard.py
```

### Dashboard Capabilities

* Run **training** and **baseline** directly from the interface by providing parameters in the sidebar
* Automatically compare AI vs fixed-time baseline with side-by-side results
* Real-time charts for queues, throughput, and travel times
* Per-intersection performance breakdown
* Export metrics and summary reports

No manual command-line execution is required anymore; everything is handled within the dashboard.

## Outputs

Training and evaluation results are saved under the `outputs/` directory:

* `metrics.json`: per-episode logs
* `metrics.csv`: spreadsheet-friendly metrics
* `summary.txt`: human-readable summary
* `live_metrics.json`: last-episode summary for the dashboard
* `policy_final.pth`: trained neural network weights
* `final_report.json`: final averages and narrative

Baseline runs generate:

* `baseline_metrics.json`: episode records (dashboard-ready)
* `baseline_summary.txt`: plain-English summary

## How It Works

* The system simulates a line of intersections with vehicles arriving randomly.
* Each intersection is an agent but all share the same neural network (shared policy).
* At each step, the policy chooses whether to keep or switch the light phase.
* Rewards are based on reducing waiting queues.
* The baseline controller switches phases at fixed intervals for comparison.
* The dashboard automates running both AI and baseline under identical conditions and presents results side by side.

## Project Structure

```
mini-traffic-marl/
├── src/                          # Source code
│   ├── agent.py                  # DQN agent and replay buffer
│   ├── env.py                    # Traffic environment simulation
│   ├── train.py                  # Training logic
│   ├── baseline.py               # Fixed-time baseline
│   ├── dashboard.py              # Streamlit dashboard
│   ├── scenarios.py              # Multi-scenario evaluation
│   └── generate_baseline.py      # Extended baseline generator
├── outputs/                      # Generated results (created after first run)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Troubleshooting

**Common Issues:**

* **Python not found**: Ensure Python 3.8+ is installed and added to PATH
* **Dashboard not showing results**: Run at least one training episode through the dashboard
* **Interactive comparison fails**: Ensure both training and baseline runs were completed first

**Performance tips:**

* Reduce `N` (intersections) or number of episodes for faster runs
* Increase episodes for better AI performance

## Use Cases

* **Research & Education**: Learn and demonstrate multi-agent reinforcement learning
* **Traffic Engineering**: Prototype adaptive traffic control systems
* **Benchmarking**: Compare MARL vs fixed-time strategies
* **Academic Projects**: Practical demonstration of AI in smart city applications

## Contributing

We welcome contributions through issues, pull requests, or documentation improvements.

## License

This project is open source. Please check the license file for details.

```

Do you also want me to **strip out all the old CLI examples** (like `python -m src.train` etc.) from the file entirely, or keep them in a separate section as optional advanced usage?
```
