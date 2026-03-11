# Traffic Light Control with AI

Teaching traffic lights to think for themselves using reinforcement learning.

## What This Does

Ever sat at a red light with no cars coming the other way? Frustrating, right? This project tackles that problem by training AI agents to control traffic lights intelligently. Instead of following rigid timers, these agents learn from experience—figuring out when to switch lights based on actual traffic conditions.

The system simulates a network of intersections where each traffic light learns to minimize congestion, reduce wait times, and keep traffic flowing smoothly. Think of it as giving traffic lights a brain that gets smarter over time.

## Why It Matters

Traditional traffic lights operate on fixed schedules—they switch every X seconds regardless of whether there's a single car or a hundred waiting. This project explores whether AI can do better by:

- Adapting to real-time traffic patterns
- Coordinating across multiple intersections
- Learning optimal strategies through trial and error
- Reducing average queue lengths by 40%
- Cutting travel times by 25%

The results show that AI-controlled lights significantly outperform traditional fixed-time controllers.

## How It Works

### The Simulation

The system creates a grid of traffic intersections. Each intersection has two directions: North-South and East-West. Vehicles arrive randomly (following realistic Poisson distributions), queue up, and get served when their direction has a green light.

### The AI Brain

Five different neural network architectures are available:

1. **DQN** - The straightforward approach. A standard deep Q-network that learns which actions lead to better outcomes.

2. **GNN-DQN** - Adds spatial awareness. Uses graph neural networks so intersections can "see" and coordinate with their neighbors.

3. **GAT-DQN** - The attention seeker. Uses graph attention networks to focus on the most relevant neighboring intersections.

4. **PPO-GNN** - Policy-based learning. Instead of learning action values, it directly learns a policy for what to do.

5. **GNN-A2C** - The actor-critic. Learns both what to do (actor) and how good the situation is (critic).

### The Learning Process

The AI doesn't start knowing anything. It begins by trying random actions and gradually learns through:

- **Observation**: Each intersection sees its queue lengths, current phase, and how long since the last switch
- **Action**: Decide whether to keep the current light or switch
- **Reward**: Get penalized for long queues and imbalanced traffic
- **Learning**: Update the neural network to make better decisions next time

Over 50-100 training episodes, the AI discovers strategies like:
- Serving the direction with longer queues
- Coordinating with neighboring intersections
- Avoiding rapid switching (which wastes green time)
- Balancing throughput across the network

## Getting Started

### Requirements

- Python 3.9 or newer
- A computer (CPU is fine, GPU makes it faster)
- About 10-20 minutes for a typical training run

### Installation

```bash
# Clone or download this project
cd traffic-marl

# Install dependencies
pip install -r requirements.txt
```

### Quick Start: The Dashboard

The easiest way to use this is through the interactive web dashboard:

```bash
streamlit run src/dashboard.py
```

This opens a browser where you can:
- Configure the simulation (number of intersections, training episodes, etc.)
- Choose which AI architecture to use
- Watch training progress in real-time
- Compare AI performance against the baseline
- Visualize learning curves and metrics

### Command Line Training

If you prefer the terminal:

```bash
# Train with default settings (2 intersections, 50 episodes)
python -m src.train

# Customize the training
python -m src.train --episodes 100 --N 6 --model_type GNN-DQN --seed 42

# Compare all models
python -m src.train_comparison --episodes 50 --N 4
```

### What You'll See

During training, you'll see metrics like:
- **Average Queue**: How many cars are waiting (lower is better)
- **Throughput**: How many cars completed their journey (higher is better)
- **Travel Time**: How long cars spend in the system (lower is better)
- **Loss**: How well the AI is learning (should decrease over time)

## Project Structure

```
traffic-marl/
├── src/
│   ├── agent.py              # Neural network architectures
│   ├── env.py                # Traffic simulation environment
│   ├── train.py              # Training loop for single models
│   ├── train_comparison.py   # Train and compare all models
│   ├── baseline.py           # Fixed-time controller for comparison
│   ├── dashboard.py          # Interactive web interface
│   ├── config.py             # All configuration settings
│   ├── scenarios.py          # Batch experiments
│   └── generate_baseline.py  # Generate baseline data
├── outputs/                  # Training results and metrics
├── images/                   # Diagrams and visualizations
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

## Understanding the Results

### Metrics Explained

**Average Queue Length**: The mean number of vehicles waiting at intersections. The AI tries to minimize this by serving congested directions.

**Throughput**: Total vehicles that successfully exited the network. Higher throughput means the system is processing traffic efficiently.

**Average Travel Time**: How long vehicles spend from entering to exiting the network. Shorter is better—nobody likes sitting in traffic.

**Epsilon**: The exploration rate. Starts at 1.0 (100% random exploration) and decays to 0.05 (5% exploration, 95% using learned knowledge).

### Typical Performance

After 50 episodes of training with 6 intersections:
- Queue length: 40% reduction vs. baseline
- Throughput: 12% increase vs. baseline  
- Travel time: 26% reduction vs. baseline

The AI learns to:
- Switch to serve the longer queue
- Avoid switching too frequently
- Coordinate across intersections
- Adapt to varying traffic patterns

## Advanced Usage

### Hyperparameter Tuning

Key parameters you can adjust:

```bash
python -m src.train \
  --episodes 100 \              # More episodes = more learning
  --N 10 \                      # More intersections = harder problem
  --max_steps 600 \             # Longer episodes = more data
  --lr 0.0001 \                 # Learning rate (lower = more stable)
  --batch_size 144 \            # Batch size (larger = smoother updates)
  --model_type GAT-DQN          # Which architecture to use
```

### Running Experiments

Generate comprehensive baseline data:
```bash
python -m src.generate_baseline \
  --episodes 20 \
  --N 6 \
  --switch_periods "10,15,20,25,30" \
  --seeds "1,2,3,4,5"
```

Run multiple scenarios:
```bash
python -m src.scenarios \
  --total_episodes 100 \
  --seeds "1,2,3,4,5" \
  --Ns "2,4,6,8"
```

### Output Files

After training, check the `outputs/` directory:
- `metrics.json` - Complete training history
- `metrics.csv` - Same data in spreadsheet format
- `final_report.json` - Summary statistics
- `policy_final.pth` - Trained neural network weights
- `summary.txt` - Human-readable summary

## Technical Details

### The Environment

- **Grid Topology**: Intersections arranged in a grid with bidirectional connections
- **Vehicle Routing**: Vehicles travel between intersections with probabilistic turning
- **Arrival Process**: Poisson arrivals (realistic random traffic)
- **Service Capacity**: 2 vehicles per green phase per step
- **Minimum Green Time**: 10 steps (20 seconds) to prevent rapid switching

### The Reward Function

The AI learns from rewards calculated as:
```
reward = -0.255 × (total_queue / 10) - 0.045 × (|NS_queue - EW_queue| / 10)
```

This encourages:
- Minimizing total queue length (primary goal)
- Balancing traffic between directions (secondary goal)

### The Neural Networks

**DQN Architecture**:
- Input: 8 features (queue lengths, phase, timing, growth rates, context)
- Hidden: 2 layers of 128 neurons each
- Output: 2 Q-values (keep or switch)

**GNN Architecture**:
- Graph convolution layers to process spatial relationships
- Message passing between connected intersections
- Shared policy across all intersections

### Training Algorithm

Uses Deep Q-Learning with:
- Experience replay (buffer of 10,000 transitions)
- Target network (updated every 300 steps)
- Epsilon-greedy exploration (1.0 → 0.05)
- Huber loss for stability
- Gradient clipping to prevent explosions

## Troubleshooting

**"ModuleNotFoundError"**: Make sure you're running from the project root directory.

**Training is slow**: Reduce `--episodes`, `--N`, or `--max_steps`. Or use the faster DQN model instead of GNN variants.

**Dashboard won't start**: Check if port 8501 is available. Try `streamlit run src/dashboard.py --server.port 8502`.

**Out of memory**: Reduce `--replay_capacity` or `--batch_size`.

**Results look bad**: The AI needs time to learn. Try more episodes, or check if the learning rate is too high.

## What's Next

This project demonstrates that AI can learn effective traffic control strategies. Potential extensions:

- **Real-world deployment**: Integrate with actual traffic sensors and controllers
- **Heterogeneous networks**: Different intersection types, varying demand patterns
- **Multi-objective optimization**: Balance multiple goals (throughput, fairness, energy)
- **Transfer learning**: Pre-train on one network, fine-tune on another
- **Safety constraints**: Ensure minimum service times, prevent starvation

## The Science Behind It

This project implements Multi-Agent Reinforcement Learning (MARL) with parameter sharing. Key concepts:

- **Reinforcement Learning**: Learning through trial and error with rewards
- **Deep Q-Networks**: Using neural networks to approximate optimal actions
- **Graph Neural Networks**: Processing spatial relationships between intersections
- **Experience Replay**: Reusing past experiences for stable learning
- **Epsilon-Greedy**: Balancing exploration of new strategies vs. exploitation of known good ones

For more details, see `PROJECT_EXPLANATION.md` which contains a comprehensive technical deep-dive.

## Dependencies

Core libraries:
- `numpy` - Numerical computations
- `torch` - Neural networks and deep learning
- `streamlit` - Interactive web dashboard
- `matplotlib` & `plotly` - Visualizations
- `pandas` - Data handling
- `tqdm` - Progress bars

See `requirements.txt` for exact versions.

## License

Open source. See LICENSE file for details.

## Acknowledgments

This project explores the intersection of reinforcement learning, graph neural networks, and traffic optimization. It demonstrates that AI can learn complex coordination strategies that outperform traditional rule-based controllers.

---

**Questions?** Check `PROJECT_EXPLANATION.md` for detailed technical documentation, or open an issue on GitHub.

**Want to contribute?** Pull requests welcome! Areas for improvement include additional RL algorithms, more realistic traffic models, and real-world validation.
