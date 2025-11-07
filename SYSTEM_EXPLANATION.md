# Complete Guide to Mini Traffic MARL System

Let me explain this entire Multi-Agent Reinforcement Learning traffic control system in simple terms, covering every component and how they work together.

## 🎯 **What This System Does**

Imagine you have multiple traffic intersections that need to coordinate their traffic lights to reduce congestion. Instead of using fixed timing (like most real traffic lights), this system uses **artificial intelligence** to learn the best timing through trial and error.

## 🏗️ **System Architecture Overview**

```
User Input → Dashboard → Training Process → AI Learning → Results Display
     ↓              ↓            ↓              ↓            ↓
[Settings] → [Orchestration] → [Environment] → [Neural Net] → [Visualization]
```

## 📁 **File Structure & Purpose**

### **Core Files:**
- **`src/env.py`** - The traffic world simulator
- **`src/agent.py`** - The AI brain (neural network)
- **`src/train.py`** - The learning process
- **`src/dashboard.py`** - The user interface

### **Support Files:**
- **`src/baseline.py`** - Simple fixed-time controller for comparison
- **`src/scenarios.py`** - Batch testing across different settings

## 🚦 **The Traffic Environment (`env.py`)**

### **What it simulates:**
- **N intersections** in a line (like a highway with multiple traffic lights)
- Each intersection has **2 phases**: North-South green OR East-West green
- **Vehicles arrive randomly** (Poisson distribution) at each intersection
- **Vehicles wait in queues** until the light turns green for their direction

### **Key Components:**
```python
class MiniTrafficEnv:
    def __init__(self, config):
        self.num_intersections = N  # Number of traffic lights
        self.max_steps = 300       # Episode length (15 seconds)
        self.ns_queues = []        # North-South waiting cars
        self.ew_queues = []        # East-West waiting cars
        self.phase = []            # Current light state (0=NS, 1=EW)
```

### **What happens each step:**
1. **New cars arrive** randomly at each intersection
2. **Cars leave** if their light is green (up to 2 cars per step)
3. **AI decides** whether to switch lights or keep current phase
4. **Reward calculated** based on total queue length (lower = better)

### **Observation Space (what AI sees):**
For each intersection, the AI sees:
- `ns_len`: Number of cars waiting North-South
- `ew_len`: Number of cars waiting East-West  
- `phase`: Current light state (0 or 1)
- `time_since_switch`: How long since last light change

## 🧠 **The AI Agent (`agent.py`)**

### **Deep Q-Network (DQN) Architecture:**
```python
class DQNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        self.fc1 = nn.Linear(obs_dim, 128)    # Input layer
        self.fc2 = nn.Linear(128, 128)        # Hidden layer
        self.fc3 = nn.Linear(128, n_actions)  # Output layer (2 actions)
```

### **Two Actions Available:**
- **Action 0**: Keep current light phase
- **Action 1**: Switch to other phase (if minimum time elapsed)

### **Learning Components:**
- **Replay Buffer**: Stores past experiences for learning
- **Target Network**: Stable target for learning (updated periodically)
- **Epsilon-Greedy**: Balances exploration vs exploitation

## 🎓 **The Learning Process (`train.py`)**

### **Episode Structure:**
Each episode = 300 steps = 15 seconds of simulation

### **Step-by-Step Learning:**
1. **Reset environment** - Clear all queues, set random arrivals
2. **For each step:**
   - Get current observation from each intersection
   - AI chooses action for each intersection
   - Environment processes actions and updates state
   - Calculate reward (negative queue length)
   - Store experience in replay buffer
3. **After episode:**
   - Train neural network on random batch from replay buffer
   - Update target network periodically
   - Save metrics to files

### **Key Learning Parameters:**
```python
learning_rate = 1e-3        # How fast AI learns
gamma = 0.99               # Future reward importance
epsilon_start = 1.0        # Initial exploration rate
epsilon_end = 0.05         # Final exploration rate
batch_size = 64            # Training batch size
```

## 📊 **The Dashboard (`dashboard.py`)**

### **User Interface Components:**

#### **Sidebar Controls:**
- **Intersections (N)**: Number of traffic lights (1-20)
- **Seed**: Random number for reproducible results
- **Episodes**: How many learning episodes to run (1-50)
- **Info**: Shows each episode takes ~15 seconds

#### **Main Display Areas:**

1. **Interactive Results** (when simulation runs):
   - **Overview Tab**: KPI cards, AI vs Baseline comparison table
   - **Intersections Tab**: Per-intersection analysis

2. **Episode Graphs** (learning curves):
   - Average Queue Length over episodes
   - Throughput over episodes  
   - Average Travel Time over episodes

3. **Run History Table**:
   - All previous simulation results
   - Sortable by episode, metrics, etc.

4. **Per-Intersection Analysis**:
   - Individual intersection performance
   - Bar charts showing queue/throughput per intersection

### **What Happens When You Click "Run Simulation":**

1. **Baseline Run** (5-10 seconds):
   - Simulates fixed-time controller (switches every 20 steps)
   - Records baseline performance metrics

2. **AI Training** (varies by episodes):
   - Shows progress bar with episode count
   - Runs Multi-Agent RL training
   - Updates files after each episode

3. **Results Display**:
   - Automatically refreshes dashboard
   - Shows comparison between AI and baseline
   - Updates all charts and tables

## 📈 **Output Files Explained**

### **`outputs/metrics.json`** - Complete History
```json
[
  {
    "episode": 0,
    "agents": 6,
    "epsilon": 0.95,
    "avg_queue": 3.25,
    "throughput": 152,
    "avg_travel_time": 13.47,
    "loss": 1.234,
    "updates": 45
  }
]
```

### **`outputs/live_metrics.json`** - Latest Episode
Same format as above, but only the most recent episode.

### **`outputs/summary.txt`** - Human Readable
```
Multi-Agent RL Traffic Control Training Summary
Episodes completed: 10/10
Agents (intersections): 6
Current epsilon: 0.05
Latest episode performance:
  - Avg Queue: 2.15 cars
  - Throughput: 178 vehicles
  - Avg Travel Time: 11.2s
  - Training Loss: 0.456
```

### **`outputs/policy_final.pth`** - Trained AI
PyTorch model file containing the learned neural network weights.

### **`outputs/final_report.json`** - Aggregate Stats
```json
{
  "episodes": 10,
  "average_metrics": {
    "avg_queue": 2.8,
    "throughput": 165.3,
    "avg_travel_time": 12.1
  },
  "final_episode": {...}
}
```

## 🔄 **Complete Data Flow**

### **1. User Input Processing:**
```
User sets: N=6, Seed=42, Episodes=10
↓
Dashboard validates inputs
↓
Creates environment config
```

### **2. Baseline Simulation:**
```
Create MiniTrafficEnv(N=6, seed=42)
↓
Run 300 steps with fixed switching (every 20 steps)
↓
Record: queue_length, throughput, travel_time
↓
Save baseline results
```

### **3. AI Training Process:**
```
Initialize DQN neural network
Initialize replay buffer
↓
For each episode (1 to 10):
  Reset environment
  For each step (1 to 300):
    Get observations from all intersections
    AI chooses actions (keep/switch) for each
    Environment processes actions
    Calculate rewards
    Store experience in replay buffer
    Train neural network on random batch
  Save episode metrics to files
↓
Save final trained model
```

### **4. Results Display:**
```
Load metrics from files
↓
Update dashboard visualizations
↓
Show comparison tables and charts
↓
Display per-intersection analysis
```

## 🎯 **Key Learning Concepts**

### **Multi-Agent Learning:**
- **Shared Policy**: All intersections use the same neural network
- **Parameter Sharing**: Efficient learning across multiple agents
- **Coordinated Actions**: Agents learn to work together

### **Reinforcement Learning:**
- **Reward Signal**: Negative queue length (minimize waiting)
- **Exploration**: Try random actions to discover better strategies
- **Exploitation**: Use learned knowledge to make good decisions
- **Experience Replay**: Learn from past experiences

### **Traffic Control:**
- **Adaptive Timing**: AI learns optimal light switching
- **Queue Management**: Minimize waiting time for vehicles
- **Throughput Optimization**: Maximize vehicles served
- **Coordination**: Multiple intersections work together

## 🚀 **Why This System Works**

1. **Realistic Simulation**: Models actual traffic behavior with random arrivals
2. **Shared Learning**: All agents benefit from each other's experiences
3. **Continuous Learning**: AI improves with each episode
4. **Fair Comparison**: Fixed baseline provides objective comparison
5. **Visual Feedback**: Dashboard shows learning progress in real-time

## 🔧 **Technical Implementation Details**

### **Neural Network Training:**
- **Loss Function**: Mean Squared Error between predicted and target Q-values
- **Optimizer**: Adam with learning rate 1e-3
- **Gradient Clipping**: Prevents exploding gradients
- **Target Network**: Updated every 200 steps for stability

### **Environment Dynamics:**
- **Arrival Rate**: Poisson distribution (0.3 vehicles/step per direction)
- **Departure Capacity**: 2 vehicles per green phase per step
- **Minimum Green Time**: 5 steps before switching allowed
- **Episode Length**: 300 steps (15 seconds at 2s/step)

### **Performance Metrics:**
- **Queue Length**: Average cars waiting per intersection
- **Throughput**: Total vehicles completing their journey
- **Travel Time**: Average time vehicles spend in system
- **Loss**: Neural network training error
- **Updates**: Number of gradient steps per episode

## 📋 **Run History Table - Column Explanations**

- **Episode**: 1-based episode index (which learning episode this was)
- **Agents**: Number of learning agents (equals intersections `N`)
- **Epsilon**: Exploration rate used for that episode (higher = more random actions)
- **Avg Queue**: Average cars waiting per intersection (lower is better)
- **Throughput**: Vehicles that finished during the episode (higher is better)
- **Avg Travel Time (s)**: Average time spent in the network (lower is better)
- **Loss**: Average DQN training loss during the episode (measure of learning progress)
- **Updates**: Gradient updates performed in the episode (how much learning happened)

## 🎮 **How to Use the System**

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Dashboard:**
   ```bash
   streamlit run src/dashboard.py
   ```

3. **Set Parameters:**
   - Choose number of intersections (1-20)
   - Set random seed for reproducibility
   - Select number of learning episodes (1-50)

4. **Run Simulation:**
   - Click "Run Simulation" button
   - Watch progress bar and status updates
   - View results automatically when complete

5. **Analyze Results:**
   - Check Run History table for all previous runs
   - View learning curves to see AI improvement
   - Compare AI vs Baseline performance
   - Examine per-intersection analysis

## 🔍 **Understanding the Results**

### **Good AI Performance Indicators:**
- **Decreasing Queue Length** over episodes
- **Increasing Throughput** over episodes
- **Decreasing Travel Time** over episodes
- **Lower Loss** values (but not always)
- **AI beats Baseline** in comparison

### **Learning Progress Signs:**
- **Epsilon decreases** (less random, more learned actions)
- **Consistent improvement** in metrics over episodes
- **Stable performance** in later episodes
- **Higher Updates** in early episodes (more learning)

This system demonstrates how AI can learn to control complex, multi-agent systems through trial and error, providing a practical example of reinforcement learning in action!
