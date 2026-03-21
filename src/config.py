
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

METRICS_JSON = OUTPUTS_DIR / "metrics.json"
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
LIVE_METRICS_JSON = OUTPUTS_DIR / "live_metrics.json"
SUMMARY_TXT = OUTPUTS_DIR / "summary.txt"
FINAL_REPORT_JSON = OUTPUTS_DIR / "final_report.json"
POLICY_PTH = OUTPUTS_DIR / "policy_final.pth"
BASELINE_METRICS_JSON = OUTPUTS_DIR / "baseline_metrics.json"
BASELINE_DETAILED_JSON = OUTPUTS_DIR / "baseline_detailed.json"
BASELINE_SUMMARY_TXT = OUTPUTS_DIR / "baseline_summary.txt"
SCENARIOS_REPORT_JSON = OUTPUTS_DIR / "scenarios_report.json"

ModelType = Literal["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"]

# SUMO and Mixed Traffic Configuration (IRC:106-1990 PCU Standards)
VEHICLE_PCU = {
    "two_wheeler": 0.5,      # Motorcycles, scooters
    "auto_rickshaw": 0.75,   # 3-wheelers
    "car": 1.0,              # Personal cars (baseline)
    "bus_truck": 3.0         # Buses and delivery vehicles
}

VEHICLE_MIX = {
    "two_wheeler": 0.60,     # 60%
    "auto_rickshaw": 0.16,   # 16%
    "car": 0.18,             # 18%
    "bus_truck": 0.06        # 6%
}

# Legacy format for backward compatibility
VEHICLE_CLASSES = {
    "TWO_WHEELER": {"service_rate": 3, "pcu": 0.5, "arrival_weight": 0.60, "vtype": "two_wheeler"},
    "AUTO_RICKSHAW": {"service_rate": 2, "pcu": 0.75, "arrival_weight": 0.16, "vtype": "auto_rickshaw"},
    "CAR": {"service_rate": 2, "pcu": 1.0, "arrival_weight": 0.18, "vtype": "car"},
    "BUS_TRUCK": {"service_rate": 1, "pcu": 3.0, "arrival_weight": 0.06, "vtype": "bus_truck"},
}

PEAK_HOUR_CONFIG = {
    "morning_peak": {"steps": (0, 1200), "NS_multiplier": 1.3, "EW_multiplier": 1.0},
    "evening_peak": {"steps": (2400, 3600), "NS_multiplier": 1.0, "EW_multiplier": 1.8},
    "uniform": {"steps": (0, 3600), "NS_multiplier": 1.0, "EW_multiplier": 1.0},
}

BASELINE_CONFIG = {
    "webster_lost_time": 3.0,
    "webster_saturation_flow": 1800.0,
    "fixed_time_cycle": 30,
    "max_pressure_threshold": 0.5,   # Tuned for actual traffic regime
}

SUMO_CONFIG = {
    "config_file": "sumo_config/pune_network.sumocfg",
    "step_length": 1.0,
    "min_green_steps": 5,
    "clearance_steps": 2,
    "lane_split_probability": 0.15,
    "lane_split_min_queue": 3,
}

SCENARIOS = ["uniform", "morning_peak", "evening_peak"]
PHASE_TYPES = ["NS_GREEN", "ALL_RED_CLEARANCE", "EW_GREEN"]
STATS_SEEDS = [1, 2, 3, 4, 5]
OBS_FEATURES_PER_AGENT = 24  # 15 self + 6 neighbor + 1 action_mask + 2 inflow

# Vehicle Injection Configuration
INJECTION_CONFIG = {
    # Base injection rate per route per step
    # Reduced from 0.25 to 0.19 to compensate for bus_truck (3.0 PCU) replacing pedestrian_group (0.0 PCU)
    # New avg PCU/vehicle: 0.78 (was 0.60), so 0.19 × 0.78 ≈ 0.25 × 0.60 (same load)
    "base_rate": 0.19,  # Balanced: moderate congestion without saturation
    # Two-wheeler turning bonus (Indian traffic behavior)
    "two_wheeler_turn_multiplier": 1.4,
}

# Federated Learning Configuration
FEDERATED_CONFIG = {
    "fed_interval": 20,    # T_fed: aggregate weights every 20 episodes
    "min_local_steps": 20,
    "aggregation": "fedavg",
    "track_communication_cost": True,
    "n_agents": 9,         # number of intersection edge nodes
    # FedAvg: simple uniform averaging across all 9 local models
    # No differential weighting — all intersections contribute equally
}

# Temporal Module Configuration
TEMPORAL_CONFIG = {
    "history_length": 5,
    "gru_hidden_dim": 32,
    "window": 5,      # number of past timesteps fed to GRU
    "hidden_dim": 64,  # GRU and GAT hidden dimension
    "gat_heads": 4,    # number of GAT attention heads
    "gru_layers": 1,   # single GRU layer sufficient for T=5
}

# Transformer Configuration
TRANSFORMER_CONFIG = {
    "heads": 4,
    "dropout": 0.1,
    "positional_encoding_dim": 16,
}

DEFAULT_ARRIVAL_RATE_NS = 0.8
DEFAULT_ARRIVAL_RATE_EW = 0.7
DEFAULT_MIN_GREEN = 10
DEFAULT_MAX_STEPS = 300
DEFAULT_STEP_LENGTH = 2.0
DEFAULT_DEPART_CAPACITY = 2

DEFAULT_REWARD_QUEUE_WEIGHT = -0.5
DEFAULT_REWARD_IMBALANCE_WEIGHT = -1.5
DEFAULT_REWARD_GOOD_SWITCH = 3.0
DEFAULT_REWARD_BAD_SWITCH = -2.0
DEFAULT_REWARD_IMBALANCE_THRESHOLD = 3.0
DEFAULT_REWARD_QUEUE_NORM = 10.0

EPSILON_CONFIG = {
    # Decay is computed over total STEPS not episodes
    # This is the mathematically correct approach per DQN convergence theory
    "start":            1.0,
    "end":              0.01,     # Reduced from 0.05 - less random noise late in training
    
    # Decay completes at this fraction of total training steps
    # Remaining steps use epsilon_end (pure exploitation)
    # Set to 0.85 to reach minimum at episode 85 of 100
    "decay_fraction":   0.85,
    
    # Graph models need more steps to learn spatial coordination
    # These multipliers stretch the decay window proportionally
    "model_complexity": {
        "DQN":          1.0,
        "GNN-DQN":      1.5,
        "GAT-DQN-Base": 1.5,
        "GAT-DQN":      1.7,
        "ST-GAT":       1.9,
        "Fed-ST-GAT":   2.0,
    },
}

REWARD_CONFIG = {
    # Pure pressure reward (Equation 3-4 from paper)
    "w_pressure":         1.0,    # Φ/η term (dominant signal)
    "reward_queue_norm":  50.0,   # η normalization constant (increased for 3.5x traffic)
    
    # Switching costs (paper specification)
    "w_switch_penalty":   0.01,   # λ_s switching penalty
    "w_clearance_penalty": 0.01,  # λ_c clearance penalty
    
    # Excessive green penalty (prevent starvation)
    "w_green_penalty":    0.0,    # Not in paper - disabled
    "max_green_steps":    30,     # Not enforced
    
    # Capacity penalty (prevent spillback)
    "w_capacity_penalty": 0.0,    # Not in paper - disabled
    "queue_threshold":    20.0,   # Not used
}

MODEL_GAMMA = {
    "DQN":          0.99,
    "GNN-DQN":      0.99,
    "GAT-DQN-Base": 0.99,
    "GAT-DQN":      0.99,
    "ST-GAT":       0.95,   # reduced to prevent Q-value divergence over 300 steps
    "Fed-ST-GAT":   0.95,   # same reason
}

# Minimum episodes per model for fair comparison
# Complex models need more exploration time
TRAINING_EPISODES = {
    "DQN":          100,
    "GNN-DQN":      120,
    "GAT-DQN-Base": 140,
    "GAT-DQN":      150,
    "ST-GAT":       200,
    "Fed-ST-GAT":   250,
}

PER_CONFIG = {
    # Prioritization exponent — 0=uniform sampling, 1=full priority
    "alpha":        0.6,
    
    # Importance sampling correction — anneals from beta_start to 1.0
    "beta_start":   0.4,
    "beta_end":     1.0,
    
    # Small constant preventing zero priority
    "epsilon":      1e-6,
}

@dataclass
class TrainingConfig:
    """Training hyperparameters for multiple RL architectures."""

    num_intersections: int = 9
    max_steps: int = DEFAULT_MAX_STEPS
    step_length: float = DEFAULT_STEP_LENGTH
    min_green: int = DEFAULT_MIN_GREEN
    arrival_rate_ns: float = DEFAULT_ARRIVAL_RATE_NS
    arrival_rate_ew: float = DEFAULT_ARRIVAL_RATE_EW
    depart_capacity: int = DEFAULT_DEPART_CAPACITY
    neighbor_obs: bool = False

    reward_queue_weight: float = DEFAULT_REWARD_QUEUE_WEIGHT
    reward_imbalance_weight: float = DEFAULT_REWARD_IMBALANCE_WEIGHT
    reward_good_switch: float = DEFAULT_REWARD_GOOD_SWITCH
    reward_bad_switch: float = DEFAULT_REWARD_BAD_SWITCH
    reward_imbalance_threshold: float = DEFAULT_REWARD_IMBALANCE_THRESHOLD
    reward_queue_norm: float = DEFAULT_REWARD_QUEUE_NORM

    model_type: ModelType = "DQN"
    

    comparison_mode: bool = False

    episodes: int = 100
    learning_rate: float = 0.001  # Base rate (ST-GAT uses 0.1x = 0.0001)
    batch_size: int = 256  # Paper specification
    gamma: float = 0.99  # Discount factor from paper
    replay_capacity: int = 100000  # Paper specification
    min_buffer_size: int = 1000  # Reduced from 2000 to start training earlier

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    update_target_steps: int = 200  # Hard target update for DQN/GNN-DQN (GAT models use tau=0.01 soft updates)

    # PPO/A2C parameters (legacy - not used, all models are DQN-style now)
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    a2c_value_coef: float = 0.5
    a2c_entropy_coef: float = 0.01

    gat_n_heads: int = 4
    gat_dropout: float = 0.1

    hidden_dim: int = 128
    grad_clip_norm: float = 1.0

    seed: int = 123
    save_dir: Path = OUTPUTS_DIR

@dataclass
class BaselineConfig:
    """Configuration for fixed-time baseline. Uses same env defaults as training for fair comparison."""

    episodes: int = 10
    num_intersections: int = 2
    max_steps: int = DEFAULT_MAX_STEPS
    switch_period: int = 20
    seed: int = 123
    save_dir: Path = OUTPUTS_DIR

    arrival_rate_ns: float = DEFAULT_ARRIVAL_RATE_NS
    arrival_rate_ew: float = DEFAULT_ARRIVAL_RATE_EW
    min_green: int = DEFAULT_MIN_GREEN
    step_length: float = DEFAULT_STEP_LENGTH
    depart_capacity: int = DEFAULT_DEPART_CAPACITY
