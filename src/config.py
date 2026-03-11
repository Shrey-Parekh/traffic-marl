
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

# SUMO and Mixed Traffic Configuration
VEHICLE_CLASSES = {
    "TWO_WHEELER": {"service_rate": 3, "pcu": 0.5, "arrival_weight": 0.60, "vtype": "two_wheeler"},
    "AUTO_RICKSHAW": {"service_rate": 2, "pcu": 0.75, "arrival_weight": 0.15, "vtype": "auto_rickshaw"},
    "CAR": {"service_rate": 2, "pcu": 1.0, "arrival_weight": 0.20, "vtype": "car"},
    "PEDESTRIAN_GROUP": {"service_rate": 4, "pcu": 0.0, "arrival_weight": 0.05, "vtype": "pedestrian_group"},
}

PEAK_HOUR_CONFIG = {
    "morning_peak": {"steps": (0, 1200), "NS_multiplier": 1.8, "EW_multiplier": 1.0},
    "evening_peak": {"steps": (2400, 3600), "NS_multiplier": 1.0, "EW_multiplier": 1.8},
    "uniform": {"steps": (0, 3600), "NS_multiplier": 1.0, "EW_multiplier": 1.0},
}

BASELINE_CONFIG = {
    "max_pressure_threshold": 3.0,
    "webster_lost_time": 4.0,
    "webster_saturation_flow": 0.5,
    "fixed_time_cycle": 30,
}

SUMO_CONFIG = {
    "config_file": "sumo_config/pune_network.sumocfg",
    "step_length": 1.0,
    "min_green_steps": 10,
    "clearance_steps": 2,
    "lane_split_probability": 0.15,
    "lane_split_min_queue": 3,
}

SCENARIOS = ["uniform", "morning_peak", "evening_peak"]
PHASE_TYPES = ["NS_GREEN", "ALL_RED_CLEARANCE", "EW_GREEN"]
STATS_SEEDS = [1, 2, 3, 4, 5]
OBS_FEATURES_PER_AGENT = 15

# Federated Learning Configuration
FEDERATED_CONFIG = {
    "fed_round_interval": 50,
    "min_local_steps": 20,
    "aggregation": "fedavg",
    "track_communication_cost": True,
}

# Temporal Module Configuration
TEMPORAL_CONFIG = {
    "history_length": 5,
    "gru_hidden_dim": 32,
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
DEFAULT_MAX_STEPS = 600
DEFAULT_STEP_LENGTH = 2.0
DEFAULT_DEPART_CAPACITY = 2

DEFAULT_REWARD_QUEUE_WEIGHT = -0.5
DEFAULT_REWARD_IMBALANCE_WEIGHT = -1.5
DEFAULT_REWARD_GOOD_SWITCH = 3.0
DEFAULT_REWARD_BAD_SWITCH = -2.0
DEFAULT_REWARD_IMBALANCE_THRESHOLD = 3.0
DEFAULT_REWARD_QUEUE_NORM = 10.0

@dataclass
class TrainingConfig:
    """Training hyperparameters for multiple RL architectures."""

    num_intersections: int = 2
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
    learning_rate: float = 0.0001
    batch_size: int = 144
    gamma: float = 0.99
    replay_capacity: int = 10000
    min_buffer_size: int = 500

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    epsilon_warmup_fraction: float = 0.03
    epsilon_decay_power: float = 2.0
    update_target_steps: int = 300

    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95

    a2c_value_coef: float = 0.5
    a2c_entropy_coef: float = 0.01
    a2c_max_grad_norm: float = 0.5

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
