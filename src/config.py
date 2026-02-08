"""Centralized configuration for Traffic MARL project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

# Project root and output directories
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# File paths
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

# Model architecture types
ModelType = Literal["DQN", "GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C", "Multi-Model Comparison"]


@dataclass
class TrainingConfig:
    """Training hyperparameters for multiple RL architectures."""

    # Environment
    num_intersections: int = 2  # REDUCED: 2 intersections for learnable credit assignment (was 10)
    max_steps: int = 600  # Increased from 300 to 600 (20 minutes instead of 10)
    step_length: float = 2.0
    min_green: int = 10  # Increased from 5 to 10 (20 seconds minimum green time)
    arrival_rate_ns: float = 0.8  # INCREASED: Higher load (160% of capacity) to create learning pressure
    arrival_rate_ew: float = 0.7  # INCREASED: Asymmetric high load (140% of capacity)
    depart_capacity: int = 2
    neighbor_obs: bool = False

    # Model Selection
    model_type: ModelType = "DQN"
    
    # Multi-model comparison settings
    comparison_mode: bool = False  # True when running multi-model comparison

    # Training
    episodes: int = 100  # INCREASED: More episodes needed for 3x reward scale and slower LR to converge
    learning_rate: float = 0.0001  # REDUCED: Lower LR for larger reward scale (3x) to prevent overshooting
    batch_size: int = 144  # INCREASED: Larger batches for more stable gradients with reward variance
    gamma: float = 0.99  # Perfect for long-term traffic planning
    replay_capacity: int = 10000  # Moderate buffer - balance between experience diversity and relevance
    min_buffer_size: int = 500  # REDUCED: Start training earlier (2 agents × 300 steps = 600 transitions per episode)

    # DQN-specific parameters
    epsilon_start: float = 1.0  # Start with full exploration
    epsilon_end: float = 0.05  # REDUCED: Lower final epsilon for more exploitation in limited episodes
    epsilon_decay_steps: int = 5000  # Legacy parameter (kept for backward compatibility)
    epsilon_warmup_fraction: float = 0.03  # REDUCED: Only 3 episodes warm-up (was 10%), faster exploitation
    epsilon_decay_power: float = 2.0  # Quadratic decay (faster than linear)
    update_target_steps: int = 300  # REDUCED: Update target network every 300 training steps (once per episode) for faster adaptation

    # PPO-specific parameters
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95

    # A2C-specific parameters
    a2c_value_coef: float = 0.5
    a2c_entropy_coef: float = 0.01
    a2c_max_grad_norm: float = 0.5

    # GAT-specific parameters
    gat_n_heads: int = 4  # Standard multi-head attention
    gat_dropout: float = 0.1  # Mild regularization to prevent overfitting

    # Network
    hidden_dim: int = 128  # Good capacity for traffic patterns
    grad_clip_norm: float = 1.0  # Critical for preventing gradient explosion in GNNs

    # Misc
    seed: int = 123
    save_dir: Path = OUTPUTS_DIR


@dataclass
class BaselineConfig:
    """Configuration for fixed-time baseline."""

    episodes: int = 10
    num_intersections: int = 2
    max_steps: int = 300
    switch_period: int = 20
    seed: int = 123
    save_dir: Path = OUTPUTS_DIR
