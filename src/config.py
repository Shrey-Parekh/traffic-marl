"""Centralized configuration for Traffic MARL project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class TrainingConfig:
    """Training hyperparameters for DQN with meta-learning support."""

    # Environment
    num_intersections: int = 2
    max_steps: int = 300
    step_length: float = 2.0
    min_green: int = 5
    arrival_rate_ns: float = 0.3
    arrival_rate_ew: float = 0.3
    depart_capacity: int = 2
    neighbor_obs: bool = False

    # Training
    episodes: int = 50
    learning_rate: float = 0.0005  # Reduced from 0.001 for stability
    batch_size: int = 32  # Reduced from 64 for more stable gradients
    gamma: float = 0.95  # Reduced from 0.99 for faster convergence
    replay_capacity: int = 20000
    min_buffer_size: int = 2000  # Increased from 1000 for more stable sampling

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1  # Increased from 0.05 for more exploration
    epsilon_decay_steps: int = 8000  # Increased from 5000 for slower decay

    # Meta-Learning
    use_meta_learning: bool = False
    meta_epsilon_min: float = 0.05
    meta_epsilon_max: float = 0.3
    meta_lr_scale_min: float = 0.5
    meta_lr_scale_max: float = 1.5
    meta_controller_lr: float = 0.001
    meta_update_frequency: int = 10  # Update meta-controller every N episodes

    # Network
    update_target_steps: int = 500  # Increased from 200 for more stable targets
    hidden_dim: int = 128
    grad_clip_norm: float = 1.0  # Reduced from 5.0 for tighter gradient control

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
