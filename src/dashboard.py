from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Robust import handling for Streamlit execution
try:
    from .env import MiniTrafficEnv, EnvConfig  # type: ignore
    from .config import (  # type: ignore
        OUTPUTS_DIR,
        LIVE_METRICS_JSON,
        METRICS_JSON,
        METRICS_CSV,
        SUMMARY_TXT,
        FINAL_REPORT_JSON,
        BASELINE_METRICS_JSON,
    )
except ImportError:
    _HERE = Path(__file__).parent
    _ROOT = _HERE.parent
    for p in {str(_HERE), str(_ROOT)}:
        if p not in sys.path:
            sys.path.append(p)
    try:
        from env import MiniTrafficEnv, EnvConfig  # type: ignore
        from config import (  # type: ignore
            OUTPUTS_DIR,
            LIVE_METRICS_JSON,
            METRICS_JSON,
            METRICS_CSV,
            SUMMARY_TXT,
            FINAL_REPORT_JSON,
            BASELINE_METRICS_JSON,
        )
    except ImportError as _e:
        raise ImportError(
            "Failed to import modules. Please run 'streamlit run src/dashboard.py' from the project root."
        ) from _e

# For backwards compatibility, keep string paths for Streamlit
LIVE_PATH = str(LIVE_METRICS_JSON)
METRICS_PATH = str(METRICS_JSON)
CSV_PATH = str(METRICS_CSV)
SUMMARY_PATH = str(SUMMARY_TXT)
FINAL_PATH = str(FINAL_REPORT_JSON)
BASELINE_PATH = str(BASELINE_METRICS_JSON)


def load_json(path: str | Path) -> Any:
    path_obj = Path(path) if isinstance(path, str) else path
    if not path_obj.exists():
        return None
    try:
        with open(path_obj, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except (IOError, json.JSONDecodeError):
        return None


def safe_get_data(df, column, default_list):
    """Safely get data from DataFrame, handling both Series and list returns."""
    if column in df.columns:
        return df[column].tolist()
    else:
        return default_list


def get_model_description(model_type: str) -> str:
    """Get human-readable description of model architecture."""
    descriptions = {
        "DQN": "Deep Q-Network - Standard value-based RL",
        "GNN-DQN": "Graph Neural Network DQN - Spatial coordination",
        "PPO-GNN": "Proximal Policy Optimization with GNN - Policy gradient",
        "GAT-DQN": "Graph Attention Network DQN - Attention-based coordination",
        "GNN-A2C": "Actor-Critic with GNN - Policy and value learning",
        "Multi-Model Comparison": "Compare all models - Comprehensive benchmarking",
        "Baseline": "Fixed-time controller - Simple rule-based switching"
    }
    return descriptions.get(model_type, "Unknown model type")


def load_comparison_results(path: str | Path) -> Any:
    """Load comparison results with error handling."""
    return load_json(path)


st.set_page_config(
    page_title="Traffic MARL Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .improvement-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .degradation-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .indicator-green {
        color: #27AE60;
        font-weight: bold;
        font-size: 1.2em;
    }
    .indicator-yellow {
        color: #F39C12;
        font-weight: bold;
        font-size: 1.2em;
    }
    .indicator-red {
        color: #E74C3C;
        font-weight: bold;
        font-size: 1.2em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚦 Traffic MARL Dashboard</h1>
    <p style="margin:0; font-size:1.1em;">Multi-Agent Reinforcement Learning for Traffic Light Control</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Display settings
refresh_enabled = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", min_value=3, max_value=30, value=5)
chart_style = st.sidebar.selectbox("Chart Style", ["Plotly (Interactive)", "Matplotlib"], index=0)

st.sidebar.markdown("---")

# Simulation Parameters
st.sidebar.subheader("📊 Simulation Parameters")
with st.sidebar.form("simulation_form", clear_on_submit=False):
    st.markdown("### Environment Settings")
    with st.expander("📖 What do these mean?", expanded=False):
        st.markdown("""
        **Number of Intersections (N)**: How many traffic intersections are in your network. More intersections = more complex problem.
        
        **Steps per Episode**: How many simulation steps each training episode runs. Each step = 2 seconds of simulation time. More steps = longer episodes but more data.
        
        **Random Seed**: A number that controls randomness. Same seed = same traffic patterns (for reproducible experiments).
        """)
    N_input = st.number_input("Number of Intersections (N)", min_value=1, max_value=20, value=6, 
                             help="Number of traffic intersections in the network")
    max_steps_input = st.number_input("Steps per Episode", min_value=100, max_value=1000, value=300, step=50,
                                     help="Number of simulation steps per episode")
    seed_input = st.number_input("Random Seed", min_value=0, max_value=10_000, value=42,
                                help="Seed for reproducibility")
    
    st.markdown("### Training Settings")
    with st.expander("📖 Training Parameters Explained", expanded=False):
        st.markdown("""
        **Training Episodes**: Total number of complete simulation runs. Each episode is a full simulation from start to finish. More episodes = more learning but takes longer.
        
        **Learning Rate**: How fast the neural network learns. Higher = learns faster but may be unstable. Lower = more stable but slower. Typical range: 0.0001 to 0.01.
        
        **Batch Size**: How many past experiences the AI uses at once to update its knowledge. Larger = smoother updates but slower. Smaller = faster but noisier.
        
        **Discount Factor (γ)**: How much the AI values future rewards vs immediate rewards. 0.99 = very long-term thinking. 0.8 = more short-term focus.
        """)
    episodes_input = st.number_input("Training Episodes", min_value=1, max_value=200, value=50,
                                    help="Number of episodes to train")
    lr_input = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
                              help="Learning rate for the neural network")
    batch_size_input = st.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16,
                                      help="Batch size for training")
    gamma_input = st.number_input("Discount Factor (γ)", min_value=0.8, max_value=0.999, value=0.99, step=0.01, format="%.2f",
                                 help="Discount factor for future rewards")
    
    st.markdown("### Baseline Settings")
    with st.expander("📖 Baseline Explanation", expanded=False):
        st.markdown("""
        **Baseline Switch Period**: For the fixed-time baseline controller, this is how many steps the traffic lights wait before automatically switching. Lower = switches more often. Higher = switches less often. This is a simple rule-based controller used for comparison.
        """)
    baseline_switch_period = st.number_input("Baseline Switch Period", min_value=5, max_value=50, value=20, step=5,
                                            help="Steps between light switches in fixed-time baseline")
    
    model_type = st.radio(
        "Model Architecture", 
        ["DQN", "GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C", "Multi-Model Comparison"], 
        index=0, 
        help="Choose the RL architecture or run all models for comparison"
    )
    
    # Model-specific parameters (only show if not in comparison mode)
    if model_type != "Multi-Model Comparison":
        if model_type in ["PPO-GNN"]:
            st.markdown("#### PPO Parameters")
            ppo_col1, ppo_col2 = st.columns(2)
            with ppo_col1:
                ppo_epochs = st.number_input("PPO Epochs", min_value=1, max_value=10, value=4, help="Number of optimization epochs per episode")
                ppo_clip_ratio = st.number_input("PPO Clip Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Clipping parameter for PPO")
            with ppo_col2:
                ppo_value_coef = st.number_input("Value Coefficient", min_value=0.1, max_value=1.0, value=0.5, step=0.1, help="Value loss coefficient")
                ppo_entropy_coef = st.number_input("Entropy Coefficient", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="Entropy bonus coefficient")
        else:
            # Default values for PPO parameters
            ppo_epochs = 4
            ppo_clip_ratio = 0.2
            ppo_value_coef = 0.5
            ppo_entropy_coef = 0.01
        
        if model_type in ["GNN-A2C"]:
            st.markdown("#### A2C Parameters")
            a2c_col1, a2c_col2 = st.columns(2)
            with a2c_col1:
                a2c_value_coef = st.number_input("Value Coefficient", min_value=0.1, max_value=1.0, value=0.5, step=0.1, help="Value loss coefficient")
            with a2c_col2:
                a2c_entropy_coef = st.number_input("Entropy Coefficient", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="Entropy bonus coefficient")
        else:
            # Default values for A2C parameters
            a2c_value_coef = 0.5
            a2c_entropy_coef = 0.01
        
        if model_type in ["GAT-DQN"]:
            st.markdown("#### Graph Attention Parameters")
            gat_col1, gat_col2 = st.columns(2)
            with gat_col1:
                gat_n_heads = st.number_input("Attention Heads", min_value=1, max_value=8, value=4, help="Number of attention heads")
            with gat_col2:
                gat_dropout = st.number_input("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, help="Dropout rate for attention layers")
        else:
            # Default values for GAT parameters
            gat_n_heads = 4
            gat_dropout = 0.1
    else:
        # Multi-model comparison mode: use default values for all model-specific parameters
        st.info("🔄 **Multi-Model Comparison Mode**: All models will use the same global parameters above. Model-specific parameters are standardized for fair comparison.")
        ppo_epochs = 4
        ppo_clip_ratio = 0.2
        ppo_value_coef = 0.5
        ppo_entropy_coef = 0.01
        a2c_value_coef = 0.5
        a2c_entropy_coef = 0.01
        gat_n_heads = 4
        gat_dropout = 0.1
    
    # Enhanced Meta-learning settings
    st.markdown("### Enhanced Meta-Learning Settings")
    with st.expander("📖 Enhanced Meta-Learning Explained", expanded=False):
        st.markdown("""
        **Enhanced Meta-Learning** enables the AI to adapt its exploration and learning behavior automatically based on performance trends and traffic context.
        
        **New Features:**
        - **Explicit Training**: Meta-controller learns to predict performance improvements
        - **Trend Analysis**: Considers reward and queue trends from recent episodes
        - **Context Awareness**: Responds to traffic patterns and congestion levels
        - **Performance Feedback**: Adapts based on actual performance improvements
        
        **Benefits:**
        - **Smarter Exploration**: Higher exploration when performance is poor, lower when doing well
        - **Context Adaptation**: Different strategies for different traffic conditions
        - **Learning Rate Adaptation**: Faster learning when struggling, careful learning when succeeding
        - **Automatic Tuning**: Reduces need for manual hyperparameter adjustment
        
        **When to Use:**
        - For more sophisticated and adaptive learning behavior
        - When traffic patterns vary significantly over time
        - For automatic hyperparameter optimization
        - When you want the AI to adapt its learning strategy
        
        **Parameters:**
        - **Update Frequency**: How often the meta-controller learns (lower = more frequent updates)
        - **Epsilon Range**: Min/max bounds for adaptive exploration
        - **LR Scale Range**: Min/max bounds for learning rate adjustment
        """)
    
    use_meta_learning = st.checkbox("Enable Enhanced Meta-Learning", value=False, 
                                   help="Enable adaptive exploration and context-aware learning with explicit meta-controller training")
    
    if use_meta_learning:
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            meta_epsilon_min = st.number_input("Meta Epsilon Min", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                                             help="Minimum exploration rate for meta-controller")
            meta_lr_scale_min = st.number_input("Meta LR Scale Min", min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                                              help="Minimum learning rate scale factor")
            meta_update_frequency = st.number_input("Meta Update Frequency", min_value=1, max_value=20, value=5, step=1,
                                                   help="Update meta-controller every N episodes (lower = more frequent)")
        with meta_col2:
            meta_epsilon_max = st.number_input("Meta Epsilon Max", min_value=0.2, max_value=0.5, value=0.3, step=0.01,
                                             help="Maximum exploration rate for meta-controller")
            meta_lr_scale_max = st.number_input("Meta LR Scale Max", min_value=1.0, max_value=2.0, value=1.5, step=0.1,
                                              help="Maximum learning rate scale factor")
    
    use_advanced = st.checkbox("Show Advanced Options", value=False)
    if use_advanced:
        # DQN-specific advanced options
        if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
            st.markdown("#### DQN Advanced Options")
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                epsilon_start = st.number_input("Epsilon Start", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
                epsilon_end = st.number_input("Epsilon End", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
            with adv_col2:
                epsilon_decay_steps = st.number_input("Epsilon Decay Steps", min_value=100, max_value=20000, value=8000, step=500)
                update_target_steps = st.number_input("Target Update Steps", min_value=50, max_value=1000, value=500, step=50)
        else:
            # Default values for DQN parameters
            epsilon_start = 1.0
            epsilon_end = 0.1
            epsilon_decay_steps = 8000
            update_target_steps = 500
        
        min_buffer_size = st.number_input("Min Buffer Size", min_value=100, max_value=5000, value=2000, step=100)
        neighbor_obs = st.checkbox("Enable Neighbor Observations", value=False)
    else:
        # Default values for advanced options
        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay_steps = 8000
        update_target_steps = 500
        min_buffer_size = 2000
        neighbor_obs = False
    
    # Default values for meta-learning if not enabled
    if not use_meta_learning:
        meta_epsilon_min = 0.05
        meta_epsilon_max = 0.3
        meta_lr_scale_min = 0.5
        meta_lr_scale_max = 1.5
        meta_update_frequency = 5
    
    st.info(f"⏱️ Estimated time: ~{episodes_input * max_steps_input * 2 / 60:.1f} minutes per run")
    
    submitted = st.form_submit_button("🚀 Run Simulation", width='stretch')

# Helper function to calculate improvement percentage
def calc_improvement(ai_val: float, baseline_val: float, higher_is_better: bool = False) -> tuple[float, str]:
    """Calculate improvement percentage and return badge HTML."""
    if baseline_val == 0:
        return 0.0, ""
    if higher_is_better:
        pct = ((ai_val - baseline_val) / baseline_val) * 100
    else:
        pct = ((baseline_val - ai_val) / baseline_val) * 100
    
    if pct > 0:
        badge = f'<span class="improvement-badge">+{pct:.1f}% better</span>'
    elif pct < 0:
        badge = f'<span class="degradation-badge">{pct:.1f}% worse</span>'
    else:
        badge = '<span style="color:#6c757d;">No change</span>'
    return pct, badge


def get_performance_indicator(pct_improvement: float) -> tuple[str, str]:
    """Get color-coded performance indicator based on improvement percentage.
    
    Returns:
        tuple: (emoji_indicator, help_text) where emoji is 🟢, 🟡, or 🔴
    """
    if pct_improvement > 15:
        return "🟢", "Excellent (>15% improvement over baseline)"
    elif pct_improvement >= 5:
        return "🟡", "Good (5-15% improvement over baseline)"
    elif pct_improvement >= 0:
        return "🔴", "Needs improvement (<5% improvement over baseline)"
    else:
        return "🔴", "Worse than baseline (negative improvement)"

# Run simulation
if submitted:
    max_steps = int(max_steps_input)
    episodes = int(episodes_input)
    N_val = int(N_input)
    seed_val = int(seed_input)
    
    # Create status container
    status_container = st.container()
    
    with status_container:
        st.info("🔄 Running simulation... This may take several minutes.")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Step 1: Run baseline
    with status_container:
        status_text.text("📊 Running baseline (fixed-time controller)...")
        progress_bar.progress(0.1)
    
    try:
        env_b = MiniTrafficEnv(EnvConfig(
            num_intersections=N_val, 
            max_steps=max_steps, 
            seed=seed_val,
        ))
        obs_b = env_b.reset(seed=seed_val)
        done_b = False
        t_b = 0
        info_b: Dict[str, Any] = {}
        
        while not done_b:
            do_sw = 1 if (t_b % baseline_switch_period == 0 and t_b > 0) else 0
            act_b = {aid: do_sw for aid in obs_b.keys()}
            obs_b, rewards_b, done_b, info_b = env_b.step(act_b)
            t_b += 1
        
        # Save baseline results
        baseline_result = {
            "avg_queue": info_b.get("avg_queue", 0.0),
            "throughput": info_b.get("throughput", 0.0),
            "avg_travel_time": info_b.get("avg_travel_time", 0.0),
            "avg_reward": float(np.mean(list(rewards_b.values()))) if rewards_b else 0.0,
        }
        
        st.session_state["baseline_result"] = baseline_result
        st.session_state["baseline_params"] = {
            "N": N_val,
            "steps": max_steps,
            "seed": seed_val,
            "switch_period": baseline_switch_period,
        }
        st.session_state["current_baseline_period"] = baseline_switch_period
        
    except (ValueError, TypeError, RuntimeError) as e:
        st.error(f"❌ Baseline simulation failed: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Step 2: Run training
    with status_container:
        status_text.text("🤖 Training AI controller...")
        progress_bar.progress(0.3)
    
    # Get the project root directory - use OUTPUTS_DIR which is already correctly configured
    # OUTPUTS_DIR is defined as PROJECT_ROOT / "outputs", so we can get project root from it
    project_root = OUTPUTS_DIR.parent
    project_root_str = str(project_root.resolve())
    
    # Construct train script path
    train_script_path = project_root / "src" / "train.py"
    train_script_path = train_script_path.resolve()
    
    # Verify the file exists
    if not train_script_path.exists():
        st.error(f"❌ Training script not found at: {train_script_path}")
        st.error(f"Project root: {project_root_str}")
        st.error("Please ensure you're running from the project root directory.")
        st.stop()
    
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    
    # Determine which script to run
    if model_type == "Multi-Model Comparison":
        # Run the comparison script
        train_script_path = project_root / "src" / "train_comparison.py"
        train_script_path = train_script_path.resolve()
        
        # Verify the file exists
        if not train_script_path.exists():
            st.error(f"❌ Comparison training script not found at: {train_script_path}")
            st.error(f"Project root: {project_root_str}")
            st.error("Please ensure you're running from the project root directory.")
            st.stop()
        
        cmd_parts = [
            python_executable, str(train_script_path),
            "--episodes", str(episodes),
            "--N", str(N_val),
            "--max_steps", str(max_steps),
            "--seed", str(seed_val),
            "--lr", str(lr_input),
            "--batch_size", str(batch_size_input),
            "--gamma", str(gamma_input),
        ]
        
        # Add meta-learning parameters if enabled
        if use_meta_learning:
            cmd_parts.append("--use_meta_learning")
            cmd_parts.extend(["--meta_epsilon_min", str(meta_epsilon_min)])
            cmd_parts.extend(["--meta_epsilon_max", str(meta_epsilon_max)])
            cmd_parts.extend(["--meta_lr_scale_min", str(meta_lr_scale_min)])
            cmd_parts.extend(["--meta_lr_scale_max", str(meta_lr_scale_max)])
            cmd_parts.extend(["--meta_update_frequency", str(meta_update_frequency)])
        
        if use_advanced:
            cmd_parts.extend(["--epsilon_start", str(epsilon_start)])
            cmd_parts.extend(["--epsilon_end", str(epsilon_end)])
            cmd_parts.extend(["--epsilon_decay_steps", str(epsilon_decay_steps)])
            cmd_parts.extend(["--update_target_steps", str(update_target_steps)])
            cmd_parts.extend(["--min_buffer_size", str(min_buffer_size)])
    else:
        # Run the single model training script
        train_script_path = project_root / "src" / "train.py"
        train_script_path = train_script_path.resolve()
        
        # Verify the file exists
        if not train_script_path.exists():
            st.error(f"❌ Training script not found at: {train_script_path}")
            st.error(f"Project root: {project_root_str}")
            st.error("Please ensure you're running from the project root directory.")
            st.stop()
        
        cmd_parts = [
            python_executable, str(train_script_path),
            "--episodes", str(episodes),
            "--N", str(N_val),
            "--max_steps", str(max_steps),
            "--seed", str(seed_val),
            "--lr", str(lr_input),
            "--batch_size", str(batch_size_input),
            "--gamma", str(gamma_input),
            "--model_type", model_type,
        ]
        
        # Add model-specific parameters
        if model_type == "PPO-GNN":
            cmd_parts.extend(["--ppo_epochs", str(ppo_epochs)])
            cmd_parts.extend(["--ppo_clip_ratio", str(ppo_clip_ratio)])
            cmd_parts.extend(["--ppo_value_coef", str(ppo_value_coef)])
            cmd_parts.extend(["--ppo_entropy_coef", str(ppo_entropy_coef)])
        elif model_type == "GNN-A2C":
            cmd_parts.extend(["--a2c_value_coef", str(a2c_value_coef)])
            cmd_parts.extend(["--a2c_entropy_coef", str(a2c_entropy_coef)])
        elif model_type == "GAT-DQN":
            cmd_parts.extend(["--gat_n_heads", str(gat_n_heads)])
            cmd_parts.extend(["--gat_dropout", str(gat_dropout)])
        
        if use_meta_learning:
            cmd_parts.append("--use_meta_learning")
            cmd_parts.extend(["--meta_epsilon_min", str(meta_epsilon_min)])
            cmd_parts.extend(["--meta_epsilon_max", str(meta_epsilon_max)])
            cmd_parts.extend(["--meta_lr_scale_min", str(meta_lr_scale_min)])
            cmd_parts.extend(["--meta_lr_scale_max", str(meta_lr_scale_max)])
            cmd_parts.extend(["--meta_update_frequency", str(meta_update_frequency)])
        
        if use_advanced:
            if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
                cmd_parts.extend(["--epsilon_start", str(epsilon_start)])
                cmd_parts.extend(["--epsilon_end", str(epsilon_end)])
                cmd_parts.extend(["--epsilon_decay_steps", str(epsilon_decay_steps)])
                cmd_parts.extend(["--update_target_steps", str(update_target_steps)])
            cmd_parts.extend(["--min_buffer_size", str(min_buffer_size)])
            if neighbor_obs:
                cmd_parts.append("--neighbor_obs")
    
    import subprocess  # noqa: S404
    try:
        # Capture stderr to a file so we can see errors, but redirect stdout to avoid blocking
        error_log_path = OUTPUTS_DIR / "training_error.log"
        
        # Set up environment with PYTHONPATH to ensure src module can be found
        env = os.environ.copy()
        # Add project root to PYTHONPATH if not already there
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}"
        else:
            env["PYTHONPATH"] = project_root_str
        
        with open(error_log_path, "w", encoding="utf-8") as error_file:
            # On Windows, use list format instead of shell=True for better reliability
            process = subprocess.Popen(
                cmd_parts, 
                stdout=subprocess.DEVNULL, 
                stderr=error_file, 
                text=True, 
                cwd=project_root_str,
                env=env,
                bufsize=0  # Unbuffered
            )  # noqa: S602
        
        start_time = time.time()
        timeout = 3600
        last_progress = 0.3
        
        # Wait for process to complete, reading output periodically to avoid blocking
        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                st.error(f"⏱️ Training timed out after {timeout} seconds")
                break
            time.sleep(0.5)  # Check more frequently
            # Estimate progress
            estimated_progress = min(0.3 + (elapsed / (episodes * max_steps * 0.01)) * 0.65, 0.95)
            if estimated_progress > last_progress:
                progress_bar.progress(estimated_progress)
                status_text.text(f"🤖 Training... Episode ~{int((estimated_progress - 0.3) / 0.65 * episodes)}/{episodes}")
                last_progress = estimated_progress
        
        # Wait for process to fully terminate
        return_code = process.wait()
        
        if return_code != 0:
            # Read error log to show user what went wrong
            error_log_path = OUTPUTS_DIR / "training_error.log"
            error_message = "Unknown error occurred."
            if error_log_path.exists():
                try:
                    with open(error_log_path, "r", encoding="utf-8") as f:
                        error_lines = f.readlines()
                        if error_lines:
                            # Show last 20 lines of error
                            error_message = "".join(error_lines[-20:])
                except (IOError, OSError):
                    error_message = "Could not read error log."
            
            st.error(f"❌ Training failed with return code {return_code}")
            with st.expander("🔍 View Error Details", expanded=True):
                st.code(error_message, language="text")
            st.stop()
        else:
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed! Loading results...")
            
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        st.error(f"❌ Error running training: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Step 3: Load results
    time.sleep(1)
    
    if model_type == "Multi-Model Comparison":
        # Load comparison results
        comparison_results = load_json(OUTPUTS_DIR / "comparison_results.json")
        
        if comparison_results:
            st.session_state["comparison_results"] = comparison_results
            st.session_state["simulation_complete"] = True
            st.session_state["comparison_mode"] = True
            st.session_state["simulation_params"] = {
                "N": N_val,
                "episodes": episodes,
                "max_steps": max_steps,
                "seed": seed_val,
                "lr": lr_input,
                "batch_size": batch_size_input,
                "gamma": gamma_input,
                "model_type": model_type,
            }
            st.success("✅ Multi-Model Comparison completed successfully!")
            st.rerun()
        else:
            st.error("❌ Comparison completed but results not found. Check the console output.")
    else:
        # Load single model results
        metrics = load_json(METRICS_PATH)
        live = load_json(LIVE_PATH)
        final_report = load_json(FINAL_PATH)
        
        if live:
            st.session_state["latest_metrics"] = metrics
            st.session_state["latest_live"] = live
            st.session_state["latest_final_report"] = final_report
            st.session_state["simulation_complete"] = True
            st.session_state["comparison_mode"] = False
            st.session_state["simulation_params"] = {
                "N": N_val,
                "episodes": episodes,
                "max_steps": max_steps,
                "seed": seed_val,
                "lr": lr_input,
                "batch_size": batch_size_input,
                "gamma": gamma_input,
                "model_type": model_type,
            }
            st.session_state["current_baseline_period"] = baseline_switch_period
            st.success("✅ Simulation completed successfully!")
            st.rerun()
        else:
            st.error("❌ Training completed but results not found. Check the console output.")

# Load data
comparison_results = st.session_state.get("comparison_results", None)
comparison_mode = st.session_state.get("comparison_mode", False)
metrics = st.session_state.get("latest_metrics", load_json(METRICS_PATH))
live = st.session_state.get("latest_live", load_json(LIVE_PATH))
final_report = st.session_state.get("latest_final_report", load_json(FINAL_PATH))
baseline_result = st.session_state.get("baseline_result", None)
baseline_params = st.session_state.get("baseline_params", {})
simulation_params = st.session_state.get("simulation_params", {})

# Homepage: Parameters and Results Explanation
st.markdown("---")
st.header("📋 Current Configuration & Results")

# Show parameters section
param_col1, param_col2 = st.columns(2)

with param_col1:
    st.subheader("⚙️ Simulation Parameters")
    # Show parameters from last run if available, otherwise show current sidebar values
    if simulation_params:
        # Show parameters from last run
        st.markdown(f"""
        **Network Setup:**
        - **Intersections**: {simulation_params.get('N', 'N/A')} traffic lights
        - **Steps per Episode**: {simulation_params.get('max_steps', 'N/A')} steps (each step = 2 seconds)
        - **Random Seed**: {simulation_params.get('seed', 'N/A')} (for reproducible results)
        
        **Model Architecture:**
        - **Type**: {simulation_params.get('model_type', 'N/A')} ({get_model_description(simulation_params.get('model_type', 'DQN'))})
        
        **Training Configuration:**
        - **Episodes**: {simulation_params.get('episodes', 'N/A')} complete training runs
        - **Learning Rate**: {simulation_params.get('lr', 'N/A')} (how fast AI learns)
        - **Batch Size**: {simulation_params.get('batch_size', 'N/A')} (experiences per update)
        - **Discount Factor**: {simulation_params.get('gamma', 'N/A')} (future reward importance)
        """)
    else:
        # Show default values or last used values
        st.markdown("""
        **Network Setup:**
        - **Intersections**: Configure in sidebar (default: 6)
        - **Steps per Episode**: Configure in sidebar (default: 300)
        - **Random Seed**: Configure in sidebar (default: 42)
        
        **Model Architecture:**
        - **Type**: Configure in sidebar (default: DQN)
        
        **Training Configuration:**
        - **Episodes**: Configure in sidebar (default: 50)
        - **Learning Rate**: Configure in sidebar (default: 0.001)
        - **Batch Size**: Configure in sidebar (default: 64)
        - **Discount Factor**: Configure in sidebar (default: 0.99)
        """)
        st.caption("💡 Adjust parameters in the sidebar and click 'Run Simulation' to start training. Your settings will appear here after running.")
    
    with st.expander("📖 What Do These Parameters Mean?", expanded=False):
        st.markdown("""
        **Network Parameters:**
        - **Intersections (N)**: Number of traffic lights in your simulated city. More intersections = more complex coordination needed.
        - **Steps per Episode**: How long each training run lasts. More steps = longer episodes but more learning data.
        - **Random Seed**: Controls randomness. Same seed = same traffic patterns (useful for fair comparisons).
        
        **Training Parameters:**
        - **Episodes**: How many times the AI will practice. More episodes = better learning but takes longer.
        - **Learning Rate**: How quickly the AI adjusts its strategy. Too high = unstable, too low = slow learning.
        - **Batch Size**: How many past experiences the AI reviews at once. Larger = smoother learning.
        - **Discount Factor (γ)**: How much the AI cares about future rewards vs immediate ones. 0.99 = very forward-thinking.
        """)

with param_col2:
    st.subheader("📊 Baseline Settings")
    if baseline_params:
        st.markdown(f"""
        **Baseline Controller:**
        - **Switch Period**: {baseline_params.get('switch_period', 'N/A')} steps between light changes
        - **Network Size**: {baseline_params.get('N', 'N/A')} intersections
        - **Seed**: {baseline_params.get('seed', 'N/A')}
        """)
    else:
        baseline_period = st.session_state.get("current_baseline_period", 20)
        st.markdown(f"""
        **Baseline Controller:**
        - **Switch Period**: {baseline_period} steps between light changes (configure in sidebar)
        - **Network Size**: Same as simulation (from sidebar)
        - **Seed**: Same as simulation (for fair comparison)
        """)
        st.caption("💡 Baseline runs automatically before AI training for comparison.")
    
    with st.expander("📖 What Is Baseline?", expanded=False):
        st.markdown("""
        **Baseline Controller** is a simple rule-based traffic light system:
        - Switches lights automatically every fixed number of steps
        - No learning or adaptation
        - Used as a comparison point to see if the AI is actually improving
        
        **Why Compare?** If the AI performs worse than this simple baseline, we know something needs adjustment!
        """)

# Results explanation section (only show if results exist)
if live or metrics or baseline_result or comparison_results:
    st.markdown("---")
    
    if comparison_mode and comparison_results:
        st.subheader("📈 Understanding Your Comparison Results")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.markdown("#### 🏆 Multi-Model Comparison")
            models_compared = comparison_results.get("models_compared", [])
            best_model = comparison_results.get("best_model", {})
            
            if best_model:
                st.success(f"**Best Model**: {best_model['name']} with average rank score of {best_model['score']:.2f}")
                
                best_metrics = best_model.get("metrics", {})
                if best_metrics:
                    st.markdown(f"""
                    **Best Model Performance:**
                    - **Queue**: {best_metrics.get('avg_queue', 0):.2f} cars
                    - **Throughput**: {best_metrics.get('throughput', 0):.0f} vehicles
                    - **Travel Time**: {best_metrics.get('avg_travel_time', 0):.2f}s
                    """)
            
            st.markdown(f"""
            **Comparison Details:**
            - **Models Tested**: {len(models_compared)} ({', '.join(models_compared)})
            - **Episodes per Model**: {comparison_results.get('episodes_per_model', 0)}
            - **Meta-Learning**: {'✅ Enabled' if comparison_results.get('meta_learning_enabled') else '❌ Disabled'}
            """)
        
        with results_col2:
            st.markdown("#### 📊 How Rankings Work")
            st.markdown("""
            **Ranking System:**
            - Each model is ranked on 3 key metrics: Queue Length, Throughput, Travel Time
            - **Lower rank = Better performance** (1st place = rank 1)
            - **Average Rank Score** combines all metrics (lower is better)
            
            **Metrics Explained:**
            - **Queue Length**: Average waiting vehicles (lower is better)
            - **Throughput**: Total vehicles served (higher is better)  
            - **Travel Time**: Average journey time (lower is better)
            
            **Best Model Selection:**
            - Model with lowest average rank across all metrics
            - Indicates most consistent performance
            """)
        
        with st.expander("📖 Understanding Multi-Model Results", expanded=False):
            st.markdown("""
            **What This Comparison Shows:**
            
            This comparison runs all 5 RL models plus a baseline controller using identical settings:
            - Same network topology and traffic patterns
            - Same training episodes and hyperparameters  
            - Same random seed for reproducible results
            - Fair comparison without model-specific advantages
            
            **Key Insights:**
            - **Best Overall Model**: Consistently performs well across all metrics
            - **Specialized Models**: Some models may excel at specific metrics
            - **Baseline Comparison**: Shows if AI models actually improve over simple rules
            - **Meta-Learning Impact**: Compare performance with/without adaptive learning
            
            **Use This To:**
            - Choose the best model for your specific traffic scenario
            - Understand trade-offs between different approaches
            - Validate that AI models outperform simple baselines
            - See the impact of meta-learning on different architectures
            """)
    else:
        st.subheader("📈 Understanding Your Results")
    
    results_col1, results_col2 = st.columns(2)
    
    with results_col1:
        st.markdown("#### 🤖 AI Controller Performance")
        if live:
            ai_queue = live.get("avg_queue", 0.0)
            ai_throughput = live.get("throughput", 0.0)
            ai_travel_time = live.get("avg_travel_time", 0.0)
            ai_loss = live.get("loss", 0.0)
            ai_policy_loss = live.get("policy_loss", 0.0)
            ai_value_loss = live.get("value_loss", 0.0)
            ai_epsilon = live.get("epsilon", 0.0)
            model_type_used = live.get("model_type", "Unknown")
            
            # Meta-learning metrics
            meta_epsilon = live.get("meta_epsilon")
            meta_lr_scale = live.get("meta_lr_scale")
            meta_value = live.get("meta_value")
            time_of_day = live.get("time_of_day", 0.0)
            global_congestion = live.get("global_congestion", 0.0)
            
            # Display model type
            st.info(f"**Model Architecture**: {model_type_used} - {get_model_description(model_type_used)}")
            
            # Create columns for metrics display
            ai_cols = st.columns(4)
            
            with ai_cols[0]:
                st.metric("Avg Queue", f"{ai_queue:.2f}", help="Average vehicles waiting at intersections")
            with ai_cols[1]:
                st.metric("Throughput", f"{ai_throughput:.0f}", help="Total vehicles that completed journeys")
            with ai_cols[2]:
                st.metric("Travel Time", f"{ai_travel_time:.2f}s", help="Average journey time")
            with ai_cols[3]:
                if model_type_used in ["DQN", "GNN-DQN", "GAT-DQN"]:
                    st.metric("Training Loss", f"{ai_loss:.4f}", help="Neural network prediction error")
                else:
                    st.metric("Policy Loss", f"{ai_policy_loss:.4f}", help="Policy gradient loss")
            
            # Model-specific metrics row
            if model_type_used in ["PPO-GNN", "GNN-A2C"]:
                st.subheader("📊 Policy-Based Metrics")
                policy_cols = st.columns(4)
                
                with policy_cols[0]:
                    st.metric("Policy Loss", f"{ai_policy_loss:.4f}", help="Actor/policy network loss")
                with policy_cols[1]:
                    st.metric("Value Loss", f"{ai_value_loss:.4f}", help="Critic/value network loss")
                with policy_cols[2]:
                    st.metric("Exploration", "Policy-based", help="Uses stochastic policy for exploration")
                with policy_cols[3]:
                    st.metric("Learning Type", "On-policy", help="Learns from current policy interactions")
            
            # Meta-learning metrics row
            if meta_epsilon is not None or meta_lr_scale is not None:
                st.subheader("🧠 Enhanced Meta-Learning Metrics")
                meta_cols = st.columns(4)
                
                with meta_cols[0]:
                    if meta_epsilon is not None:
                        st.metric("Meta Epsilon", f"{meta_epsilon:.3f}", 
                                help="Adaptive exploration rate determined by meta-controller")
                    else:
                        st.metric("Epsilon", f"{ai_epsilon:.3f}", 
                                help="Fixed exploration rate (traditional scheduling)")
                
                with meta_cols[1]:
                    if meta_lr_scale is not None:
                        st.metric("Meta LR Scale", f"{meta_lr_scale:.3f}", 
                                help="Learning rate adjustment factor from meta-controller")
                    else:
                        st.metric("LR Scale", "1.000", 
                                help="No meta-learning (fixed learning rate)")
                
                with meta_cols[2]:
                    if meta_value is not None:
                        st.metric("Meta Value", f"{meta_value:.3f}", 
                                help="Meta-controller's performance prediction")
                    else:
                        st.metric("Meta Value", "N/A", help="Meta-learning not enabled")
                
                with meta_cols[3]:
                    st.metric("Time Context", f"{time_of_day:.3f}", 
                            help="Normalized time context (0-1, simulating daily traffic patterns)")
                
                # Additional context row
                context_cols = st.columns(2)
                with context_cols[0]:
                    st.metric("Global Congestion", f"{global_congestion:.2f}", 
                            help="Average queue length across all intersections")
                with context_cols[1]:
                    adaptation_status = "🟢 Active" if meta_epsilon is not None else "🔴 Disabled"
                    st.metric("Meta Adaptation", adaptation_status, 
                            help="Whether meta-learning is actively adapting hyperparameters")
                
                # Meta-learning explanation
                with st.expander("📖 Understanding Enhanced Meta-Learning", expanded=False):
                    st.markdown("""
                    **Enhanced Meta-Learning** allows the AI to adapt its learning behavior based on performance and context:
                    
                    **Meta Epsilon**: Instead of fixed exploration decay, the meta-controller adjusts exploration based on:
                    - Recent performance trends (if doing poorly, explore more; if doing well, explore less)
                    - Traffic context (rush hour vs low traffic)
                    - Training progress and performance predictions
                    
                    **Meta LR Scale**: Adjusts learning rate dynamically:
                    - Scale > 1.0: Learn faster (when performance is poor or trends are negative)
                    - Scale < 1.0: Learn more carefully (when performance is good)
                    - Based on explicit training to predict performance improvements
                    
                    **Meta Value**: The meta-controller's prediction of performance improvement:
                    - Positive values: Expects performance to improve
                    - Negative values: Expects performance to decline
                    - Used to train the meta-controller explicitly
                    
                    **Context Features**:
                    - **Time of Day**: Simulates daily traffic patterns (rush hour, off-peak)
                    - **Global Congestion**: System-wide traffic load for adaptive responses
                    
                    **Benefits**: More efficient learning, better adaptation to changing conditions, automatic hyperparameter tuning, explicit performance prediction.
                    """)
            else:
                # Traditional metrics
                ai_cols_2 = st.columns(2)
                with ai_cols_2[0]:
                    if model_type_used in ["DQN", "GNN-DQN", "GAT-DQN"]:
                        st.metric("Epsilon", f"{ai_epsilon:.3f}", help="Exploration rate")
                    else:
                        st.metric("Exploration", "Stochastic Policy", help="Policy-based exploration")
                with ai_cols_2[1]:
                    st.metric("Context Features", f"Time: {time_of_day:.2f}, Congestion: {global_congestion:.2f}", 
                            help="Traffic context information")
        else:
            st.info("Run a simulation to see AI performance results.")
    
    with results_col2:
        st.markdown("#### 📊 Performance Comparison")
        if baseline_result and live:
            queue_improvement, _ = calc_improvement(live.get("avg_queue", 0.0), baseline_result["avg_queue"], False)
            throughput_improvement, _ = calc_improvement(live.get("throughput", 0.0), baseline_result["throughput"], True)
            travel_improvement, _ = calc_improvement(live.get("avg_travel_time", 0.0), baseline_result["avg_travel_time"], False)
            
            st.markdown(f"""
            **AI vs Baseline Comparison:**
            
            **Queue Improvement: {queue_improvement:+.1f}%**
            - {f"AI reduced waiting by {abs(queue_improvement):.1f}% compared to baseline" if queue_improvement > 0 else f"AI has {abs(queue_improvement):.1f}% more waiting than baseline"}
            - {get_performance_indicator(queue_improvement)[1]}
            
            **Throughput Improvement: {throughput_improvement:+.1f}%**
            - {f"AI moved {abs(throughput_improvement):.1f}% more vehicles through" if throughput_improvement > 0 else f"AI moved {abs(throughput_improvement):.1f}% fewer vehicles"}
            - {get_performance_indicator(throughput_improvement)[1]}
            
            **Travel Time Improvement: {travel_improvement:+.1f}%**
            - {f"AI reduced travel time by {abs(travel_improvement):.1f}%" if travel_improvement > 0 else f"AI increased travel time by {abs(travel_improvement):.1f}%"}
            - {get_performance_indicator(travel_improvement)[1]}
            
            **What This Means:**
            - Positive percentages = AI is better than baseline ✅
            - Negative percentages = AI needs more training ⚠️
            - Color indicators show performance level at a glance
            """)
        elif baseline_result:
            st.info("AI training results will appear here after simulation completes.")
        else:
            st.info("Run a simulation with baseline comparison to see performance metrics.")
    
    # Overall interpretation
    if live and baseline_result:
        st.markdown("---")
        st.markdown("#### 💡 What Do These Numbers Mean?")
        
        overall_improvement = (queue_improvement + throughput_improvement + travel_improvement) / 3
        
        if overall_improvement > 10:
            st.success("""
            **🎉 Excellent Results!** 
            
            Your AI controller is performing significantly better than the baseline. The AI has learned to:
            - Reduce traffic congestion (fewer waiting vehicles)
            - Improve traffic flow (more vehicles getting through)
            - Decrease travel times (faster journeys)
            
            The model is working well! Consider training for more episodes to see if it can improve even further.
            """)
        elif overall_improvement > 0:
            st.info("""
            **👍 Good Progress!**
            
            Your AI controller is performing better than the baseline, but there's room for improvement. The AI is learning, but may need:
            - More training episodes
            - Different learning parameters
            - More exploration time
            
            Keep training to see better results!
            """)
        else:
            st.warning("""
            **⚠️ Needs Improvement**
            
            The AI controller is not yet outperforming the baseline. This could mean:
            - Not enough training episodes yet
            - Learning rate too high or too low
            - Need more exploration before exploiting learned knowledge
            
            Try adjusting parameters or training for more episodes.
            """)

st.markdown("---")

# Main content tabs
if comparison_mode and comparison_results:
    # Multi-model comparison mode
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🏆 Agent Comparison", "📋 Detailed Results"])
    
    # Tab 1: Overview for comparison mode
    with tab1:
        st.header("Multi-Model Comparison Overview")
        
        if comparison_results.get("best_model"):
            best_model = comparison_results["best_model"]
            st.success(f"🏆 **Best Performing Model**: {best_model['name']} (Average Rank Score: {best_model['score']:.2f})")
            
            # Show best model metrics
            best_metrics = best_model.get("metrics", {})
            if best_metrics:
                best_cols = st.columns(4)
                best_cols[0].metric("Best Avg Queue", f"{best_metrics.get('avg_queue', 0):.2f}")
                best_cols[1].metric("Best Throughput", f"{best_metrics.get('throughput', 0):.0f}")
                best_cols[2].metric("Best Travel Time", f"{best_metrics.get('avg_travel_time', 0):.2f}s")
                best_cols[3].metric("Models Compared", len(comparison_results.get("models_compared", [])))
        
        # Comparison summary
        st.subheader("📈 Comparison Summary")
        models_compared = comparison_results.get("models_compared", [])
        episodes_per_model = comparison_results.get("episodes_per_model", 0)
        meta_learning_enabled = comparison_results.get("meta_learning_enabled", False)
        
        summary_cols = st.columns(3)
        summary_cols[0].metric("Models Compared", len(models_compared))
        summary_cols[1].metric("Episodes per Model", episodes_per_model)
        summary_cols[2].metric("Meta-Learning", "✅ Enabled" if meta_learning_enabled else "❌ Disabled")
        
        st.markdown(f"**Models Tested**: {', '.join(models_compared)}")
        
        # Quick performance overview
        st.subheader("🎯 Performance Overview")
        results = comparison_results.get("results", {})
        
        if results:
            overview_data = []
            for model_name, model_results in results.items():
                if "error" not in model_results and "average_metrics" in model_results:
                    avg_metrics = model_results["average_metrics"]
                    overview_data.append({
                        "Model": model_name,
                        "Avg Queue": f"{avg_metrics.get('avg_queue', 0):.2f}",
                        "Throughput": f"{avg_metrics.get('throughput', 0):.0f}",
                        "Travel Time": f"{avg_metrics.get('avg_travel_time', 0):.2f}s",
                        "Status": "✅ Success"
                    })
                else:
                    overview_data.append({
                        "Model": model_name,
                        "Avg Queue": "N/A",
                        "Throughput": "N/A", 
                        "Travel Time": "N/A",
                        "Status": "❌ Failed"
                    })
            
            if overview_data:
                overview_df = pd.DataFrame(overview_data)
                st.dataframe(overview_df, width='stretch', hide_index=True)
    
    # Tab 2: Agent Comparison
    with tab2:
        st.header("🏆 Agent Performance Comparison")
        
        results = comparison_results.get("results", {})
        rankings = comparison_results.get("ranking", {})
        
        if results and rankings:
            # Performance comparison table
            st.subheader("📊 Performance Metrics Comparison")
            
            comparison_data = []
            for model_name, model_results in results.items():
                if "error" not in model_results and "average_metrics" in model_results:
                    avg_metrics = model_results["average_metrics"]
                    comparison_data.append({
                        "Model": model_name,
                        "Avg Queue": avg_metrics.get("avg_queue", 0),
                        "Throughput": avg_metrics.get("throughput", 0),
                        "Travel Time": avg_metrics.get("avg_travel_time", 0),
                        "Avg Reward": avg_metrics.get("avg_reward", 0),
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create comparison charts
                if chart_style == "Plotly (Interactive)":
                    # Bar charts for each metric
                    fig_comparison = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Average Queue (Lower is Better)", "Throughput (Higher is Better)", 
                                      "Travel Time (Lower is Better)", "Average Reward (Higher is Better)"),
                        vertical_spacing=0.15,
                    )
                    
                    models = comparison_df["Model"].tolist()
                    
                    # Queue comparison
                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Avg Queue"], name="Avg Queue", 
                              marker_color='#ff6b6b', text=comparison_df["Avg Queue"].round(2), textposition='auto'),
                        row=1, col=1
                    )
                    
                    # Throughput comparison
                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Throughput"], name="Throughput", 
                              marker_color='#4ecdc4', text=comparison_df["Throughput"].round(0), textposition='auto'),
                        row=1, col=2
                    )
                    
                    # Travel time comparison
                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Travel Time"], name="Travel Time", 
                              marker_color='#45b7d1', text=comparison_df["Travel Time"].round(2), textposition='auto'),
                        row=2, col=1
                    )
                    
                    # Reward comparison
                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Avg Reward"], name="Avg Reward", 
                              marker_color='#f39c12', text=comparison_df["Avg Reward"].round(1), textposition='auto'),
                        row=2, col=2
                    )
                    
                    fig_comparison.update_layout(height=700, showlegend=False, template="plotly_white")
                    fig_comparison.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_comparison, width='stretch')
                else:
                    # Matplotlib version
                    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                    
                    models = comparison_df["Model"].tolist()
                    
                    axes[0, 0].bar(models, comparison_df["Avg Queue"], color='#ff6b6b')
                    axes[0, 0].set_title("Average Queue (Lower is Better)", fontweight='bold')
                    axes[0, 0].set_ylabel("Cars")
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    
                    axes[0, 1].bar(models, comparison_df["Throughput"], color='#4ecdc4')
                    axes[0, 1].set_title("Throughput (Higher is Better)", fontweight='bold')
                    axes[0, 1].set_ylabel("Vehicles")
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    
                    axes[1, 0].bar(models, comparison_df["Travel Time"], color='#45b7d1')
                    axes[1, 0].set_title("Travel Time (Lower is Better)", fontweight='bold')
                    axes[1, 0].set_ylabel("Seconds")
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    axes[1, 1].bar(models, comparison_df["Avg Reward"], color='#f39c12')
                    axes[1, 1].set_title("Average Reward (Higher is Better)", fontweight='bold')
                    axes[1, 1].set_ylabel("Reward")
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Rankings table
                st.subheader("🏅 Performance Rankings")
                
                ranking_data = []
                for metric, ranking in rankings.items():
                    metric_name = metric.replace("_", " ").title()
                    for entry in ranking:
                        ranking_data.append({
                            "Metric": metric_name,
                            "Rank": entry["rank"],
                            "Model": entry["model"],
                            "Value": f"{entry['value']:.2f}",
                        })
                
                if ranking_data:
                    ranking_df = pd.DataFrame(ranking_data)
                    
                    # Create pivot table for better display
                    pivot_df = ranking_df.pivot(index="Model", columns="Metric", values="Rank")
                    pivot_df["Average Rank"] = pivot_df.mean(axis=1).round(2)
                    pivot_df = pivot_df.sort_values("Average Rank")
                    
                    st.dataframe(pivot_df, width='stretch')
                    
                    # Highlight best performers
                    st.markdown("#### 🎖️ Best Performers by Metric")
                    best_performers = {}
                    for metric, ranking in rankings.items():
                        if ranking:
                            best_model = ranking[0]["model"]
                            best_value = ranking[0]["value"]
                            best_performers[metric.replace("_", " ").title()] = f"{best_model} ({best_value:.2f})"
                    
                    perf_cols = st.columns(len(best_performers))
                    for i, (metric, performer) in enumerate(best_performers.items()):
                        perf_cols[i].metric(f"Best {metric}", performer)
        else:
            st.info("No comparison results available.")
    
    # Tab 3: Detailed Results
    with tab3:
        st.header("📋 Detailed Comparison Results")
        
        results = comparison_results.get("results", {})
        
        if results:
            # Model selector for detailed view
            selected_model = st.selectbox("Select Model for Detailed View", list(results.keys()))
            
            if selected_model and selected_model in results:
                model_results = results[selected_model]
                
                if "error" in model_results:
                    st.error(f"❌ {selected_model} failed: {model_results['error']}")
                else:
                    st.subheader(f"📊 {selected_model} Detailed Results")
                    
                    # Show average metrics
                    avg_metrics = model_results.get("average_metrics", {})
                    if avg_metrics:
                        st.markdown("#### Average Performance")
                        avg_cols = st.columns(4)
                        avg_cols[0].metric("Avg Queue", f"{avg_metrics.get('avg_queue', 0):.2f}")
                        avg_cols[1].metric("Throughput", f"{avg_metrics.get('throughput', 0):.0f}")
                        avg_cols[2].metric("Travel Time", f"{avg_metrics.get('avg_travel_time', 0):.2f}s")
                        avg_cols[3].metric("Episodes", model_results.get("episodes", 0))
                    
                    # Show episode-by-episode results if available
                    all_results = model_results.get("all_results", [])
                    if all_results:
                        st.markdown("#### Episode-by-Episode Results")
                        results_df = pd.DataFrame(all_results)
                        
                        # Show key columns
                        display_cols = ["episode", "avg_queue", "throughput", "avg_travel_time", "avg_reward"]
                        available_cols = [col for col in display_cols if col in results_df.columns]
                        
                        if available_cols:
                            st.dataframe(results_df[available_cols], width='stretch', hide_index=True)
                            
                            # Download button for detailed results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {selected_model} Results as CSV",
                                data=csv,
                                file_name=f"{selected_model.lower().replace('-', '_')}_results.csv",
                                mime="text/csv"
                            )
        else:
            st.info("No detailed results available.")

elif metrics or live or baseline_result:
    # Single model mode
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Training Progress", "🔍 Comparison", "🧠 Learning Analysis", "📋 Detailed Metrics"])
    
    # Tab 1: Overview
    with tab1:
        st.header("Performance Overview")
        
        with st.expander("📖 Understanding the Metrics", expanded=False):
            st.markdown("""
            **Average Queue**: Average number of vehicles waiting at intersections. Lower is better - means less traffic congestion.
            
            **Throughput**: Total number of vehicles that completed their journeys during the episode. Higher is better - means more traffic is flowing through the system.
            
            **Avg Travel Time**: Average time (in seconds) vehicles spend in the network from entry to exit. Lower is better - means faster trips.
            
            **Training Loss**: How much error the neural network has in predicting action values. Lower is better - means the AI is learning more accurately.
            
            **Epsilon**: Exploration rate (0-1). High values (near 1.0) mean the agent explores randomly; low values (near 0.05) mean it uses learned knowledge.
            
            **Performance Indicators** (when baseline comparison available):
            - 🟢 Green: Excellent performance (>15% improvement over baseline)
            - 🟡 Yellow: Good performance (5-15% improvement over baseline)
            - 🔴 Red: Needs improvement (<5% improvement or worse than baseline)
            """)
        
        # Comparison if baseline exists
        if baseline_result and live:
            st.subheader("📊 AI vs Baseline Comparison")
            
            # Calculate improvements for comparison section
            queue_improvement, queue_badge = calc_improvement(ai_queue, baseline_result["avg_queue"], False)
            throughput_improvement, throughput_badge = calc_improvement(ai_throughput, baseline_result["throughput"], True)
            travel_improvement, travel_badge = calc_improvement(ai_travel_time, baseline_result["avg_travel_time"], False)
            
            # Comparison metrics
            comp_cols = st.columns(3)
            
            with comp_cols[0]:
                queue_indicator, queue_help = get_performance_indicator(queue_improvement)
                st.metric(
                    "Average Queue",
                    f"{queue_indicator} {ai_queue:.2f}",
                    f"{queue_improvement:+.2f} ({baseline_result['avg_queue']:.2f} baseline)",
                    help=f"Lower is better. {queue_help}"
                )
                st.markdown(queue_badge, unsafe_allow_html=True)
            
            with comp_cols[1]:
                throughput_indicator, throughput_help = get_performance_indicator(throughput_improvement)
                st.metric(
                    "Throughput",
                    f"{throughput_indicator} {ai_throughput:.0f}",
                    f"{throughput_improvement:+.0f} ({baseline_result['throughput']:.0f} baseline)",
                    help=f"Higher is better. {throughput_help}"
                )
                st.markdown(throughput_badge, unsafe_allow_html=True)
            
            with comp_cols[2]:
                travel_indicator, travel_help = get_performance_indicator(travel_improvement)
                st.metric(
                    "Travel Time",
                    f"{travel_indicator} {ai_travel_time:.2f}s",
                    f"{travel_improvement:+.2f}s ({baseline_result['avg_travel_time']:.2f}s baseline)",
                    help=f"Lower is better. {travel_help}"
                )
                st.markdown(travel_badge, unsafe_allow_html=True)
            
            # Comparison chart
            comparison_data = pd.DataFrame({
                "Metric": ["Avg Queue (lower is better)", "Throughput (higher is better)", "Travel Time (lower is better)"],
                "Baseline": [
                    baseline_result["avg_queue"],
                    baseline_result["throughput"],
                    baseline_result["avg_travel_time"],
                ],
                "AI Controller": [
                    ai_queue,
                    ai_throughput,
                    ai_travel_time,
                ],
            })
            
            # Normalize for visualization (invert queue and travel time)
            comparison_data_norm = comparison_data.copy()
            comparison_data_norm.loc[0, "Baseline"] = -comparison_data_norm.loc[0, "Baseline"]
            comparison_data_norm.loc[0, "AI Controller"] = -comparison_data_norm.loc[0, "AI Controller"]
            comparison_data_norm.loc[2, "Baseline"] = -comparison_data_norm.loc[2, "Baseline"]
            comparison_data_norm.loc[2, "AI Controller"] = -comparison_data_norm.loc[2, "AI Controller"]
            
            if chart_style == "Plotly (Interactive)":
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Baseline",
                    x=comparison_data["Metric"],
                    y=[comparison_data["Baseline"][0], comparison_data["Baseline"][1], comparison_data["Baseline"][2]],
                    marker_color='#6c757d',
                    text=[f"{x:.2f}" for x in comparison_data["Baseline"]],
                    textposition='auto',
                ))
                fig_comp.add_trace(go.Bar(
                    name="AI Controller",
                    x=comparison_data["Metric"],
                    y=[comparison_data["AI Controller"][0], comparison_data["AI Controller"][1], comparison_data["AI Controller"][2]],
                    marker_color='#007bff',
                    text=[f"{x:.2f}" for x in comparison_data["AI Controller"]],
                    textposition='auto',
                ))
                fig_comp.update_layout(
                    title="Performance Comparison: Baseline vs AI Controller",
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    barmode='group',
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig_comp, width='stretch')
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(comparison_data))
                width = 0.35
                ax.bar(x - width/2, comparison_data["Baseline"], width, label="Baseline", color='#6c757d')
                ax.bar(x + width/2, comparison_data["AI Controller"], width, label="AI Controller", color='#007bff')
                ax.set_xlabel("Metric")
                ax.set_ylabel("Value")
                ax.set_title("Performance Comparison: Baseline vs AI Controller")
                ax.set_xticks(x)
                ax.set_xticklabels(comparison_data["Metric"], rotation=15, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Final report summary
        if final_report:
            st.subheader("📈 Training Summary")
            avg_metrics = final_report.get("average_metrics", {})
            if avg_metrics:
                summary_cols = st.columns(4)
                summary_cols[0].metric("Avg Queue (all episodes)", f"{avg_metrics.get('avg_queue', 0):.2f}")
                summary_cols[1].metric("Avg Throughput", f"{avg_metrics.get('throughput', 0):.0f}")
                summary_cols[2].metric("Avg Travel Time", f"{avg_metrics.get('avg_travel_time', 0):.2f}s")
                summary_cols[3].metric("Episodes Trained", final_report.get("episodes", 0))
    
    # Tab 2: Training Progress
    with tab2:
        st.header("Training Progress Over Time")
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            episodes_list = safe_get_data(df_metrics, "episode", list(range(len(metrics))))
            
            # Extract metrics
            queue_data = safe_get_data(df_metrics, "avg_queue", [0.0] * len(metrics))
            throughput_data = safe_get_data(df_metrics, "throughput", [0.0] * len(metrics))
            travel_time_data = safe_get_data(df_metrics, "avg_travel_time", [0.0] * len(metrics))
            loss_data = safe_get_data(df_metrics, "loss", [0.0] * len(metrics))
            policy_loss_data = safe_get_data(df_metrics, "policy_loss", [0.0] * len(metrics))
            value_loss_data = safe_get_data(df_metrics, "value_loss", [0.0] * len(metrics))
            epsilon_data = safe_get_data(df_metrics, "epsilon", [0.0] * len(metrics))
            
            # Check model type from data
            model_type_from_data = df_metrics.get("model_type", pd.Series(["DQN"] * len(metrics))).iloc[0] if len(df_metrics) > 0 else "DQN"
            
            if chart_style == "Plotly (Interactive)":
                # Create subplots based on model type
                if model_type_from_data in ["PPO-GNN", "GNN-A2C"]:
                    # Policy-based models: show policy and value losses
                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=("Average Queue Length", "Throughput", "Average Travel Time", 
                                      "Policy Loss", "Value Loss", "Combined Loss"),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.08,
                    )
                    
                    # Queue
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers', 
                                 name='Queue', line=dict(color='#ff6b6b', width=2)),
                        row=1, col=1
                    )
                    
                    # Throughput
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                                 name='Throughput', line=dict(color='#4ecdc4', width=2)),
                        row=1, col=2
                    )
                    
                    # Travel Time
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=travel_time_data, mode='lines+markers',
                                 name='Travel Time', line=dict(color='#45b7d1', width=2)),
                        row=1, col=3
                    )
                    
                    # Policy Loss
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=policy_loss_data, mode='lines+markers',
                                 name='Policy Loss', line=dict(color='#f39c12', width=2)),
                        row=2, col=1
                    )
                    
                    # Value Loss
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=value_loss_data, mode='lines+markers',
                                 name='Value Loss', line=dict(color='#9b59b6', width=2)),
                        row=2, col=2
                    )
                    
                    # Combined Loss (for comparison)
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=loss_data, mode='lines+markers',
                                 name='Combined Loss', line=dict(color='#e74c3c', width=2)),
                        row=2, col=3
                    )
                    
                    fig.update_xaxes(title_text="Episode", row=2, col=1)
                    fig.update_xaxes(title_text="Episode", row=2, col=2)
                    fig.update_xaxes(title_text="Episode", row=2, col=3)
                    fig.update_yaxes(title_text="Cars", row=1, col=1)
                    fig.update_yaxes(title_text="Vehicles", row=1, col=2)
                    fig.update_yaxes(title_text="Seconds", row=1, col=3)
                    fig.update_yaxes(title_text="Policy Loss", row=2, col=1)
                    fig.update_yaxes(title_text="Value Loss", row=2, col=2)
                    fig.update_yaxes(title_text="Combined Loss", row=2, col=3)
                    
                    fig.update_layout(height=700, showlegend=False, template="plotly_white")
                    st.plotly_chart(fig, width='stretch')
                else:
                    # Value-based models: show standard DQN metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Average Queue Length", "Throughput", "Average Travel Time", "Training Loss"),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1,
                    )
                    
                    # Queue
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers', 
                                 name='Queue', line=dict(color='#ff6b6b', width=2)),
                        row=1, col=1
                    )
                    
                    # Throughput
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                                 name='Throughput', line=dict(color='#4ecdc4', width=2)),
                        row=1, col=2
                    )
                    
                    # Travel Time
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=travel_time_data, mode='lines+markers',
                                 name='Travel Time', line=dict(color='#45b7d1', width=2)),
                        row=2, col=1
                    )
                    
                    # Loss
                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=loss_data, mode='lines+markers',
                                 name='Loss', line=dict(color='#f39c12', width=2)),
                        row=2, col=2
                    )
                    
                    fig.update_xaxes(title_text="Episode", row=2, col=1)
                    fig.update_xaxes(title_text="Episode", row=2, col=2)
                    fig.update_yaxes(title_text="Cars", row=1, col=1)
                    fig.update_yaxes(title_text="Vehicles", row=1, col=2)
                    fig.update_yaxes(title_text="Seconds", row=2, col=1)
                    fig.update_yaxes(title_text="Loss", row=2, col=2)
                    
                    fig.update_layout(height=600, showlegend=False, template="plotly_white")
                    st.plotly_chart(fig, width='stretch')
                
                # Epsilon decay (only for value-based models)
                if model_type_from_data in ["DQN", "GNN-DQN", "GAT-DQN"] and epsilon_data and any(epsilon_data):
                    fig_eps = go.Figure()
                    fig_eps.add_trace(go.Scatter(
                        x=episodes_list, y=epsilon_data, mode='lines+markers',
                        name='Epsilon', line=dict(color='#9b59b6', width=2)
                    ))
                    fig_eps.update_layout(
                        title="Exploration Rate (Epsilon) Over Time",
                        xaxis_title="Episode",
                        yaxis_title="Epsilon",
                        height=300,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_eps, width='stretch')
            else:
                # Matplotlib version
                if model_type_from_data in ["PPO-GNN", "GNN-A2C"]:
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    
                    axes[0, 0].plot(episodes_list, queue_data, color='#ff6b6b', linewidth=2, marker='o', markersize=4)
                    axes[0, 0].set_title("Average Queue Length", fontweight='bold')
                    axes[0, 0].set_xlabel("Episode")
                    axes[0, 0].set_ylabel("Cars")
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    axes[0, 1].plot(episodes_list, throughput_data, color='#4ecdc4', linewidth=2, marker='o', markersize=4)
                    axes[0, 1].set_title("Throughput", fontweight='bold')
                    axes[0, 1].set_xlabel("Episode")
                    axes[0, 1].set_ylabel("Vehicles")
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    axes[0, 2].plot(episodes_list, travel_time_data, color='#45b7d1', linewidth=2, marker='o', markersize=4)
                    axes[0, 2].set_title("Average Travel Time", fontweight='bold')
                    axes[0, 2].set_xlabel("Episode")
                    axes[0, 2].set_ylabel("Seconds")
                    axes[0, 2].grid(True, alpha=0.3)
                    
                    axes[1, 0].plot(episodes_list, policy_loss_data, color='#f39c12', linewidth=2, marker='o', markersize=4)
                    axes[1, 0].set_title("Policy Loss", fontweight='bold')
                    axes[1, 0].set_xlabel("Episode")
                    axes[1, 0].set_ylabel("Policy Loss")
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    axes[1, 1].plot(episodes_list, value_loss_data, color='#9b59b6', linewidth=2, marker='o', markersize=4)
                    axes[1, 1].set_title("Value Loss", fontweight='bold')
                    axes[1, 1].set_xlabel("Episode")
                    axes[1, 1].set_ylabel("Value Loss")
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    axes[1, 2].plot(episodes_list, loss_data, color='#e74c3c', linewidth=2, marker='o', markersize=4)
                    axes[1, 2].set_title("Combined Loss", fontweight='bold')
                    axes[1, 2].set_xlabel("Episode")
                    axes[1, 2].set_ylabel("Combined Loss")
                    axes[1, 2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    axes[0, 0].plot(episodes_list, queue_data, color='#ff6b6b', linewidth=2, marker='o', markersize=4)
                    axes[0, 0].set_title("Average Queue Length", fontweight='bold')
                    axes[0, 0].set_xlabel("Episode")
                    axes[0, 0].set_ylabel("Cars")
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    axes[0, 1].plot(episodes_list, throughput_data, color='#4ecdc4', linewidth=2, marker='o', markersize=4)
                    axes[0, 1].set_title("Throughput", fontweight='bold')
                    axes[0, 1].set_xlabel("Episode")
                    axes[0, 1].set_ylabel("Vehicles")
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    axes[1, 0].plot(episodes_list, travel_time_data, color='#45b7d1', linewidth=2, marker='o', markersize=4)
                    axes[1, 0].set_title("Average Travel Time", fontweight='bold')
                    axes[1, 0].set_xlabel("Episode")
                    axes[1, 0].set_ylabel("Seconds")
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    axes[1, 1].plot(episodes_list, loss_data, color='#f39c12', linewidth=2, marker='o', markersize=4)
                    axes[1, 1].set_title("Training Loss", fontweight='bold')
                    axes[1, 1].set_xlabel("Episode")
                    axes[1, 1].set_ylabel("Loss")
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("No training data available. Run a simulation to see training progress.")
    
    # Tab 3: Comparison
    with tab3:
        st.header("Detailed Comparison: AI vs Baseline")
        
        if baseline_result and metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            
            # Calculate baseline average (if multiple baseline episodes exist)
            baseline_metrics = load_json(BASELINE_PATH)
            if baseline_metrics and isinstance(baseline_metrics, list):
                baseline_df = pd.DataFrame(baseline_metrics)
                baseline_avg_queue = baseline_df["avg_queue"].mean()
                baseline_avg_throughput = baseline_df["throughput"].mean()
                baseline_avg_travel = baseline_df["avg_travel_time"].mean()
            else:
                baseline_avg_queue = baseline_result["avg_queue"]
                baseline_avg_throughput = baseline_result["throughput"]
                baseline_avg_travel = baseline_result["avg_travel_time"]
            
            # Get AI metrics - safely access columns
            ai_avg_queue = df_metrics["avg_queue"].mean() if "avg_queue" in df_metrics.columns else 0.0
            ai_avg_throughput = df_metrics["throughput"].mean() if "throughput" in df_metrics.columns else 0.0
            ai_avg_travel = df_metrics["avg_travel_time"].mean() if "avg_travel_time" in df_metrics.columns else 0.0
            ai_final_queue = df_metrics["avg_queue"].iloc[-1] if "avg_queue" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            ai_final_throughput = df_metrics["throughput"].iloc[-1] if "throughput" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            ai_final_travel = df_metrics["avg_travel_time"].iloc[-1] if "avg_travel_time" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            
            # Comparison table
            st.subheader("Average Performance Comparison")
            comparison_table = pd.DataFrame({
                "Metric": ["Average Queue", "Throughput", "Average Travel Time"],
                "Baseline": [baseline_avg_queue, baseline_avg_throughput, baseline_avg_travel],
                "AI (Average)": [ai_avg_queue, ai_avg_throughput, ai_avg_travel],
                "AI (Final)": [ai_final_queue, ai_final_throughput, ai_final_travel],
                "Improvement (Avg)": [
                    f"{((baseline_avg_queue - ai_avg_queue) / baseline_avg_queue * 100):.1f}%",
                    f"{((ai_avg_throughput - baseline_avg_throughput) / baseline_avg_throughput * 100):.1f}%",
                    f"{((baseline_avg_travel - ai_avg_travel) / baseline_avg_travel * 100):.1f}%",
                ],
            })
            st.dataframe(comparison_table, width='stretch', hide_index=True)
            
            # Side-by-side comparison charts
            st.subheader("Learning Curves with Baseline Reference")
            episodes_list = safe_get_data(df_metrics, "episode", list(range(len(metrics))))
            queue_data = safe_get_data(df_metrics, "avg_queue", [0.0] * len(metrics))
            throughput_data = safe_get_data(df_metrics, "throughput", [0.0] * len(metrics))
            travel_time_data = safe_get_data(df_metrics, "avg_travel_time", [0.0] * len(metrics))
            
            if chart_style == "Plotly (Interactive)":
                fig_comp = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("Queue Length", "Throughput", "Travel Time"),
                )
                
                # Queue with baseline line
                fig_comp.add_trace(
                    go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers',
                             name='AI Queue', line=dict(color='#007bff', width=2)),
                    row=1, col=1
                )
                fig_comp.add_hline(y=baseline_avg_queue, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline", row=1, col=1)
                
                # Throughput with baseline line
                fig_comp.add_trace(
                    go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                             name='AI Throughput', line=dict(color='#28a745', width=2)),
                    row=1, col=2
                )
                fig_comp.add_hline(y=baseline_avg_throughput, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline", row=1, col=2)
                
                # Travel time with baseline line
                fig_comp.add_trace(
                    go.Scatter(x=episodes_list, y=travel_time_data, mode='lines+markers',
                             name='AI Travel Time', line=dict(color='#dc3545', width=2)),
                    row=1, col=3
                )
                fig_comp.add_hline(y=baseline_avg_travel, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline", row=1, col=3)
                
                fig_comp.update_xaxes(title_text="Episode", row=1, col=1)
                fig_comp.update_xaxes(title_text="Episode", row=1, col=2)
                fig_comp.update_xaxes(title_text="Episode", row=1, col=3)
                fig_comp.update_yaxes(title_text="Cars", row=1, col=1)
                fig_comp.update_yaxes(title_text="Vehicles", row=1, col=2)
                fig_comp.update_yaxes(title_text="Seconds", row=1, col=3)
                
                fig_comp.update_layout(height=400, showlegend=False, template="plotly_white")
                st.plotly_chart(fig_comp, width='stretch')
            else:
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                
                axes[0].plot(episodes_list, queue_data, color='#007bff', linewidth=2, label='AI')
                axes[0].axhline(y=baseline_avg_queue, color='gray', linestyle='--', label='Baseline')
                axes[0].set_title("Queue Length", fontweight='bold')
                axes[0].set_xlabel("Episode")
                axes[0].set_ylabel("Cars")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(episodes_list, throughput_data, color='#28a745', linewidth=2, label='AI')
                axes[1].axhline(y=baseline_avg_throughput, color='gray', linestyle='--', label='Baseline')
                axes[1].set_title("Throughput", fontweight='bold')
                axes[1].set_xlabel("Episode")
                axes[1].set_ylabel("Vehicles")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(episodes_list, travel_time_data, color='#dc3545', linewidth=2, label='AI')
                axes[2].axhline(y=baseline_avg_travel, color='gray', linestyle='--', label='Baseline')
                axes[2].set_title("Travel Time", fontweight='bold')
                axes[2].set_xlabel("Episode")
                axes[2].set_ylabel("Seconds")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Run a simulation with baseline comparison to see detailed comparisons.")
    
    # Tab 4: Learning Analysis
    with tab4:
        st.header("🧠 AI Learning Analysis")
        st.markdown("### Evidence that the AI is learning from experience")
        
        with st.expander("📖 Understanding Learning Evidence", expanded=False):
            st.markdown("""
            **Learning Evidence Metrics:**
            
            - **Reward Improvement**: Compares average reward between early episodes (first 1/3) and late episodes (last 1/3). Positive values mean the agent is getting better rewards (learning to reduce queues).
            
            - **Loss Reduction %**: Shows how much the training loss decreased. Higher percentage means the neural network is learning better predictions.
            
            - **Queue Reduction %**: Shows improvement in queue management. Higher percentage means the agent learned to reduce traffic congestion.
            
            - **Throughput Gain %**: Shows improvement in moving vehicles through the system. Higher percentage means more efficient traffic flow.
            
            **Early vs Late Comparison**: We compare the first third of episodes (when the agent is still learning) with the last third (when it should be using learned strategies) to see if performance improved.
            
            **Trend Lines**: The smooth bold lines show moving averages, making it easier to see overall trends despite natural variations between episodes.
            """)
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            episodes_list = safe_get_data(df_metrics, "episode", list(range(len(metrics))))
            
            # Extract key learning metrics - handle both pandas Series and list defaults
            reward_data = safe_get_data(df_metrics, "avg_reward", [0.0] * len(metrics))
            loss_data = safe_get_data(df_metrics, "loss", [0.0] * len(metrics))
            queue_data = safe_get_data(df_metrics, "avg_queue", [0.0] * len(metrics))
            throughput_data = safe_get_data(df_metrics, "throughput", [0.0] * len(metrics))
            epsilon_data = safe_get_data(df_metrics, "epsilon", [0.0] * len(metrics))
            
            # Meta-learning metrics
            meta_epsilon_data = safe_get_data(df_metrics, "meta_epsilon", [None] * len(metrics))
            meta_lr_scale_data = safe_get_data(df_metrics, "meta_lr_scale", [None] * len(metrics))
            time_of_day_data = safe_get_data(df_metrics, "time_of_day", [0.0] * len(metrics))
            global_congestion_data = safe_get_data(df_metrics, "global_congestion", [0.0] * len(metrics))
            
            # Check if meta-learning was used
            has_meta_learning = any(x is not None for x in meta_epsilon_data)
            
            # Calculate moving averages for trend visualization
            window_size = max(5, len(metrics) // 10)
            if len(reward_data) > window_size:
                reward_ma = pd.Series(reward_data).rolling(window=window_size, center=True).mean().tolist()
                loss_ma = pd.Series(loss_data).rolling(window=window_size, center=True).mean().tolist()
                queue_ma = pd.Series(queue_data).rolling(window=window_size, center=True).mean().tolist()
                throughput_ma = pd.Series(throughput_data).rolling(window=window_size, center=True).mean().tolist()
            else:
                reward_ma = reward_data
                loss_ma = loss_data
                queue_ma = queue_data
                throughput_ma = throughput_data
            
            # Calculate learning statistics - compare early vs late episodes
            early_episodes = max(1, len(metrics) // 3)
            late_start = len(metrics) - (len(metrics) // 3)
            
            early_reward = np.mean(reward_data[:early_episodes]) if early_episodes > 0 else 0
            late_reward = np.mean(reward_data[late_start:]) if late_start < len(reward_data) else 0
            reward_improvement = late_reward - early_reward
            
            early_loss_list = [l for l in loss_data[:early_episodes] if l > 0]
            late_loss_list = [l for l in loss_data[late_start:] if l > 0]
            early_loss = np.mean(early_loss_list) if early_loss_list else 0
            late_loss = np.mean(late_loss_list) if late_loss_list else 0
            loss_improvement_pct = ((early_loss - late_loss) / early_loss * 100) if early_loss > 0 else 0
            
            early_queue = np.mean(queue_data[:early_episodes]) if early_episodes > 0 else 0
            late_queue = np.mean(queue_data[late_start:]) if late_start < len(queue_data) else 0
            queue_improvement_pct = ((early_queue - late_queue) / early_queue * 100) if early_queue > 0 else 0
            
            early_throughput = np.mean(throughput_data[:early_episodes]) if early_episodes > 0 else 0
            late_throughput = np.mean(throughput_data[late_start:]) if late_start < len(throughput_data) else 0
            throughput_improvement_pct = ((late_throughput - early_throughput) / early_throughput * 100) if early_throughput > 0 else 0
            
            # Learning Evidence Cards
            st.subheader("📈 Learning Evidence")
            evidence_cols = st.columns(4)
            
            with evidence_cols[0]:
                st.metric("Reward Improvement", f"{reward_improvement:+.1f}", 
                         f"{late_reward:.1f} (late) vs {early_reward:.1f} (early)",
                         help="Reward should increase (become less negative) over time")
                if reward_improvement > 0:
                    st.success("✅ Learning detected")
                else:
                    st.warning("⚠️ Needs more training")
            
            with evidence_cols[1]:
                st.metric("Loss Reduction", f"{loss_improvement_pct:.1f}%", 
                         f"{late_loss:.2f} vs {early_loss:.2f}",
                         help="Training loss should decrease over time. Negative % means loss increased (warning sign)")
                if loss_improvement_pct > 10:
                    st.success("✅ Learning detected")
                elif loss_improvement_pct > 0:
                    st.info("🔄 Learning in progress")
                elif loss_improvement_pct > -20:
                    st.warning("⚠️ Loss increasing - may need more episodes or lower learning rate")
                else:
                    st.error("❌ Loss significantly increasing - check hyperparameters or training stability")
            
            with evidence_cols[2]:
                st.metric("Queue Reduction", f"{queue_improvement_pct:.1f}%", 
                         f"{late_queue:.2f} vs {early_queue:.2f}",
                         help="Average queue should decrease as agent learns")
                if queue_improvement_pct > 5:
                    st.success("✅ Learning detected")
                else:
                    st.info("🔄 Learning in progress")
            
            with evidence_cols[3]:
                st.metric("Throughput Gain", f"{throughput_improvement_pct:.1f}%", 
                         f"{late_throughput:.0f} vs {early_throughput:.0f}",
                         help="Throughput should increase as agent improves")
                if throughput_improvement_pct > 5:
                    st.success("✅ Learning detected")
                else:
                    st.info("🔄 Learning in progress")
            
            # Key Learning Indicators Chart
            st.subheader("🎯 Key Learning Indicators with Trend Lines")
            
            with st.expander("📖 Understanding Trend Lines", expanded=False):
                st.markdown("""
                **What are Trend Lines?**
                
                Each chart shows two things:
                - **Faint lines**: Raw data from each episode (shows day-to-day variations)
                - **Bold lines**: Moving average trend (smooths out noise to show the overall direction)
                
                **What each chart means:**
                
                1. **Reward Trend (Top Left)**: 
                   - Shows average reward per episode (negative because it's -queue_length)
                   - **Upward trend = GOOD**: Means rewards are improving (less negative = queues decreasing)
                   - This proves the agent is learning to reduce traffic congestion
                
                2. **Training Loss (Top Right)**:
                   - Shows how accurately the neural network predicts action values
                   - **Downward trend = GOOD**: Means the network is getting better at predicting which actions are best
                   - This proves the AI is learning from past experiences
                
                3. **Queue Length (Bottom Left)**:
                   - Shows average number of waiting vehicles
                   - **Downward trend = GOOD**: Means traffic is flowing better
                   - This proves the agent is learning to manage traffic more efficiently
                
                4. **Throughput (Bottom Right)**:
                   - Shows number of vehicles that completed their journey
                   - **Upward trend = GOOD**: Means more vehicles are getting through
                   - This proves the agent is learning to move traffic more effectively
                
                **Key Insight**: If all four trends are moving in the right direction, the AI is successfully learning from experience!
                """)
            
            if chart_style == "Plotly (Interactive)":
                fig_learning = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Reward Trend", "Training Loss", "Queue Length", "Throughput"),
                    vertical_spacing=0.15,
                )
                
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=reward_data, mode='lines', name='Reward', line=dict(color='rgba(0,123,255,0.3)', width=1)), row=1, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=reward_ma, mode='lines', name='Trend', line=dict(color='#007bff', width=3)), row=1, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=loss_data, mode='lines', name='Loss', line=dict(color='rgba(243,156,18,0.3)', width=1)), row=1, col=2)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=loss_ma, mode='lines', name='Trend', line=dict(color='#f39c12', width=3)), row=1, col=2)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=queue_data, mode='lines', name='Queue', line=dict(color='rgba(255,107,107,0.3)', width=1)), row=2, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=queue_ma, mode='lines', name='Trend', line=dict(color='#ff6b6b', width=3)), row=2, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=throughput_data, mode='lines', name='Throughput', line=dict(color='rgba(78,205,196,0.3)', width=1)), row=2, col=2)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=throughput_ma, mode='lines', name='Trend', line=dict(color='#4ecdc4', width=3)), row=2, col=2)
                
                fig_learning.update_xaxes(title_text="Episode", row=2, col=1)
                fig_learning.update_xaxes(title_text="Episode", row=2, col=2)
                fig_learning.update_yaxes(title_text="Reward", row=1, col=1)
                fig_learning.update_yaxes(title_text="Loss", row=1, col=2)
                fig_learning.update_yaxes(title_text="Queue", row=2, col=1)
                fig_learning.update_yaxes(title_text="Throughput", row=2, col=2)
                fig_learning.update_layout(height=700, showlegend=False, template="plotly_white")
                st.plotly_chart(fig_learning, width='stretch')
            else:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes[0, 0].plot(episodes_list, reward_data, alpha=0.3, color='#007bff', linewidth=1)
                axes[0, 0].plot(episodes_list, reward_ma, color='#007bff', linewidth=3, label='Trend')
                axes[0, 0].set_title("Reward Trend", fontweight='bold')
                axes[0, 0].set_xlabel("Episode")
                axes[0, 0].set_ylabel("Reward")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                axes[0, 1].plot(episodes_list, loss_data, alpha=0.3, color='#f39c12', linewidth=1)
                axes[0, 1].plot(episodes_list, loss_ma, color='#f39c12', linewidth=3, label='Trend')
                axes[0, 1].set_title("Training Loss", fontweight='bold')
                axes[0, 1].set_xlabel("Episode")
                axes[0, 1].set_ylabel("Loss")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
                axes[1, 0].plot(episodes_list, queue_data, alpha=0.3, color='#ff6b6b', linewidth=1)
                axes[1, 0].plot(episodes_list, queue_ma, color='#ff6b6b', linewidth=3, label='Trend')
                axes[1, 0].set_title("Queue Length", fontweight='bold')
                axes[1, 0].set_xlabel("Episode")
                axes[1, 0].set_ylabel("Queue")
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                axes[1, 1].plot(episodes_list, throughput_data, alpha=0.3, color='#4ecdc4', linewidth=1)
                axes[1, 1].plot(episodes_list, throughput_ma, color='#4ecdc4', linewidth=3, label='Trend')
                axes[1, 1].set_title("Throughput", fontweight='bold')
                axes[1, 1].set_xlabel("Episode")
                axes[1, 1].set_ylabel("Throughput")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            # Exploration vs Exploitation
            st.subheader("🔍 Exploration vs Exploitation")
            exp_cols = st.columns(2)
            
            with exp_cols[0]:
                if epsilon_data and any(epsilon_data):
                    if chart_style == "Plotly (Interactive)":
                        fig_eps = go.Figure()
                        if has_meta_learning and any(x is not None for x in meta_epsilon_data):
                            # Show both traditional and meta epsilon
                            fig_eps.add_trace(go.Scatter(x=episodes_list, y=meta_epsilon_data, mode='lines+markers', 
                                                       name='Meta Epsilon', line=dict(color='#9b59b6', width=2)))
                            fig_eps.add_trace(go.Scatter(x=episodes_list, y=epsilon_data, mode='lines', 
                                                       name='Traditional Epsilon', line=dict(color='#95a5a6', width=1, dash='dash')))
                        else:
                            fig_eps.add_trace(go.Scatter(x=episodes_list, y=epsilon_data, mode='lines+markers', 
                                                       name='Epsilon', line=dict(color='#9b59b6', width=2)))
                        
                        fig_eps.add_hline(y=0.1, line_dash="dash", line_color="gray", annotation_text="Low Exploration")
                        fig_eps.update_layout(title="Exploration Rate Over Time", xaxis_title="Episode", yaxis_title="Epsilon", height=300, template="plotly_white")
                        st.plotly_chart(fig_eps, width='stretch')
            
            with exp_cols[1]:
                st.markdown("#### Learning Process")
                if has_meta_learning:
                    st.info("""
                    **Meta-Learning Exploration:**
                    
                    **Adaptive ε:** Meta-controller adjusts exploration based on performance and context
                    
                    **Context-Aware:** Responds to traffic patterns and learning progress
                    
                    **Evidence:** Meta epsilon adapts → Smarter exploration strategy
                    """)
                else:
                    st.info("""
                    **Traditional Exploration:**
                    
                    **Exploration (High ε):** Agent tries random actions to discover strategies
                    
                    **Exploitation (Low ε):** Agent uses learned knowledge
                    
                    **Evidence:** Epsilon decreases → Agent relies more on learned policy
                    """)
            
            # Meta-learning specific charts
            if has_meta_learning:
                st.subheader("🧠 Meta-Learning Adaptation")
                
                meta_chart_cols = st.columns(2)
                
                with meta_chart_cols[0]:
                    # Learning rate scaling over time
                    if any(x is not None for x in meta_lr_scale_data):
                        if chart_style == "Plotly (Interactive)":
                            fig_lr = go.Figure()
                            fig_lr.add_trace(go.Scatter(x=episodes_list, y=meta_lr_scale_data, mode='lines+markers',
                                                      name='LR Scale', line=dict(color='#e74c3c', width=2)))
                            fig_lr.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline LR")
                            fig_lr.update_layout(title="Learning Rate Scaling", xaxis_title="Episode", 
                                               yaxis_title="LR Scale Factor", height=300, template="plotly_white")
                            st.plotly_chart(fig_lr, width='stretch')
                
                with meta_chart_cols[1]:
                    # Context features over time
                    if chart_style == "Plotly (Interactive)":
                        fig_context = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_context.add_trace(go.Scatter(x=episodes_list, y=time_of_day_data, mode='lines',
                                                       name='Time of Day', line=dict(color='#f39c12', width=2)))
                        fig_context.add_trace(go.Scatter(x=episodes_list, y=global_congestion_data, mode='lines',
                                                       name='Global Congestion', line=dict(color='#3498db', width=2)), 
                                            secondary_y=True)
                        fig_context.update_xaxes(title_text="Episode")
                        fig_context.update_yaxes(title_text="Time of Day (0-1)", secondary_y=False)
                        fig_context.update_yaxes(title_text="Congestion Level", secondary_y=True)
                        fig_context.update_layout(title="Context Features", height=300, template="plotly_white")
                        st.plotly_chart(fig_context, width='stretch')
                
                # Meta-learning performance analysis
                st.markdown("#### 📊 Meta-Learning Performance Analysis")
                
                if len(meta_epsilon_data) > 10:  # Need enough data for analysis
                    # Calculate correlation between context and adaptation
                    valid_indices = [i for i, x in enumerate(meta_epsilon_data) if x is not None]
                    if len(valid_indices) > 5:
                        meta_eps_clean = [meta_epsilon_data[i] for i in valid_indices]
                        congestion_clean = [global_congestion_data[i] for i in valid_indices]
                        
                        if len(meta_eps_clean) > 1 and len(congestion_clean) > 1:
                            correlation = np.corrcoef(meta_eps_clean, congestion_clean)[0, 1]
                            
                            analysis_cols = st.columns(3)
                            with analysis_cols[0]:
                                st.metric("Epsilon-Congestion Correlation", f"{correlation:.3f}",
                                        help="Positive correlation means higher exploration during high congestion")
                            
                            with analysis_cols[1]:
                                avg_meta_eps = np.mean(meta_eps_clean)
                                avg_traditional_eps = np.mean([x for x in epsilon_data if x > 0])
                                st.metric("Avg Meta Epsilon", f"{avg_meta_eps:.3f}",
                                        f"vs {avg_traditional_eps:.3f} traditional")
                            
                            with analysis_cols[2]:
                                if any(x is not None for x in meta_lr_scale_data):
                                    avg_lr_scale = np.mean([x for x in meta_lr_scale_data if x is not None])
                                    st.metric("Avg LR Scale", f"{avg_lr_scale:.3f}",
                                            help="Average learning rate adjustment factor")
                
                with st.expander("📖 Meta-Learning Insights", expanded=False):
                    st.markdown("""
                    **What to Look For:**
                    
                    **Adaptive Exploration**: Meta epsilon should vary based on performance and context, not just decrease linearly.
                    
                    **Learning Rate Scaling**: Should increase (>1.0) when performance is poor, decrease (<1.0) when performance is good.
                    
                    **Context Responsiveness**: Meta-controller should adapt to traffic patterns (time of day, congestion levels).
                    
                    **Performance Correlation**: Better adaptation should correlate with improved performance metrics.
                    
                    **Benefits**: More efficient learning, automatic hyperparameter tuning, better adaptation to changing conditions.
                    """)
            
            # Learning Statistics
            st.subheader("📊 Early vs Late Performance Comparison")
            learning_stats = pd.DataFrame({
                "Metric": ["Reward", "Training Loss", "Queue Length", "Throughput"],
                "Early (First 1/3)": [f"{early_reward:.2f}", f"{early_loss:.4f}", f"{early_queue:.2f}", f"{early_throughput:.0f}"],
                "Late (Last 1/3)": [f"{late_reward:.2f}", f"{late_loss:.4f}", f"{late_queue:.2f}", f"{late_throughput:.0f}"],
                "Improvement": [
                    f"{reward_improvement:+.2f} ({'✅' if reward_improvement > 0 else '⚠️'})",
                    f"{loss_improvement_pct:.1f}% ({'✅' if loss_improvement_pct > 0 else '⚠️'})",
                    f"{queue_improvement_pct:.1f}% ({'✅' if queue_improvement_pct > 0 else '⚠️'})",
                    f"{throughput_improvement_pct:.1f}% ({'✅' if throughput_improvement_pct > 0 else '⚠️'})",
                ],
            })
            st.dataframe(learning_stats, width='stretch', hide_index=True)
            
            # Learning Assessment
            learning_score = sum([reward_improvement > 0, loss_improvement_pct > 10, queue_improvement_pct > 5, throughput_improvement_pct > 5])
            if learning_score == 4:
                st.success("✅ **Strong Learning Detected!** The AI is clearly learning from experience - rewards improving, loss decreasing, performance metrics improving.")
            elif learning_score >= 2:
                st.info("🔄 **Learning in Progress** - Some metrics improving. Consider more episodes for stronger effects.")
            else:
                st.warning("⚠️ **Limited Learning** - May need more episodes, different hyperparameters, or more exploration time.")
        else:
            st.info("No training data available. Run a simulation to see learning analysis.")
    
    # Tab 5: Detailed Metrics
    with tab5:
        st.header("Detailed Metrics Table")
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            
            # Show all available columns
            display_cols = st.multiselect(
                "Select columns to display",
                options=list(df_metrics.columns),
                default=["episode", "avg_queue", "throughput", "avg_travel_time", "loss", "epsilon", "updates"]
            )
            
            if display_cols:
                # Filter display_cols to only include columns that actually exist
                valid_cols = [col for col in display_cols if col in df_metrics.columns]
                if valid_cols:
                    st.dataframe(df_metrics[valid_cols], width='stretch', hide_index=True)
                else:
                    st.warning("Selected columns are not available in the current data.")
            
            # Download button
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics as CSV",
                data=csv,
                file_name="training_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("No metrics data available. Run a simulation first.")
else:
    st.info("👈 Configure and run a simulation using the sidebar to see results here.")

# Auto-refresh
if refresh_enabled:
    time.sleep(refresh_seconds)
    st.rerun()
