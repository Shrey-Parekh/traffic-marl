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

try:
    from .env_sumo import PuneSUMOEnv
    from .config import (
        OUTPUTS_DIR,
        LIVE_METRICS_JSON,
        METRICS_JSON,
        METRICS_CSV,
        SUMMARY_TXT,
        FINAL_REPORT_JSON,
        BASELINE_METRICS_JSON,
        TrainingConfig,
        SCENARIOS,
        STATS_SEEDS,
    )
except ImportError:
    _HERE = Path(__file__).parent
    _ROOT = _HERE.parent
    for p in {str(_HERE), str(_ROOT)}:
        if p not in sys.path:
            sys.path.append(p)
    try:
        from env_sumo import PuneSUMOEnv
        from config import (
            OUTPUTS_DIR,
            LIVE_METRICS_JSON,
            METRICS_JSON,
            METRICS_CSV,
            SUMMARY_TXT,
            FINAL_REPORT_JSON,
            BASELINE_METRICS_JSON,
            TrainingConfig,
            SCENARIOS,
            STATS_SEEDS,
        )
    except ImportError as _e:
        raise ImportError(
            "Failed to import modules. Please run 'streamlit run src/dashboard.py' from the project root."
        ) from _e

# Check SUMO availability
try:
    import traci
    import sumolib
    SUMO_AVAILABLE = True
    try:
        sumo_binary = sumolib.checkBinary('sumo')
        SUMO_BINARY_FOUND = True
    except:
        SUMO_BINARY_FOUND = False
except ImportError:
    SUMO_AVAILABLE = False
    SUMO_BINARY_FOUND = False

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

st.markdown("""
<div class="main-header">
    <h1>Traffic MARL Dashboard</h1>
    <p style="margin:0; font-size:1.1em;">Multi-Agent Reinforcement Learning for Traffic Light Control</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Configuration")

# SUMO Connection Status Indicator
if SUMO_AVAILABLE and SUMO_BINARY_FOUND:
    st.sidebar.markdown('<span style="color: #27AE60; font-size: 1.2em;">●</span> SUMO Connected', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span style="color: #E74C3C; font-size: 1.2em;">●</span> SUMO Not Found', unsafe_allow_html=True)
    st.sidebar.error("Install SUMO: https://sumo.dlr.de/docs/Installing/index.html")

st.sidebar.markdown("---")

if st.sidebar.button("Clear Old Results", help="Clear any cached results from previous simulations"):

    st.session_state.pop("latest_metrics", None)
    st.session_state.pop("latest_live", None)
    st.session_state.pop("latest_final_report", None)
    st.session_state.pop("baseline_result", None)
    st.session_state.pop("comparison_results", None)
    st.session_state.pop("simulation_complete", None)
    st.session_state.pop("comparison_mode", None)
    st.session_state.pop("simulation_params", None)
    st.session_state.pop("baseline_params", None)
    

    import os
    old_files = [METRICS_PATH, LIVE_PATH, FINAL_PATH, OUTPUTS_DIR / "comparison_results.json"]
    for file_path in old_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
    
    st.sidebar.success("Old results cleared!")
    st.rerun()

st.sidebar.markdown("---")

refresh_enabled = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15)
st.sidebar.caption("⏱️ SUMO training: ~2-3 min per episode")

st.sidebar.markdown("---")

st.sidebar.subheader(" Simulation Parameters")
with st.sidebar.form("simulation_form", clear_on_submit=False):
    st.markdown("### Environment Settings")
    with st.expander(" What do these mean?", expanded=False):
        st.markdown("""
        **SUMO Simulation Time**: Duration of each training episode in simulation seconds. Longer episodes provide more data but take more time.
        
        **Scenario**: Traffic pattern to simulate:
        - Uniform: Balanced NS/EW traffic throughout
        - Morning Peak: Higher NS traffic (0-1200s)
        - Evening Peak: Higher EW traffic (2400-3600s)
        
        **Seeds**: Multiple random seeds for statistical analysis. Select multiple for mean±std±CI results.
        """)
    
    # Fixed at 9 intersections (3x3 SUMO grid)
    st.info("🚦 **Network**: 3×3 grid (9 intersections) - Pune urban configuration")
    
    max_steps_input = st.number_input("SUMO Simulation Time (seconds)", min_value=300, max_value=3600, value=600, step=100,
                                     help="Duration of each episode in simulation seconds")
    
    scenario_input = st.selectbox("Traffic Scenario", 
                                  options=SCENARIOS,
                                  index=0,
                                  help="Select traffic pattern: uniform, morning peak, or evening peak")
    
    seeds_input = st.multiselect("Random Seeds (for statistical analysis)",
                                 options=STATS_SEEDS,
                                 default=[STATS_SEEDS[0]],
                                 help="Select multiple seeds for mean±std±95% CI analysis")
    
    st.markdown("### Training Settings")
    with st.expander("Training Parameters Explained", expanded=False):
        st.markdown("""
        **Training Episodes**: Total number of complete simulation runs. Each episode is a full simulation from start to finish. More episodes = more learning but takes longer.
        
        **Batch Size**: How many past experiences the AI uses at once to update its knowledge. Larger = smoother updates but slower. Smaller = faster but noisier.
        
        **Note**: Learning Rate and Discount Factor are automatically optimized based on the selected model architecture for best performance.
        """)
    episodes_input = st.number_input("Training Episodes", min_value=1, max_value=200, value=50,
                                    help="Number of episodes to train")
    batch_size_input = st.number_input("Batch Size", min_value=16, max_value=256, value=32, step=16,
                                      help="Batch size for training")
    

    st.info(" **Optimized Settings**: Learning Rate (0.0001) and Discount Factor (0.99) are automatically configured for optimal performance.")
    
    model_type = st.radio(
        "Model Architecture", 
        ["DQN", "GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C"], 
        index=3,  # Default to GAT-DQN (our novel contribution)
        help="Choose the RL architecture"
    )
    

    st.info(f"**Selected:** {get_model_description(model_type)}")
    
    complexity_indicators = {
        "DQN": " Fast",
        "GNN-DQN": " Medium", 
        "PPO-GNN": " Slow",
        "GAT-DQN": " Medium-Slow (Novel: VehicleClassAttention)",
        "GNN-A2C": " Medium-Slow"
    }
    st.caption(f"Training Speed: {complexity_indicators.get(model_type, ' Medium')}")
    
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

            gat_n_heads = 4
            gat_dropout = 0.1
    
    use_advanced = st.checkbox("Show Advanced Options", value=False)
    if use_advanced:

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

            epsilon_start = 1.0
            epsilon_end = 0.1
            epsilon_decay_steps = 8000
            update_target_steps = 500
        
        min_buffer_size = st.number_input("Min Buffer Size", min_value=100, max_value=5000, value=2000, step=100)
    else:

        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay_steps = 8000
        update_target_steps = 500
        min_buffer_size = 2000
    
    total_time_est = len(seeds_input) * episodes_input * 2.5  # ~2.5 min per episode with SUMO
    st.info(f"Estimated total time: ~{total_time_est:.0f} minutes ({len(seeds_input)} seed(s) × {episodes_input} episodes)")
    

    with st.expander("Time Breakdown Estimate", expanded=False):
        baseline_time_est = max_steps_input * 0.05
        episode_time_est = max_steps_input * 0.1
        

        if model_type == "PPO-GNN":
            episode_time_est *= 1.5
            complexity_note = "PPO requires multiple optimization epochs per episode"
        elif model_type == "GAT-DQN":
            episode_time_est *= 1.3
            complexity_note = "Graph Attention Networks are computationally intensive"
        elif model_type == "GNN-A2C":
            episode_time_est *= 1.4
            complexity_note = "Actor-Critic training requires both policy and value updates"
        elif model_type == "GNN-DQN":
            episode_time_est *= 1.2
            complexity_note = "Graph Neural Networks require message passing computations"
        else:
            complexity_note = "Standard DQN is the fastest architecture"
        
        training_time_est = episode_time_est * episodes_input
        total_time_est = baseline_time_est + training_time_est
        
        st.markdown(f"""
        **Estimated Time Breakdown:**
        - **Baseline:** ~{baseline_time_est:.0f} seconds
        - **AI Training:** ~{training_time_est/60:.1f} minutes ({episode_time_est:.1f}s per episode)
        - **Total:** ~{total_time_est/60:.1f} minutes
        
        **Model Complexity:** {complexity_note}
        
        **Note:** Actual times may vary based on system performance and training dynamics.
        """)
    
    submitted = st.form_submit_button(" Run Simulation", width='stretch')

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
        tuple: (emoji_indicator, help_text) where emoji is , , or 
    """
    if pct_improvement > 15:
        return "", "Excellent (>15% improvement over baseline)"
    elif pct_improvement >= 5:
        return "", "Good (5-15% improvement over baseline)"
    elif pct_improvement >= 0:
        return "", "Needs improvement (<5% improvement over baseline)"
    else:
        return "", "Worse than baseline (negative improvement)"

if submitted:
    max_steps = int(max_steps_input)
    episodes = int(episodes_input)
    N_val = int(N_input)
    seed_val = int(seed_input)
    

    st.session_state.pop("latest_metrics", None)
    st.session_state.pop("latest_live", None)
    st.session_state.pop("latest_final_report", None)
    st.session_state.pop("baseline_result", None)
    st.session_state.pop("comparison_results", None)
    st.session_state.pop("simulation_complete", None)
    st.session_state.pop("comparison_mode", None)
    

    setup_status = st.empty()
    setup_status.info(" Initializing simulation environment and clearing old results...")
    

    import os
    old_files = [METRICS_PATH, LIVE_PATH, FINAL_PATH, OUTPUTS_DIR / "comparison_results.json"]
    for file_path in old_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
    

    status_container = st.container()
    
    with status_container:

        if model_type == "Multi-Model Comparison":
            st.markdown("###  Multi-Model Comparison in Progress")
            st.markdown(f"** Comparing:** DQN, GNN-DQN, PPO-GNN, GAT-DQN, GNN-A2C")
        else:
            st.markdown(f"###  {model_type} Training in Progress")
            st.markdown(f"** Model:** {get_model_description(model_type)}")
        

        phase_col1, phase_col2, phase_col3, phase_col4 = st.columns([1, 1, 1, 1])
        with phase_col1:
            phase1_status = st.empty()
        with phase_col2:
            phase2_status = st.empty()
        with phase_col3:
            phase3_status = st.empty()
        with phase_col4:

            if st.button(" Cancel", key="cancel_sim"):
                st.warning("Simulation cancelled by user.")
                st.stop()
        

        phase1_status.markdown(" **Phase 1:** Baseline")
        if model_type == "Multi-Model Comparison":
            phase2_status.markdown(" **Phase 2:** Multi-Training")
        else:
            phase2_status.markdown(" **Phase 2:** Training")
        phase3_status.markdown(" **Phase 3:** Results")
        

        progress_col1, progress_col2 = st.columns([2, 1])
        
        with progress_col1:

            main_progress = st.progress(0)
            status_text = st.empty()
            

            progress_details = st.empty()
            
        with progress_col2:

            current_model_display = st.empty()
            

            time_info = st.empty()
            stats_info = st.empty()
            

            live_metrics = st.empty()
    

    setup_status.empty()
    

    start_time = time.time()
    
    with status_container:
        status_text.markdown("**Phase 1/3:**  Running baseline controller...")
        

        if model_type == "Multi-Model Comparison":
            current_model_display.markdown(f"""
            ** Multi-Model Comparison**
            
            **Current Phase:** Baseline
            **Next:** Train all 5 models
            **Models:** DQN, GNN-DQN, PPO-GNN, GAT-DQN, GNN-A2C
            """)
        else:
            current_model_display.markdown(f"""
            ** Current Model:**
            
            **{model_type}**
            {get_model_description(model_type)}
            
            **Phase:** Baseline Setup
            """)
        
        progress_details.markdown(f"""
        **Current Task:** Fixed-time baseline simulation
        **Model:** Simple rule-based controller (switch every {baseline_switch_period} steps)
        **Purpose:** Establishing performance benchmark for comparison
        """)
        time_info.markdown(f"""
        **Estimated Time:** ~30 seconds
        **Progress:** Baseline simulation
        **Status:**  Running...
        """)
        main_progress.progress(0.05)
    
    try:
        baseline_start = time.time()
        env_b = MiniTrafficEnv(EnvConfig(
            num_intersections=N_val,
            max_steps=max_steps,
            seed=seed_val,
            arrival_rate_ns=DEFAULT_ARRIVAL_RATE_NS,
            arrival_rate_ew=DEFAULT_ARRIVAL_RATE_EW,
            min_green=DEFAULT_MIN_GREEN,
            step_length=DEFAULT_STEP_LENGTH,
            depart_capacity=DEFAULT_DEPART_CAPACITY,
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
            

            if t_b % 50 == 0:
                baseline_progress = min(0.05 + (t_b / max_steps) * 0.15, 0.2)
                main_progress.progress(baseline_progress)
                elapsed_baseline = time.time() - baseline_start
                remaining_baseline = (elapsed_baseline / t_b) * (max_steps - t_b) if t_b > 0 else 30
                
                progress_details.markdown(f"""
                **Current Task:** Fixed-time baseline simulation
                **Step:** {t_b}/{max_steps} ({(t_b/max_steps)*100:.1f}%)
                **Intersections:** {N_val} traffic lights
                **Switch Period:** Every {baseline_switch_period} steps
                """)
                time_info.markdown(f"""
                **Elapsed:** {elapsed_baseline:.1f}s
                **Remaining:** ~{remaining_baseline:.1f}s
                **Status:**  Running baseline...
                """)
        
        baseline_time = time.time() - baseline_start
        main_progress.progress(0.2)
        

        baseline_result = {
            "avg_queue": info_b.get("avg_queue", 0.0),
            "throughput": info_b.get("throughput", 0.0),
            "avg_travel_time": info_b.get("avg_travel_time", 0.0),
        }
        
        st.session_state["baseline_result"] = baseline_result
        st.session_state["baseline_params"] = {
            "N": N_val,
            "steps": max_steps,
            "seed": seed_val,
            "switch_period": baseline_switch_period,
        }
        st.session_state["current_baseline_period"] = baseline_switch_period
        

        phase1_status.markdown(" **Phase 1:** Baseline")
        if model_type == "Multi-Model Comparison":
            phase2_status.markdown(" **Phase 2:** Multi-Training")
        else:
            phase2_status.markdown(" **Phase 2:** Training")
        
        status_text.markdown("**Phase 1/3:**  Baseline completed successfully!")
        

        if model_type == "Multi-Model Comparison":
            current_model_display.markdown(f"""
            ** Multi-Model Comparison**
            
            **Phase:** Starting Training
            **Next:** Train all 5 models sequentially
            **Total Models:** 5 architectures
            """)
        else:
            current_model_display.markdown(f"""
            ** Ready for Training:**
            
            **{model_type}**
            {get_model_description(model_type)}
            
            **Phase:** AI Training
            """)
        
        progress_details.markdown(f"""
        ** Baseline Results:**
        - **Average Queue:** {baseline_result['avg_queue']:.2f} cars
        - **Throughput:** {baseline_result['throughput']:.0f} vehicles
        - **Travel Time:** {baseline_result['avg_travel_time']:.2f}s
        - **Completion Time:** {baseline_time:.1f}s
        """)
        time_info.markdown(f"""
        **Phase 1 Complete:** 
        **Time Taken:** {baseline_time:.1f}s
        **Status:** Moving to AI training...
        """)
        

        time.sleep(1)
        
    except (ValueError, TypeError, RuntimeError) as e:
        st.error(f"Baseline simulation failed: {str(e)}")
        st.exception(e)
        st.stop()
    

    training_start = time.time()
    
    with status_container:
        if model_type == "Multi-Model Comparison":
            status_text.markdown("**Phase 2/3:**  Training multiple AI models...")
        else:
            status_text.markdown(f"**Phase 2/3:**  Training {model_type} model...")
        

        estimated_episode_time = max_steps * 0.1
        if model_type in ["PPO-GNN", "GNN-A2C"]:
            estimated_episode_time *= 1.5
        elif model_type == "GAT-DQN":
            estimated_episode_time *= 1.3
        elif model_type == "Multi-Model Comparison":
            estimated_episode_time *= 2.5
        
        total_estimated_time = estimated_episode_time * episodes
        

        if model_type == "Multi-Model Comparison":
            current_model_display.markdown(f"""
            ** Multi-Model Training**
            
            **Status:** Training all models
            **Models:** 5 architectures
            **Episodes Each:** {episodes}
            **Current:** Starting...
            """)
        else:
            current_model_display.markdown(f"""
            ** Training Active:**
            
            **{model_type}**
            {get_model_description(model_type)}
            
            **Episodes:** {episodes}
            """)
        
        progress_details.markdown(f"""
        **Current Task:** Training {'multiple models' if model_type == 'Multi-Model Comparison' else model_type + ' model'}
        **Architecture:** {get_model_description(model_type) if model_type != 'Multi-Model Comparison' else 'All 5 architectures'}
        **Episodes:** {episodes} training runs {'per model' if model_type == 'Multi-Model Comparison' else ''}
        **Steps per Episode:** {max_steps}
        """)
        
        time_info.markdown(f"""
        **Estimated Time:** ~{total_estimated_time/60:.1f} minutes
        **Per Episode:** ~{estimated_episode_time:.1f}s
        **Status:**  Preparing training...
        """)
        
        main_progress.progress(0.25)
    

    project_root = OUTPUTS_DIR.parent
    project_root_str = str(project_root.resolve())
    

    train_script_path = project_root / "src" / "train.py"
    train_script_path = train_script_path.resolve()
    

    if not train_script_path.exists():
        st.error(f" Training script not found at: {train_script_path}")
        st.error(f"Project root: {project_root_str}")
        st.error("Please ensure you're running from the project root directory.")
        st.stop()
    

    python_executable = sys.executable
    

    python_executable = sys.executable
    

    if model_type == "Multi-Model Comparison":

        train_script_path = project_root / "src" / "train_comparison.py"
        train_script_path = train_script_path.resolve()
        

        if not train_script_path.exists():
            st.error(f" Comparison training script not found at: {train_script_path}")
            st.error(f"Project root: {project_root_str}")
            st.error("Please ensure you're running from the project root directory.")
            st.stop()
        
        cmd_parts = [
            python_executable, str(train_script_path),
            "--episodes", str(episodes),
            "--N", str(N_val),
            "--max_steps", str(max_steps),
            "--seed", str(seed_val),
            "--batch_size", str(batch_size_input),
            "--save_dir", str(OUTPUTS_DIR.resolve()),
        ]

        
        if use_advanced:
            cmd_parts.extend(["--epsilon_start", str(epsilon_start)])
            cmd_parts.extend(["--epsilon_end", str(epsilon_end)])
            cmd_parts.extend(["--epsilon_decay_steps", str(epsilon_decay_steps)])
            cmd_parts.extend(["--update_target_steps", str(update_target_steps)])
            cmd_parts.extend(["--min_buffer_size", str(min_buffer_size)])
    else:

        train_script_path = project_root / "src" / "train.py"
        train_script_path = train_script_path.resolve()
        

        if not train_script_path.exists():
            st.error(f" Training script not found at: {train_script_path}")
            st.error(f"Project root: {project_root_str}")
            st.error("Please ensure you're running from the project root directory.")
            st.stop()
        
        cmd_parts = [
            python_executable, str(train_script_path),
            "--episodes", str(episodes),
            "--N", str(N_val),
            "--max_steps", str(max_steps),
            "--seed", str(seed_val),
            "--batch_size", str(batch_size_input),
            "--model_type", model_type,
            "--save_dir", str(OUTPUTS_DIR.resolve()),
        ]

        

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
        
        if use_advanced:
            if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
                cmd_parts.extend(["--epsilon_start", str(epsilon_start)])
                cmd_parts.extend(["--epsilon_end", str(epsilon_end)])
                cmd_parts.extend(["--epsilon_decay_steps", str(epsilon_decay_steps)])
                cmd_parts.extend(["--update_target_steps", str(update_target_steps)])
            cmd_parts.extend(["--min_buffer_size", str(min_buffer_size)])
            if neighbor_obs:
                cmd_parts.append("--neighbor_obs")
    
    import subprocess
    try:

        error_log_path = OUTPUTS_DIR / "training_error.log"
        

        env = os.environ.copy()

        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}"
        else:
            env["PYTHONPATH"] = project_root_str
        
        with open(error_log_path, "w", encoding="utf-8") as error_file:

            process = subprocess.Popen(
                cmd_parts, 
                stdout=subprocess.DEVNULL, 
                stderr=error_file, 
                text=True, 
                cwd=project_root_str,
                env=env,
                bufsize=0
            )
        
        training_start_time = time.time()
        timeout = 3600
        last_progress = 0.25
        last_episode_check = 0
        

        while process.poll() is None:
            elapsed = time.time() - training_start_time
            if elapsed > timeout:
                process.kill()
                st.error(f"Training timed out after {timeout} seconds")
                break
            

            current_live = load_json(LIVE_PATH)
            if current_live and "episode" in current_live:
                current_episode = current_live["episode"] + 1  # Episodes are 0-indexed
                current_model_from_live = current_live.get("model_type", model_type)
                actual_progress = 0.25 + (current_episode / episodes) * 0.65
                

                if actual_progress > last_progress:
                    main_progress.progress(min(actual_progress, 0.9))
                    

                    current_queue = current_live.get("avg_queue", 0.0)
                    current_throughput = current_live.get("throughput", 0.0)
                    current_loss = current_live.get("loss", 0.0)
                    current_epsilon = current_live.get("epsilon", 0.0)
                    

                    if model_type == "Multi-Model Comparison":
                        status_text.markdown(f"**Phase 2/3:**  Training {current_model_from_live} - Episode {current_episode}/{episodes}")
                    else:
                        status_text.markdown(f"**Phase 2/3:**  Training {model_type} - Episode {current_episode}/{episodes}")
                    

                    if model_type == "Multi-Model Comparison":
                        current_model_display.markdown(f"""
                        ** Multi-Model Training**
                        
                        **Current Model:** {current_model_from_live}
                        **Description:** {get_model_description(current_model_from_live)}
                        **Episode:** {current_episode}/{episodes}
                        **Progress:** {(current_episode/episodes)*100:.0f}%
                        """)
                    else:
                        current_model_display.markdown(f"""
                        ** Training Active:**
                        
                        **{current_model_from_live}**
                        {get_model_description(current_model_from_live)}
                        
                        **Episode:** {current_episode}/{episodes}
                        **Progress:** {(current_episode/episodes)*100:.0f}%
                        """)
                    
                    progress_details.markdown(f"""
                    **Current Model:** {current_model_from_live}
                    **Episode:** {current_episode}/{episodes} ({(current_episode/episodes)*100:.1f}%)
                    **Live Metrics:**
                    - Queue: {current_queue:.2f} cars
                    - Throughput: {current_throughput:.0f} vehicles  
                    - Loss: {current_loss:.4f}
                    - Epsilon: {current_epsilon:.3f}
                    """)
                    

                    if current_episode > 0:
                        time_per_episode = elapsed / current_episode
                        remaining_episodes = episodes - current_episode
                        estimated_remaining = time_per_episode * remaining_episodes
                        
                        time_info.markdown(f"""
                        **Elapsed:** {elapsed/60:.1f} min
                        **Remaining:** ~{estimated_remaining/60:.1f} min
                        **Avg per Episode:** {time_per_episode:.1f}s
                        **Status:**  Training {current_model_from_live}...
                        """)
                        

                        if baseline_result:
                            baseline_queue = baseline_result.get("avg_queue", 0.0)
                            queue_improvement = ((baseline_queue - current_queue) / baseline_queue * 100) if baseline_queue > 0 else 0
                            
                            live_metrics.markdown(f"""
                            ** Live Performance:**
                            **Model:** {current_model_from_live}
                            
                            **vs Baseline:**
                            - Queue: {queue_improvement:+.1f}%
                            - Status: {' Better' if queue_improvement > 0 else ' Worse' if queue_improvement < -5 else ' Similar'}
                            
                            **Learning:**
                            - Episode: {current_episode}/{episodes}
                            - Progress: {(current_episode/episodes)*100:.0f}%
                            """)
                    
                    last_progress = actual_progress
                    last_episode_check = current_episode
            else:

                estimated_progress = min(0.25 + (elapsed / total_estimated_time) * 0.65, 0.9)
                if estimated_progress > last_progress:
                    main_progress.progress(estimated_progress)
                    estimated_episode = int((estimated_progress - 0.25) / 0.65 * episodes)
                    
                    if model_type == "Multi-Model Comparison":
                        status_text.markdown(f"**Phase 2/3:**  Training multiple models - Episode ~{estimated_episode}/{episodes}")
                        current_model_display.markdown(f"""
                        ** Multi-Model Training**
                        
                        **Status:** Training in progress
                        **Estimated Episode:** ~{estimated_episode}/{episodes}
                        **Models:** All 5 architectures
                        **Note:** Live metrics loading...
                        """)
                    else:
                        status_text.markdown(f"**Phase 2/3:**  Training {model_type} - Episode ~{estimated_episode}/{episodes}")
                        current_model_display.markdown(f"""
                        ** Training Active:**
                        
                        **{model_type}**
                        {get_model_description(model_type)}
                        
                        **Estimated Episode:** ~{estimated_episode}/{episodes}
                        **Status:** Training neural network...
                        """)
                    
                    progress_details.markdown(f"""
                    **Estimated Progress:** {((estimated_progress - 0.25) / 0.65) * 100:.1f}%
                    **Model:** {model_type}
                    **Status:** Training neural network...
                    **Note:** Live metrics will appear once training starts
                    """)
                    
                    remaining_time = total_estimated_time - elapsed
                    time_info.markdown(f"""
                    **Elapsed:** {elapsed/60:.1f} min
                    **Remaining:** ~{remaining_time/60:.1f} min
                    **Status:**  Training...
                    """)
                    
                    last_progress = estimated_progress
            
            time.sleep(1.0)
        

        return_code = process.wait()
        training_time = time.time() - training_start_time
        
        if return_code != 0:

            error_log_path = OUTPUTS_DIR / "training_error.log"
            error_message = "Unknown error occurred."
            if error_log_path.exists():
                try:
                    with open(error_log_path, "r", encoding="utf-8") as f:
                        error_lines = f.readlines()
                        if error_lines:

                            error_message = "".join(error_lines[-20:])
                except (IOError, OSError):
                    error_message = "Could not read error log."
            
            st.error(f"Training failed with return code {return_code}")
            with st.expander(" View Error Details", expanded=True):
                st.code(error_message, language="text")
            st.stop()
        else:
            main_progress.progress(0.9)
            phase2_status.markdown(" **Phase 2:** Training")
            phase3_status.markdown(" **Phase 3:** Results")
            
            if model_type == "Multi-Model Comparison":
                status_text.markdown("**Phase 2/3:**  Multi-model training completed successfully!")
                current_model_display.markdown(f"""
                ** Multi-Model Complete**
                
                **Status:** All models trained
                **Models:** 5 architectures
                **Episodes Each:** {episodes}
                **Next:** Loading comparisons
                """)
            else:
                status_text.markdown(f"**Phase 2/3:**  {model_type} training completed successfully!")
                current_model_display.markdown(f"""
                ** Training Complete:**
                
                **{model_type}**
                {get_model_description(model_type)}
                
                **Status:**  Finished
                **Next:** Loading results
                """)
            
            progress_details.markdown(f"""
            ** Training Complete:**
            - **Model:** {model_type}
            - **Episodes:** {episodes}
            - **Training Time:** {training_time/60:.1f} minutes
            - **Avg per Episode:** {training_time/episodes:.1f}s
            """)
            time_info.markdown(f"""
            **Training Complete:** 
            **Total Time:** {training_time/60:.1f} min
            **Status:** Loading results...
            """)
            
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        st.error(f"Error running training: {str(e)}")
        st.exception(e)
        st.stop()
    

    with status_container:
        phase3_status.markdown(" **Phase 3:** Results")
        
        status_text.markdown("**Phase 3/3:**  Loading and processing results...")
        progress_details.markdown("""
        **Current Task:** Loading training results
        **Processing:** Metrics, final report, and performance data
        **Preparing:** Dashboard visualizations and comparisons
        """)
        time_info.markdown("""
        **Status:**  Loading results...
        **Almost Done:** Preparing dashboard
        """)
        main_progress.progress(0.95)
    

    time.sleep(3)
    

    metrics_path = OUTPUTS_DIR.resolve() / "metrics.json"
    live_path = OUTPUTS_DIR.resolve() / "live_metrics.json"
    final_report_path = OUTPUTS_DIR.resolve() / "final_report.json"
    
    max_wait_time = 20
    wait_start = time.time()
    metrics = None
    live = None
    final_report = None
    
    while time.time() - wait_start < max_wait_time:
        if model_type == "Multi-Model Comparison":
            comparison_results = load_json(OUTPUTS_DIR.resolve() / "comparison_results.json")
            if comparison_results:
                break
        else:
            metrics = load_json(str(metrics_path))
            live = load_json(str(live_path))
            final_report = load_json(str(final_report_path))

            has_metrics = isinstance(metrics, list) and len(metrics) > 0
            has_live = isinstance(live, dict) and "episode" in live
            if has_metrics and has_live:
                break
            if has_metrics or has_live:

                time.sleep(1)
                metrics = load_json(str(metrics_path))
                live = load_json(str(live_path))
                final_report = load_json(str(final_report_path))
                if (isinstance(metrics, list) and len(metrics) > 0) and (isinstance(live, dict) and "episode" in live):
                    break
        time.sleep(0.5)
    else:

        err_detail = ""
        if model_type != "Multi-Model Comparison":
            err_detail = f" Looked in: {metrics_path}"
            if not metrics_path.exists():
                err_detail += " (metrics.json missing)"
            elif not live_path.exists():
                err_detail += " (live_metrics.json missing)"
        st.error(
            "Training completed but new results were not generated. "
            "Please try running the simulation again." + err_detail
        )
        st.stop()
    
    if model_type == "Multi-Model Comparison":

        comparison_results = load_json(OUTPUTS_DIR / "comparison_results.json")
        
        if comparison_results:
            total_time = time.time() - start_time
            

            if not comparison_results.get('results') or len(comparison_results.get('results', {})) == 0:
                st.error(" Comparison results appear to be empty or corrupted. Please try running the simulation again.")
                st.stop()
            

            with status_container:
                phase1_status.markdown(" **Phase 1:** Baseline")
                phase2_status.markdown(" **Phase 2:** Multi-Training")
                phase3_status.markdown(" **Phase 3:** Results")
                
                main_progress.progress(1.0)
                status_text.markdown("** Multi-Model Comparison Complete!** All models trained and compared.")
                
                current_model_display.markdown(f"""
                ** Comparison Complete**
                
                **Status:**  All Done
                **Models Compared:** 5 architectures
                **Best Model:** {comparison_results.get('best_model', 'Loading...')}
                **Ready:** View comparisons below
                """)
                
                progress_details.markdown(f"""
                ** Comparison Results:**
                - **Models Compared:** {len(comparison_results.get('results', {}))} architectures
                - **Episodes per Model:** {episodes}
                - **Best Model:** {comparison_results.get('best_model', 'N/A')}
                - **Total Training Time:** {total_time/60:.1f} minutes
                """)
                
                time_info.markdown(f"""
                ** Complete!**
                **Total Time:** {total_time/60:.1f} minutes
                **Status:**  Ready to explore comparisons!
                """)
                
                time.sleep(2)
            
            st.session_state["comparison_results"] = comparison_results
            st.session_state["simulation_complete"] = True
            st.session_state["comparison_mode"] = True
            st.session_state["simulation_params"] = {
                "N": N_val,
                "episodes": episodes,
                "max_steps": max_steps,
                "seed": seed_val,
                "lr": TrainingConfig.learning_rate,  # Use optimized preset
                "batch_size": batch_size_input,
                "gamma": TrainingConfig.gamma,  # Use optimized preset
                "model_type": model_type,
            }
            

            status_container.empty()
            st.success(f" Multi-Model Comparison completed successfully in {total_time/60:.1f} minutes!")
            st.rerun()
        else:
            st.error(" Comparison completed but results not found or are empty. Check the console output.")
    else:

        metrics = load_json(str(metrics_path))
        live = load_json(str(live_path))
        final_report = load_json(str(final_report_path))
        
        if live and metrics:
            total_time = time.time() - start_time
            

            final_model = live.get("model_type", "Unknown")
            if final_model != model_type and model_type != "Multi-Model Comparison":
                st.error(f"Results show {final_model} but expected {model_type}. This appears to be old data. Please try running the simulation again.")
                st.stop()
            

            if not metrics or len(metrics) == 0:
                st.error("Training results appear to be empty. Please try running the simulation again.")
                st.stop()
            

            with status_container:
                phase1_status.markdown(" **Phase 1:** Baseline")
                phase2_status.markdown(" **Phase 2:** Training")
                phase3_status.markdown(" **Phase 3:** Results")
                
                main_progress.progress(1.0)
                status_text.markdown("** Simulation Complete!** All phases finished successfully.")
                

                final_queue = live.get("avg_queue", 0.0)
                final_throughput = live.get("throughput", 0.0)
                baseline_queue = baseline_result.get("avg_queue", 0.0)
                improvement = ((baseline_queue - final_queue) / baseline_queue * 100) if baseline_queue > 0 else 0
                
                current_model_display.markdown(f"""
                ** Training Complete:**
                
                **{final_model}**
                {get_model_description(final_model)}
                
                **Performance:** {improvement:+.1f}% vs baseline
                **Status:**  Ready to explore!
                """)
                
                progress_details.markdown(f"""
                ** Final Results Summary:**
                - **Model:** {final_model} ({episodes} episodes)
                - **Final Queue:** {final_queue:.2f} cars (vs {baseline_queue:.2f} baseline)
                - **Improvement:** {improvement:+.1f}% queue reduction
                - **Throughput:** {final_throughput:.0f} vehicles
                """)
                
                time_info.markdown(f"""
                ** Complete!**
                **Total Time:** {total_time/60:.1f} minutes
                **Baseline:** {baseline_time:.1f}s
                **Training:** {training_time/60:.1f} min
                **Status:**  Ready to explore results!
                """)
                

                live_metrics.markdown(f"""
                ** Success!**
                **Model:** {final_model}
                
                **Final Performance:**
                - Queue: {final_queue:.2f} cars
                - Improvement: {improvement:+.1f}%
                - Status: {' Better than baseline!' if improvement > 0 else ' Similar to baseline' if improvement > -5 else ' Needs more training'}
                """)
                

                time.sleep(3)
            
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
                "lr": TrainingConfig.learning_rate,  # Use optimized preset
                "batch_size": batch_size_input,
                "gamma": TrainingConfig.gamma,  # Use optimized preset
                "model_type": model_type,
            }
            st.session_state["current_baseline_period"] = baseline_switch_period
            

            status_container.empty()
            st.success(f" Simulation completed successfully in {total_time/60:.1f} minutes! Scroll down to see results.")
            st.rerun()
        else:
            st.error(" Training completed but new results were not found or are incomplete. Please try running the simulation again.")

comparison_results = st.session_state.get("comparison_results", None)
comparison_mode = st.session_state.get("comparison_mode", False)

if st.session_state.get("simulation_complete", False):
    metrics = st.session_state.get("latest_metrics", None)
    live = st.session_state.get("latest_live", None)
    final_report = st.session_state.get("latest_final_report", None)
    baseline_result = st.session_state.get("baseline_result", None)
    baseline_params = st.session_state.get("baseline_params", {})
    simulation_params = st.session_state.get("simulation_params", {})
else:

    metrics = None
    live = None
    final_report = None
    baseline_result = None
    baseline_params = {}
    simulation_params = {}

st.markdown("---")
st.header("Current Configuration & Results")

param_col1, param_col2 = st.columns(2)

with param_col1:
    st.subheader("Simulation Parameters")

    if simulation_params:

        st.markdown(f"""
        **Network Setup:**
        - **Intersections**: {simulation_params.get('N', 'N/A')} traffic lights
        - **Steps per Episode**: {simulation_params.get('max_steps', 'N/A')} steps (each step = 2 seconds)
        - **Random Seed**: {simulation_params.get('seed', 'N/A')} (for reproducible results)
        
        **Model Architecture:**
        - **Type**: {simulation_params.get('model_type', 'N/A')} ({get_model_description(simulation_params.get('model_type', 'DQN'))})
        
        **Training Configuration:**
        - **Episodes**: {simulation_params.get('episodes', 'N/A')} complete training runs
        - **Learning Rate**: {simulation_params.get('lr', 'N/A')} (optimized preset)
        - **Batch Size**: {simulation_params.get('batch_size', 'N/A')} (experiences per update)
        - **Discount Factor**: {simulation_params.get('gamma', 'N/A')} (optimized preset)
        """)
    else:

        st.markdown("""
        **Network Setup:**
        - **Intersections**: Configure in sidebar (default: 6)
        - **Steps per Episode**: Configure in sidebar (default: 300)
        - **Random Seed**: Configure in sidebar (default: 42)
        
        **Model Architecture:**
        - **Type**: Configure in sidebar (default: DQN)
        
        **Training Configuration:**
        - **Episodes**: Configure in sidebar (default: 50)
        - **Learning Rate**: 0.0001 (optimized preset for stability)
        - **Batch Size**: Configure in sidebar (default: 32)
        - **Discount Factor**: 0.99 (optimized preset for long-term planning)
        """)
        st.caption(" Adjust parameters in the sidebar and click 'Run Simulation' to start training. Your settings will appear here after running.")
    
    with st.expander(" What Do These Parameters Mean?", expanded=False):
        st.markdown("""
        **Network Parameters:**
        - **Intersections (N)**: Number of traffic lights in your simulated city. More intersections = more complex coordination needed.
        - **Steps per Episode**: How long each training run lasts. More steps = longer episodes but more learning data.
        - **Random Seed**: Controls randomness. Same seed = same traffic patterns (useful for fair comparisons).
        
        **Training Parameters:**
        - **Episodes**: How many times the AI will practice. More episodes = better learning but takes longer.
        - **Learning Rate**: How quickly the AI adjusts its strategy. Automatically optimized to 0.0001 for stable training.
        - **Batch Size**: How many past experiences the AI reviews at once. Larger = smoother learning.
        - **Discount Factor (γ)**: How much the AI cares about future rewards. Automatically optimized to 0.99 for long-term traffic planning.
        """)

with param_col2:
    st.subheader("Baseline Settings")
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
        st.caption("Baseline runs automatically before AI training for comparison.")
    
    with st.expander(" What Is Baseline?", expanded=False):
        st.markdown("""
        **Baseline Controller** is a simple rule-based traffic light system:
        - Switches lights automatically every fixed number of steps
        - No learning or adaptation
        - Used as a comparison point to see if the AI is actually improving
        
        **Why Compare?** If the AI performs worse than this simple baseline, we know something needs adjustment!
        """)

if live or metrics or baseline_result or comparison_results:
    st.markdown("---")
    
    if comparison_mode and comparison_results:
        st.subheader("Understanding Your Comparison Results")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.markdown("####  Multi-Model Comparison")
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
            """)
        
        with results_col2:
            st.markdown("####  How Rankings Work")
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
        
        with st.expander("Understanding Multi-Model Results", expanded=False):
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
            
            **Use This To:**
            - Choose the best model for your specific traffic scenario
            - Understand trade-offs between different approaches
            - Validate that AI models outperform simple baselines
            """)

st.markdown("---")

if comparison_mode and comparison_results and st.session_state.get("simulation_complete", False):

    tab1, tab2, tab3 = st.tabs(["Overview", "Agent Comparison", "Detailed Results"])
    

    with tab1:
        st.header("Multi-Model Comparison Overview")
        
        if comparison_results.get("best_model"):
            best_model = comparison_results["best_model"]
            st.success(f" **Best Performing Model**: {best_model['name']} (Average Rank Score: {best_model['score']:.2f})")
            

            best_metrics = best_model.get("metrics", {})
            if best_metrics:
                best_cols = st.columns(4)
                best_cols[0].metric("Best Avg Queue", f"{best_metrics.get('avg_queue', 0):.2f}")
                best_cols[1].metric("Best Throughput", f"{best_metrics.get('throughput', 0):.0f}")
                best_cols[2].metric("Best Travel Time", f"{best_metrics.get('avg_travel_time', 0):.2f}s")
                best_cols[3].metric("Models Compared", len(comparison_results.get("models_compared", [])))
        

        st.subheader(" Comparison Summary")
        models_compared = comparison_results.get("models_compared", [])
        episodes_per_model = comparison_results.get("episodes_per_model", 0)
        
        summary_cols = st.columns(2)
        summary_cols[0].metric("Models Compared", len(models_compared))
        summary_cols[1].metric("Episodes per Model", episodes_per_model)
        
        st.markdown(f"**Models Tested**: {', '.join(models_compared)}")
        

        st.subheader(" Performance Overview")
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
                        "Status": " Success"
                    })
                else:
                    overview_data.append({
                        "Model": model_name,
                        "Avg Queue": "N/A",
                        "Throughput": "N/A", 
                        "Travel Time": "N/A",
                        "Status": " Failed"
                    })
            
            if overview_data:
                overview_df = pd.DataFrame(overview_data)
                st.dataframe(overview_df, width='stretch', hide_index=True)
    

    with tab2:
        st.header(" Agent Performance Comparison")
        
        results = comparison_results.get("results", {})
        rankings = comparison_results.get("ranking", {})
        
        if results and rankings:

            st.subheader(" Performance Metrics Comparison")
            
            comparison_data = []
            for model_name, model_results in results.items():
                if "error" not in model_results and "average_metrics" in model_results:
                    avg_metrics = model_results["average_metrics"]
                    comparison_data.append({
                        "Model": model_name,
                        "Avg Queue": avg_metrics.get("avg_queue", 0),
                        "Throughput": avg_metrics.get("throughput", 0),
                        "Travel Time": avg_metrics.get("avg_travel_time", 0),
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                

                if chart_style == "Plotly (Interactive)":

                    fig_comparison = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Average Queue (Lower is Better)", "Throughput (Higher is Better)", 
                                      "Travel Time (Lower is Better)"),
                        horizontal_spacing=0.1,
                    )
                    
                    models = comparison_df["Model"].tolist()
                    

                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Avg Queue"], name="Avg Queue", 
                              marker_color='#ff6b6b', text=comparison_df["Avg Queue"].round(2), textposition='auto'),
                        row=1, col=1
                    )
                    

                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Throughput"], name="Throughput", 
                              marker_color='#4ecdc4', text=comparison_df["Throughput"].round(0), textposition='auto'),
                        row=1, col=2
                    )
                    

                    fig_comparison.add_trace(
                        go.Bar(x=models, y=comparison_df["Travel Time"], name="Travel Time", 
                              marker_color='#45b7d1', text=comparison_df["Travel Time"].round(2), textposition='auto'),
                        row=1, col=3
                    )
                    
                    fig_comparison.update_layout(height=500, showlegend=False, template="plotly_white")
                    fig_comparison.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_comparison, width='stretch')
                else:

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    models = comparison_df["Model"].tolist()
                    
                    axes[0].bar(models, comparison_df["Avg Queue"], color='#ff6b6b')
                    axes[0].set_title("Average Queue (Lower is Better)", fontweight='bold')
                    axes[0].set_ylabel("Cars")
                    axes[0].tick_params(axis='x', rotation=45)
                    
                    axes[1].bar(models, comparison_df["Throughput"], color='#4ecdc4')
                    axes[1].set_title("Throughput (Higher is Better)", fontweight='bold')
                    axes[1].set_ylabel("Vehicles")
                    axes[1].tick_params(axis='x', rotation=45)
                    
                    axes[2].bar(models, comparison_df["Travel Time"], color='#45b7d1')
                    axes[2].set_title("Travel Time (Lower is Better)", fontweight='bold')
                    axes[2].set_ylabel("Seconds")
                    axes[2].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                

                st.subheader(" Performance Rankings")
                
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
                    

                    pivot_df = ranking_df.pivot(index="Model", columns="Metric", values="Rank")
                    pivot_df["Average Rank"] = pivot_df.mean(axis=1).round(2)
                    pivot_df = pivot_df.sort_values("Average Rank")
                    
                    st.dataframe(pivot_df, width='stretch')
                    

                    st.markdown("#### Best Performers by Metric")
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
    

    with tab3:
        st.header("Detailed Comparison Results")
        
        results = comparison_results.get("results", {})
        
        if results:

            selected_model = st.selectbox("Select Model for Detailed View", list(results.keys()))
            
            if selected_model and selected_model in results:
                model_results = results[selected_model]
                
                if "error" in model_results:
                    st.error(f" {selected_model} failed: {model_results['error']}")
                else:
                    st.subheader(f" {selected_model} Detailed Results")
                    

                    avg_metrics = model_results.get("average_metrics", {})
                    if avg_metrics:
                        st.markdown("#### Average Performance")
                        avg_cols = st.columns(4)
                        avg_cols[0].metric("Avg Queue", f"{avg_metrics.get('avg_queue', 0):.2f}")
                        avg_cols[1].metric("Throughput", f"{avg_metrics.get('throughput', 0):.0f}")
                        avg_cols[2].metric("Travel Time", f"{avg_metrics.get('avg_travel_time', 0):.2f}s")
                        avg_cols[3].metric("Episodes", model_results.get("episodes", 0))
                    

                    all_results = model_results.get("all_results", [])
                    if all_results:
                        st.markdown("#### Episode-by-Episode Results")
                        results_df = pd.DataFrame(all_results)
                        

                        display_cols = ["episode", "avg_queue", "throughput", "avg_travel_time"]
                        available_cols = [col for col in display_cols if col in results_df.columns]
                        
                        if available_cols:
                            st.dataframe(results_df[available_cols], width='stretch', hide_index=True)
                            

                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label=f" Download {selected_model} Results as CSV",
                                data=csv,
                                file_name=f"{selected_model.lower().replace('-', '_')}_results.csv",
                                mime="text/csv"
                            )
        else:
            st.info("No detailed results available.")

elif (metrics or live or baseline_result) and st.session_state.get("simulation_complete", False):

    tab1, tab2, tab3, tab4, tab5 = st.tabs([" Overview", " Training Progress", " Comparison", " Learning Analysis", " Detailed Metrics"])
    

    with tab1:
        st.header("Performance Overview")
        
        with st.expander(" Understanding the Metrics", expanded=False):
            st.markdown("""
            **Average Queue**: Average number of vehicles waiting at intersections. Lower is better - means less traffic congestion.
            
            **Throughput**: Total number of vehicles that completed their journeys during the episode. Higher is better - means more traffic is flowing through the system.
            
            **Avg Travel Time**: Average time (in seconds) vehicles spend in the network from entry to exit. Lower is better - means faster trips.
            
            **Training Loss**: How much error the neural network has in predicting action values. Lower is better - means the AI is learning more accurately.
            
            **Epsilon**: Exploration rate (0-1). High values (near 1.0) mean the agent explores randomly; low values (near 0.05) mean it uses learned knowledge.
            
            **Performance Indicators** (when baseline comparison available):
            -  Green: Excellent performance (>15% improvement over baseline)
            -  Yellow: Good performance (5-15% improvement over baseline)
            -  Red: Needs improvement (<5% improvement or worse than baseline)
            """)
        

        if baseline_result and live:
            st.subheader("AI vs Baseline Comparison")
            

            ai_queue = live.get("avg_queue", 0.0)
            ai_throughput = live.get("throughput", 0.0)
            ai_travel_time = live.get("avg_travel_time", 0.0)
            

            queue_improvement, queue_badge = calc_improvement(ai_queue, baseline_result["avg_queue"], False)
            throughput_improvement, throughput_badge = calc_improvement(ai_throughput, baseline_result["throughput"], True)
            travel_improvement, travel_badge = calc_improvement(ai_travel_time, baseline_result["avg_travel_time"], False)
            

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
        

        if final_report:
            st.subheader("Training Summary")
            avg_metrics = final_report.get("average_metrics", {})
            if avg_metrics:
                summary_cols = st.columns(4)
                summary_cols[0].metric("Avg Queue (all episodes)", f"{avg_metrics.get('avg_queue', 0):.2f}")
                summary_cols[1].metric("Avg Throughput", f"{avg_metrics.get('throughput', 0):.0f}")
                summary_cols[2].metric("Avg Travel Time", f"{avg_metrics.get('avg_travel_time', 0):.2f}s")
                summary_cols[3].metric("Episodes Trained", final_report.get("episodes", 0))
    

    with tab2:
        st.header("Training Progress Over Time")
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            episodes_list = safe_get_data(df_metrics, "episode", list(range(len(metrics))))
            

            queue_data = safe_get_data(df_metrics, "avg_queue", [0.0] * len(metrics))
            throughput_data = safe_get_data(df_metrics, "throughput", [0.0] * len(metrics))
            travel_time_data = safe_get_data(df_metrics, "avg_travel_time", [0.0] * len(metrics))
            loss_data = safe_get_data(df_metrics, "loss", [0.0] * len(metrics))
            policy_loss_data = safe_get_data(df_metrics, "policy_loss", [0.0] * len(metrics))
            value_loss_data = safe_get_data(df_metrics, "value_loss", [0.0] * len(metrics))
            epsilon_data = safe_get_data(df_metrics, "epsilon", [0.0] * len(metrics))
            

            model_type_from_data = df_metrics.get("model_type", pd.Series(["DQN"] * len(metrics))).iloc[0] if len(df_metrics) > 0 else "DQN"
            
            if chart_style == "Plotly (Interactive)":

                if model_type_from_data in ["PPO-GNN", "GNN-A2C"]:

                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=("Average Queue Length", "Throughput", "Average Travel Time", 
                                      "Policy Loss", "Value Loss", "Combined Loss"),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.08,
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers', 
                                 name='Queue', line=dict(color='#ff6b6b', width=2)),
                        row=1, col=1
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                                 name='Throughput', line=dict(color='#4ecdc4', width=2)),
                        row=1, col=2
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=travel_time_data, mode='lines+markers',
                                 name='Travel Time', line=dict(color='#45b7d1', width=2)),
                        row=1, col=3
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=policy_loss_data, mode='lines+markers',
                                 name='Policy Loss', line=dict(color='#f39c12', width=2)),
                        row=2, col=1
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=value_loss_data, mode='lines+markers',
                                 name='Value Loss', line=dict(color='#9b59b6', width=2)),
                        row=2, col=2
                    )
                    

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

                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Average Queue Length", "Throughput", "Average Travel Time", "Training Loss"),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1,
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers', 
                                 name='Queue', line=dict(color='#ff6b6b', width=2)),
                        row=1, col=1
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                                 name='Throughput', line=dict(color='#4ecdc4', width=2)),
                        row=1, col=2
                    )
                    

                    fig.add_trace(
                        go.Scatter(x=episodes_list, y=travel_time_data, mode='lines+markers',
                                 name='Travel Time', line=dict(color='#45b7d1', width=2)),
                        row=2, col=1
                    )
                    

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
    

    with tab3:
        st.header("Detailed Comparison: AI vs Baseline")
        
        if baseline_result and metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            

            baseline_avg_queue = baseline_result["avg_queue"]
            baseline_avg_throughput = baseline_result["throughput"]
            baseline_avg_travel = baseline_result["avg_travel_time"]
            

            ai_avg_queue = df_metrics["avg_queue"].mean() if "avg_queue" in df_metrics.columns else 0.0
            ai_avg_throughput = df_metrics["throughput"].mean() if "throughput" in df_metrics.columns else 0.0
            ai_avg_travel = df_metrics["avg_travel_time"].mean() if "avg_travel_time" in df_metrics.columns else 0.0
            ai_final_queue = df_metrics["avg_queue"].iloc[-1] if "avg_queue" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            ai_final_throughput = df_metrics["throughput"].iloc[-1] if "throughput" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            ai_final_travel = df_metrics["avg_travel_time"].iloc[-1] if "avg_travel_time" in df_metrics.columns and len(df_metrics) > 0 else 0.0
            

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
                

                fig_comp.add_trace(
                    go.Scatter(x=episodes_list, y=queue_data, mode='lines+markers',
                             name='AI Queue', line=dict(color='#007bff', width=2)),
                    row=1, col=1
                )
                fig_comp.add_hline(y=baseline_avg_queue, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline", row=1, col=1)
                

                fig_comp.add_trace(
                    go.Scatter(x=episodes_list, y=throughput_data, mode='lines+markers',
                             name='AI Throughput', line=dict(color='#28a745', width=2)),
                    row=1, col=2
                )
                fig_comp.add_hline(y=baseline_avg_throughput, line_dash="dash", line_color="gray",
                                 annotation_text="Baseline", row=1, col=2)
                

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
    

    with tab4:
        st.header(" AI Learning Analysis")
        st.markdown("### Evidence that the AI is learning from experience")
        
        with st.expander(" Understanding Learning Evidence", expanded=False):
            st.markdown("""
            **Learning Evidence Metrics:**
            
            - **Loss Reduction %**: Shows how much the training loss decreased (after warm-up period). Higher percentage means the neural network is learning better predictions.
            
            - **Queue Reduction %**: Shows improvement in queue management (after warm-up period). Higher percentage means the agent learned to reduce traffic congestion.
            
            - **Throughput Gain %**: Shows improvement in moving vehicles through the system (after warm-up period). Higher percentage means more efficient traffic flow.
            
            **Early vs Late Comparison**: We skip the first 5 episodes (warm-up period when the replay buffer is filling) and compare the early learning phase with the late learning phase to see true improvement.
            
            **Why Skip Warm-up?** The first few episodes have artificially low loss because the AI hasn't collected enough data yet. Comparing after warm-up gives a fair assessment of real learning.
            
            **Trend Lines**: The smooth bold lines show moving averages, making it easier to see overall trends despite natural variations between episodes.
            """)
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            episodes_list = safe_get_data(df_metrics, "episode", list(range(len(metrics))))
            

            loss_data = safe_get_data(df_metrics, "loss", [0.0] * len(metrics))
            queue_data = safe_get_data(df_metrics, "avg_queue", [0.0] * len(metrics))
            throughput_data = safe_get_data(df_metrics, "throughput", [0.0] * len(metrics))
            epsilon_data = safe_get_data(df_metrics, "epsilon", [0.0] * len(metrics))
            reward_data = safe_get_data(df_metrics, "avg_reward", [0.0] * len(metrics))
            

            time_of_day_data = safe_get_data(df_metrics, "time_of_day", [0.0] * len(metrics))
            global_congestion_data = safe_get_data(df_metrics, "global_congestion", [0.0] * len(metrics))
            

            window_size = max(5, len(metrics) // 10)
            if len(loss_data) > window_size:
                loss_ma = pd.Series(loss_data).rolling(window=window_size, center=True).mean().tolist()
                queue_ma = pd.Series(queue_data).rolling(window=window_size, center=True).mean().tolist()
                throughput_ma = pd.Series(throughput_data).rolling(window=window_size, center=True).mean().tolist()
                reward_ma = pd.Series(reward_data).rolling(window=window_size, center=True).mean().tolist()
            else:
                loss_ma = loss_data
                queue_ma = queue_data
                throughput_ma = throughput_data
                reward_ma = reward_data
            

            warmup_episodes = min(5, len(metrics) // 4)
            

            usable_episodes = len(metrics) - warmup_episodes
            if usable_episodes > 10:

                early_start = warmup_episodes
                early_end = warmup_episodes + (usable_episodes // 3)
                late_start = len(metrics) - (usable_episodes // 3)
            else:

                early_start = warmup_episodes
                early_end = warmup_episodes + (usable_episodes // 2)
                late_start = warmup_episodes + (usable_episodes // 2)
            

            early_loss_list = [l for l in loss_data[early_start:early_end] if l > 0]
            late_loss_list = [l for l in loss_data[late_start:] if l > 0]
            early_loss = np.mean(early_loss_list) if early_loss_list else 0
            late_loss = np.mean(late_loss_list) if late_loss_list else 0
            loss_improvement_pct = ((early_loss - late_loss) / early_loss * 100) if early_loss > 0 else 0
            

            early_queue = np.mean(queue_data[early_start:early_end]) if early_end > early_start else 0
            late_queue = np.mean(queue_data[late_start:]) if late_start < len(queue_data) else 0
            queue_improvement_pct = ((early_queue - late_queue) / early_queue * 100) if early_queue > 0 else 0
            

            early_throughput = np.mean(throughput_data[early_start:early_end]) if early_end > early_start else 0
            late_throughput = np.mean(throughput_data[late_start:]) if late_start < len(throughput_data) else 0
            throughput_improvement_pct = ((late_throughput - early_throughput) / early_throughput * 100) if early_throughput > 0 else 0
            

            st.subheader(" Learning Evidence")
            evidence_cols = st.columns(3)
            
            with evidence_cols[0]:
                st.metric("Loss Reduction", f"{loss_improvement_pct:.1f}%", 
                         f"{late_loss:.2f} vs {early_loss:.2f}",
                         help="Training loss should decrease over time. Comparison excludes first 5 episodes (warm-up period). Negative % means loss increased.")
                if loss_improvement_pct > 10:
                    st.success(" Learning detected")
                elif loss_improvement_pct > 0:
                    st.info(" Learning in progress")
                elif loss_improvement_pct > -20:
                    st.warning("Loss increasing - may need more episodes or lower learning rate")
                else:
                    st.error(" Loss significantly increasing - check hyperparameters or training stability")
            
            with evidence_cols[1]:
                st.metric("Queue Reduction", f"{queue_improvement_pct:.1f}%", 
                         f"{late_queue:.2f} vs {early_queue:.2f}",
                         help="Average queue should decrease as agent learns")
                if queue_improvement_pct > 5:
                    st.success(" Learning detected")
                else:
                    st.info(" Learning in progress")
            
            with evidence_cols[2]:
                st.metric("Throughput Gain", f"{throughput_improvement_pct:.1f}%", 
                         f"{late_throughput:.0f} vs {early_throughput:.0f}",
                         help="Throughput should increase as agent improves")
                if throughput_improvement_pct > 5:
                    st.success(" Learning detected")
                else:
                    st.info(" Learning in progress")
            

            st.subheader(" Key Learning Indicators with Trend Lines")
            
            with st.expander(" Understanding Trend Lines", expanded=False):
                st.markdown("""
                **What are Trend Lines?**
                
                Each chart shows two things:
                - **Faint lines**: Raw data from each episode (shows day-to-day variations)
                - **Bold lines**: Moving average trend (smooths out noise to show the overall direction)
                
                **What each chart means:**
                
                1. **Training Loss (Top Left)**:
                   - Shows how accurately the neural network predicts action values
                   - **Downward trend = GOOD**: Means the network is getting better at predicting which actions are best
                   - This proves the AI is learning from past experiences
                
                2. **Average Reward (Top Right)**:
                   - Shows the reward the agent receives per episode
                   - **Upward trend = GOOD**: Means the agent is making better decisions
                   - Reward = -(queue_change) + 0.1×cars_served, so higher reward = better traffic management
                
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
                    subplot_titles=("Training Loss", "Average Reward", "Queue Length", "Throughput"),
                    horizontal_spacing=0.12,
                    vertical_spacing=0.15,
                )
                

                fig_learning.add_trace(go.Scatter(x=episodes_list, y=loss_data, mode='lines', name='Loss', line=dict(color='rgba(243,156,18,0.3)', width=1)), row=1, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=loss_ma, mode='lines', name='Trend', line=dict(color='#f39c12', width=3)), row=1, col=1)
                

                fig_learning.add_trace(go.Scatter(x=episodes_list, y=reward_data, mode='lines', name='Reward', line=dict(color='rgba(155,89,182,0.3)', width=1)), row=1, col=2)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=reward_ma, mode='lines', name='Trend', line=dict(color='#9b59b6', width=3)), row=1, col=2)
                

                fig_learning.add_trace(go.Scatter(x=episodes_list, y=queue_data, mode='lines', name='Queue', line=dict(color='rgba(255,107,107,0.3)', width=1)), row=2, col=1)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=queue_ma, mode='lines', name='Trend', line=dict(color='#ff6b6b', width=3)), row=2, col=1)
                

                fig_learning.add_trace(go.Scatter(x=episodes_list, y=throughput_data, mode='lines', name='Throughput', line=dict(color='rgba(78,205,196,0.3)', width=1)), row=2, col=2)
                fig_learning.add_trace(go.Scatter(x=episodes_list, y=throughput_ma, mode='lines', name='Trend', line=dict(color='#4ecdc4', width=3)), row=2, col=2)
                
                fig_learning.update_xaxes(title_text="Episode", row=1, col=1)
                fig_learning.update_xaxes(title_text="Episode", row=1, col=2)
                fig_learning.update_xaxes(title_text="Episode", row=2, col=1)
                fig_learning.update_xaxes(title_text="Episode", row=2, col=2)
                fig_learning.update_yaxes(title_text="Loss", row=1, col=1)
                fig_learning.update_yaxes(title_text="Reward", row=1, col=2)
                fig_learning.update_yaxes(title_text="Queue", row=2, col=1)
                fig_learning.update_yaxes(title_text="Throughput", row=2, col=2)
                fig_learning.update_layout(height=700, showlegend=False, template="plotly_white")
                st.plotly_chart(fig_learning, use_container_width=True)
            else:
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                

                axes[0, 0].plot(episodes_list, loss_data, alpha=0.3, color='#f39c12', linewidth=1)
                axes[0, 0].plot(episodes_list, loss_ma, color='#f39c12', linewidth=3, label='Trend')
                axes[0, 0].set_title("Training Loss", fontweight='bold')
                axes[0, 0].set_xlabel("Episode")
                axes[0, 0].set_ylabel("Loss")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                

                axes[0, 1].plot(episodes_list, reward_data, alpha=0.3, color='#9b59b6', linewidth=1)
                axes[0, 1].plot(episodes_list, reward_ma, color='#9b59b6', linewidth=3, label='Trend')
                axes[0, 1].set_title("Average Reward", fontweight='bold')
                axes[0, 1].set_xlabel("Episode")
                axes[0, 1].set_ylabel("Reward")
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
            

            st.subheader(" Exploration vs Exploitation")
            exp_cols = st.columns(2)
            
            with exp_cols[0]:
                if epsilon_data and any(epsilon_data):
                    if chart_style == "Plotly (Interactive)":
                        fig_eps = go.Figure()
                        fig_eps.add_trace(go.Scatter(x=episodes_list, y=epsilon_data, mode='lines+markers', 
                                                   name='Epsilon', line=dict(color='#9b59b6', width=2)))
                        
                        fig_eps.add_hline(y=0.1, line_dash="dash", line_color="gray", annotation_text="Low Exploration")
                        fig_eps.update_layout(title="Exploration Rate Over Time", xaxis_title="Episode", yaxis_title="Epsilon", height=300, template="plotly_white")
                        st.plotly_chart(fig_eps, width='stretch')
            
            with exp_cols[1]:
                st.markdown("#### Learning Process")
                st.info("""
                **Traditional Exploration:**
                
                **Exploration (High ε):** Agent tries random actions to discover strategies
                
                **Exploitation (Low ε):** Agent uses learned knowledge
                
                **Evidence:** Epsilon decreases → Agent relies more on learned policy
                """)
            

            st.subheader(" Early vs Late Performance Comparison (Excluding Warm-up)")
            

            st.caption(f"Comparing Episodes {early_start+1}-{early_end} (Early Learning) vs Episodes {late_start+1}-{len(metrics)} (Late Learning) | Warm-up period (Episodes 1-{warmup_episodes}) excluded")
            
            learning_stats = pd.DataFrame({
                "Metric": ["Training Loss", "Queue Length", "Throughput"],
                "Early Learning": [f"{early_loss:.4f}", f"{early_queue:.2f}", f"{early_throughput:.0f}"],
                "Late Learning": [f"{late_loss:.4f}", f"{late_queue:.2f}", f"{late_throughput:.0f}"],
                "Improvement": [
                    f"{loss_improvement_pct:.1f}%",
                    f"{queue_improvement_pct:.1f}%",
                    f"{throughput_improvement_pct:.1f}%",
                ],
            })
            st.dataframe(learning_stats, width='stretch', hide_index=True)
            

            learning_score = sum([loss_improvement_pct > 10, queue_improvement_pct > 5, throughput_improvement_pct > 5])
            if learning_score == 3:
                st.success(" **Strong Learning Detected!** The AI is clearly learning from experience - loss decreasing, performance metrics improving.")
            elif learning_score >= 2:
                st.info(" **Learning in Progress** - Some metrics improving. Consider more episodes for stronger effects.")
            else:
                st.warning("**Limited Learning** - May need more episodes, different hyperparameters, or more exploration time.")
        else:
            st.info("No training data available. Run a simulation to see learning analysis.")
    

    with tab5:
        st.header("Detailed Metrics Table")
        
        if metrics and len(metrics) > 0:
            df_metrics = pd.DataFrame(metrics)
            

            display_cols = st.multiselect(
                "Select columns to display",
                options=list(df_metrics.columns),
                default=["episode", "avg_queue", "throughput", "avg_travel_time", "loss", "epsilon", "updates"]
            )
            
            if display_cols:

                valid_cols = [col for col in display_cols if col in df_metrics.columns]
                if valid_cols:
                    st.dataframe(df_metrics[valid_cols], width='stretch', hide_index=True)
                else:
                    st.warning("Selected columns are not available in the current data.")
            

            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label=" Download Metrics as CSV",
                data=csv,
                file_name="training_metrics.csv",
                mime="text/csv"
            )
        else:
            st.info("No metrics data available. Run a simulation first.")
else:

    if st.session_state.get("simulation_complete", False):
        st.warning("Simulation data was lost. Please run a new simulation.")
    else:
        st.info(" Configure and run a simulation using the sidebar to see results here.")
        

        st.markdown("""

        
        **What happens when you run a simulation:**
        1. **Baseline Test** - Simple rule-based controller establishes performance benchmark
        2. **AI Training** - Your selected model learns to optimize traffic flow
        3. **Results Analysis** - Compare AI performance vs baseline with detailed metrics
        
        **Choose your model above and click "Run Simulation" to begin!**
        """)

if refresh_enabled:
    time.sleep(refresh_seconds)
    st.rerun()
