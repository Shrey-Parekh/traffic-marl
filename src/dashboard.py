from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Robust import handling for Streamlit execution
try:
    from .env import MiniTrafficEnv, EnvConfig  # type: ignore
except ImportError:
    import sys as _sys
    _HERE = os.path.dirname(__file__)
    _ROOT = os.path.dirname(_HERE)
    for p in {_HERE, _ROOT}:
        if p not in _sys.path:
            _sys.path.append(p)
    try:
        from env import MiniTrafficEnv, EnvConfig  # type: ignore
    except ImportError as _e:
        raise ImportError(
            "Failed to import env. Please run 'streamlit run src/dashboard.py' from the project root."
        ) from _e


OUTPUTS_DIR = "outputs"
LIVE_PATH = os.path.join(OUTPUTS_DIR, "live_metrics.json")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")
CSV_PATH = os.path.join(OUTPUTS_DIR, "metrics.csv")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "summary.txt")
FINAL_PATH = os.path.join(OUTPUTS_DIR, "final_report.json")
BASELINE_PATH = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except (IOError, json.JSONDecodeError):
        return None


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
    
    use_advanced = st.checkbox("Show Advanced Options", value=False)
    if use_advanced:
        epsilon_start = st.number_input("Epsilon Start", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        epsilon_end = st.number_input("Epsilon End", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
        epsilon_decay_steps = st.number_input("Epsilon Decay Steps", min_value=100, max_value=20000, value=5000, step=500)
        min_buffer_size = st.number_input("Min Buffer Size", min_value=100, max_value=5000, value=1000, step=100)
        neighbor_obs = st.checkbox("Enable Neighbor Observations", value=False)
    
    st.info(f"⏱️ Estimated time: ~{episodes_input * max_steps_input * 2 / 60:.1f} minutes per run")
    
    submitted = st.form_submit_button("🚀 Run Simulation", use_container_width=True)

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
        env_b = MiniTrafficEnv(EnvConfig(num_intersections=N_val, max_steps=max_steps, seed=seed_val))
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
        
    except Exception as e:
        st.error(f"❌ Baseline simulation failed: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Step 2: Run training
    with status_container:
        status_text.text("🤖 Training AI controller...")
        progress_bar.progress(0.3)
    
    cmd_parts = [
        "python", "-m", "src.train",
        "--episodes", str(episodes),
        "--N", str(N_val),
        "--max_steps", str(max_steps),
        "--seed", str(seed_val),
        "--lr", str(lr_input),
        "--batch_size", str(batch_size_input),
        "--gamma", str(gamma_input),
    ]
    
    if use_advanced:
        cmd_parts.extend(["--epsilon_start", str(epsilon_start)])
        cmd_parts.extend(["--epsilon_end", str(epsilon_end)])
        cmd_parts.extend(["--epsilon_decay_steps", str(epsilon_decay_steps)])
        cmd_parts.extend(["--min_buffer_size", str(min_buffer_size)])
        if neighbor_obs:
            cmd_parts.append("--neighbor_obs")
    
    cmd = " ".join(cmd_parts)
    
    import subprocess  # noqa: S404
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.getcwd())  # noqa: S602
        
        start_time = time.time()
        timeout = 3600
        last_progress = 0.3
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.kill()
                st.error(f"⏱️ Training timed out after {timeout} seconds")
                break
            time.sleep(1.0)
            # Estimate progress
            estimated_progress = min(0.3 + (elapsed / (episodes * max_steps * 0.01)) * 0.65, 0.95)
            if estimated_progress > last_progress:
                progress_bar.progress(estimated_progress)
                status_text.text(f"🤖 Training... Episode ~{int((estimated_progress - 0.3) / 0.65 * episodes)}/{episodes}")
                last_progress = estimated_progress
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            st.error(f"❌ Training failed with return code {process.returncode}")
            if stderr:
                st.code(stderr[:2000])
        else:
            progress_bar.progress(1.0)
            status_text.text("✅ Training completed! Loading results...")
            
    except Exception as e:
        st.error(f"❌ Error running training: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Step 3: Load results
    time.sleep(1)
    metrics = load_json(METRICS_PATH)
    live = load_json(LIVE_PATH)
    final_report = load_json(FINAL_PATH)
    
    if live:
        st.session_state["latest_metrics"] = metrics
        st.session_state["latest_live"] = live
        st.session_state["latest_final_report"] = final_report
        st.session_state["simulation_complete"] = True
        st.session_state["simulation_params"] = {
            "N": N_val,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed_val,
            "lr": lr_input,
            "batch_size": batch_size_input,
            "gamma": gamma_input,
        }
        st.success("✅ Simulation completed successfully!")
        st.rerun()
    else:
        st.error("❌ Training completed but results not found. Check the console output.")

# Load data
metrics = st.session_state.get("latest_metrics", load_json(METRICS_PATH))
live = st.session_state.get("latest_live", load_json(LIVE_PATH))
final_report = st.session_state.get("latest_final_report", load_json(FINAL_PATH))
baseline_result = st.session_state.get("baseline_result", None)
baseline_params = st.session_state.get("baseline_params", {})

# Main content tabs
if metrics or live or baseline_result:
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
            """)
        
        # Latest AI results
        if live:
            st.subheader("🤖 AI Controller - Latest Episode")
            ai_cols = st.columns(4)
            
            ai_queue = live.get("avg_queue", 0.0)
            ai_throughput = live.get("throughput", 0.0)
            ai_travel_time = live.get("avg_travel_time", 0.0)
            ai_loss = live.get("loss", 0.0)
            
            ai_cols[0].metric("Average Queue", f"{ai_queue:.2f}", help="Lower is better - fewer waiting vehicles")
            ai_cols[1].metric("Throughput", f"{ai_throughput:.0f}", help="Higher is better - more vehicles served")
            ai_cols[2].metric("Avg Travel Time", f"{ai_travel_time:.2f}s", help="Lower is better - faster journeys")
            ai_cols[3].metric("Training Loss", f"{ai_loss:.4f}", help="Lower is better - more accurate predictions")
        
        # Comparison if baseline exists
        if baseline_result and live:
            st.subheader("📊 AI vs Baseline Comparison")
            
            # Calculate improvements
            queue_improvement, queue_badge = calc_improvement(ai_queue, baseline_result["avg_queue"], False)
            throughput_improvement, throughput_badge = calc_improvement(ai_throughput, baseline_result["throughput"], True)
            travel_improvement, travel_badge = calc_improvement(ai_travel_time, baseline_result["avg_travel_time"], False)
            
            # Comparison metrics
            comp_cols = st.columns(3)
            
            with comp_cols[0]:
                st.metric(
                    "Average Queue",
                    f"{ai_queue:.2f}",
                    f"{queue_improvement:+.2f} ({baseline_result['avg_queue']:.2f} baseline)",
                    help="Lower is better"
                )
                st.markdown(queue_badge, unsafe_allow_html=True)
            
            with comp_cols[1]:
                st.metric(
                    "Throughput",
                    f"{ai_throughput:.0f}",
                    f"{throughput_improvement:+.0f} ({baseline_result['throughput']:.0f} baseline)",
                    help="Higher is better"
                )
                st.markdown(throughput_badge, unsafe_allow_html=True)
            
            with comp_cols[2]:
                st.metric(
                    "Travel Time",
                    f"{ai_travel_time:.2f}s",
                    f"{travel_improvement:+.2f}s ({baseline_result['avg_travel_time']:.2f}s baseline)",
                    help="Lower is better"
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
                st.plotly_chart(fig_comp, use_container_width=True)
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
            episodes_list = df_metrics.get("episode", range(len(metrics))).tolist()
            
            # Extract metrics
            queue_data = df_metrics.get("avg_queue", [0.0] * len(metrics)).tolist()
            throughput_data = df_metrics.get("throughput", [0.0] * len(metrics)).tolist()
            travel_time_data = df_metrics.get("avg_travel_time", [0.0] * len(metrics)).tolist()
            loss_data = df_metrics.get("loss", [0.0] * len(metrics)).tolist()
            epsilon_data = df_metrics.get("epsilon", [0.0] * len(metrics)).tolist()
            
            if chart_style == "Plotly (Interactive)":
                # Create subplots
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Epsilon decay
                if epsilon_data and any(epsilon_data):
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
                    st.plotly_chart(fig_eps, use_container_width=True)
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
            
            # Get AI metrics
            ai_avg_queue = df_metrics["avg_queue"].mean()
            ai_avg_throughput = df_metrics["throughput"].mean()
            ai_avg_travel = df_metrics["avg_travel_time"].mean()
            ai_final_queue = df_metrics["avg_queue"].iloc[-1] if len(df_metrics) > 0 else 0
            ai_final_throughput = df_metrics["throughput"].iloc[-1] if len(df_metrics) > 0 else 0
            ai_final_travel = df_metrics["avg_travel_time"].iloc[-1] if len(df_metrics) > 0 else 0
            
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
            st.dataframe(comparison_table, use_container_width=True, hide_index=True)
            
            # Side-by-side comparison charts
            st.subheader("Learning Curves with Baseline Reference")
            episodes_list = df_metrics.get("episode", range(len(metrics))).tolist()
            queue_data = df_metrics.get("avg_queue", [0.0] * len(metrics)).tolist()
            throughput_data = df_metrics.get("throughput", [0.0] * len(metrics)).tolist()
            travel_time_data = df_metrics.get("avg_travel_time", [0.0] * len(metrics)).tolist()
            
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
                st.plotly_chart(fig_comp, use_container_width=True)
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
            episodes_list = df_metrics.get("episode", range(len(metrics))).tolist()
            
            # Extract key learning metrics
            reward_data = df_metrics.get("avg_reward", [0.0] * len(metrics)).tolist()
            loss_data = df_metrics.get("loss", [0.0] * len(metrics)).tolist()
            queue_data = df_metrics.get("avg_queue", [0.0] * len(metrics)).tolist()
            throughput_data = df_metrics.get("throughput", [0.0] * len(metrics)).tolist()
            epsilon_data = df_metrics.get("epsilon", [0.0] * len(metrics)).tolist()
            
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
                         help="Training loss should decrease over time")
                if loss_improvement_pct > 10:
                    st.success("✅ Learning detected")
                elif loss_improvement_pct > 0:
                    st.info("🔄 Learning in progress")
                else:
                    st.warning("⚠️ Check training")
            
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
                st.plotly_chart(fig_learning, use_container_width=True)
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
                        fig_eps.add_trace(go.Scatter(x=episodes_list, y=epsilon_data, mode='lines+markers', name='Epsilon', line=dict(color='#9b59b6', width=2), fill='tozeroy', fillcolor='rgba(155,89,182,0.2)'))
                        fig_eps.add_hline(y=0.1, line_dash="dash", line_color="gray", annotation_text="Low Exploration")
                        fig_eps.update_layout(title="Exploration Rate Over Time", xaxis_title="Episode", yaxis_title="Epsilon", height=300, template="plotly_white")
                        st.plotly_chart(fig_eps, use_container_width=True)
            
            with exp_cols[1]:
                st.markdown("#### Learning Process")
                st.info("""
                **Exploration (High ε):** Agent tries random actions to discover strategies
                
                **Exploitation (Low ε):** Agent uses learned knowledge
                
                **Evidence:** Epsilon decreases → Agent relies more on learned policy
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
            st.dataframe(learning_stats, use_container_width=True, hide_index=True)
            
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
                options=df_metrics.columns.tolist(),
                default=["episode", "avg_queue", "throughput", "avg_travel_time", "loss", "epsilon", "updates"]
            )
            
            if display_cols:
                st.dataframe(df_metrics[display_cols], use_container_width=True, hide_index=True)
            
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
