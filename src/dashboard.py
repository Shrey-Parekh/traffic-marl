from __future__ import annotations

import json
import os
import time
from typing import Any, List, Dict

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Additional imports for interactive runs
import torch

# Robust import handling for Streamlit execution
try:
    from .env import MiniTrafficEnv, EnvConfig  # type: ignore
    from .agent import DQNet  # type: ignore
except Exception:
    import sys as _sys
    _HERE = os.path.dirname(__file__)
    _ROOT = os.path.dirname(_HERE)
    for p in {_HERE, _ROOT}:
        if p not in _sys.path:
            _sys.path.append(p)
    try:
        from env import MiniTrafficEnv, EnvConfig  # type: ignore
        from agent import DQNet  # type: ignore
    except Exception as _e:
        raise ImportError(
            "Failed to import env/agent. Please run 'streamlit run src/dashboard.py' from the project root."
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
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


st.set_page_config(
    page_title="Mini Traffic MARL Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #1e1e1e;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #00d4aa;
    margin: 0.5rem 0;
}
.explanation-box {
    background-color: #2d2d2d;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #444;
    margin: 1rem 0;
}
.success-metric {
    color: #00d4aa;
    font-weight: bold;
}
.warning-metric {
    color: #ff6b6b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("Mini Traffic MARL Dashboard")
st.markdown("### AI-Powered Traffic Light Control System")

st.sidebar.header("Settings")
refresh_enabled = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh every (s)", min_value=3, max_value=30, value=5)
show_explanations = st.sidebar.checkbox("Show explanations", value=False)
chart_style = st.sidebar.selectbox("Chart style", ["Plotly", "Matplotlib"], index=0)

with st.sidebar.expander("Help", expanded=False):
    st.markdown("- Use the form below to run a one-off training run. A fixed-time baseline will be simulated automatically for comparison.")

# Load data files
live = load_json(LIVE_PATH)
metrics = load_json(METRICS_PATH)

# Interactive one-off simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("Run a one-off simulation")
with st.sidebar.form("interactive_run_form"):
    N_input = st.number_input("Intersections (N)", min_value=1, max_value=20, value=6)
    seed_input = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
    # Fixed switch period for baseline
    episodes_input = st.number_input("Episodes (learning only)", min_value=1, max_value=50, value=10)
    st.info("Each episode runs for ~15 seconds (300 steps × 2s/step). Baseline switch period is fixed at 20 steps.")
    submitted = st.form_submit_button("Run Simulation")

if submitted:
    # Fixed episode duration: 300 steps × 2s = 15 seconds per episode
    max_steps = 300
    
    # 1) Run baseline (fixed-time) with given period
    env_b = MiniTrafficEnv(EnvConfig(num_intersections=int(N_input), max_steps=max_steps, seed=int(seed_input)))
    obs_b = env_b.reset()
    done_b = False
    t_b = 0
    info_b: Dict[str, Any] = {}
    BASELINE_SWITCH_PERIOD = 20
    while not done_b:
        do_sw = 1 if (t_b % BASELINE_SWITCH_PERIOD == 0 and t_b > 0) else 0
        act_b = {aid: do_sw for aid in obs_b.keys()}
        obs_b, rewards_b, done_b, info_b = env_b.step(act_b)
        t_b += 1

    # 2) Run learning as a subprocess with progress tracking
    episodes = int(episodes_input)
    N_val = int(N_input)
    seed_val = int(seed_input)
    
    # Do not clear previous results; keep historical data
    
    cmd = f"python -m src.train --episodes {episodes} --N {N_val} --max_steps {max_steps} --seed {seed_val}"
    st.info(f"Running Multi-Agent RL training: {cmd}")
    
    # Show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run training and track progress
    import subprocess
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Monitor training progress
    episode_count = 0
    while process.poll() is None:
        time.sleep(0.5)
        if episode_count < episodes:
            episode_count += 1
            progress = episode_count / episodes
            progress_bar.progress(progress)
            status_text.text(f"Training Episode {episode_count}/{episodes} - Learning Multi-Agent Policy...")
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        st.error(f"Training failed: {stderr}")
    else:
        progress_bar.progress(1.0)
        status_text.text("Training completed! Loading results...")

    # 3) Load AI results and refresh data
    time.sleep(1)  # Give files time to be written
    
    # Reload data files
    live = load_json(LIVE_PATH)
    metrics = load_json(METRICS_PATH)
    
    info_ai: Dict[str, Any] | None = None
    try:
        with open(os.path.join(OUTPUTS_DIR, "live_metrics.json"), "r", encoding="utf-8") as f:
            info_ai = json.load(f)
    except Exception as e:
        st.error(f"Training finished but could not read outputs/live_metrics.json: {e}")
        info_ai = None

    if info_ai is not None:
        st.session_state["interactive_info"] = info_ai
        st.session_state["interactive_baseline_info"] = info_b
        st.session_state["interactive_params"] = {
            "N": int(N_input),
            "steps": max_steps,
            "seed": int(seed_input),
            "period": 20,
            "episodes": int(episodes_input),
        }
        st.success(f"✅ Multi-Agent RL Training Completed! {episodes} episodes learned.")
        st.rerun()  # Refresh the dashboard to show new results
    else:
        st.error("❌ Training completed but no results found. Check the training logs.")

"""
Main content layout
Tabs: Overview | Progress | Intersections
"""

# Render last interactive results if present
if "interactive_info" in st.session_state:
    info = st.session_state["interactive_info"]
    params = st.session_state.get("interactive_params", {})

    tab_overview, tab_intersections = st.tabs(["Overview", "Intersections"])

    with tab_overview:
        st.caption(f"Params: {params}")
        # Top KPI cards
        kpi = st.columns(3)
        kpi[0].metric("Avg Queue", f"{info.get('avg_queue', 0.0):.2f}")
        kpi[1].metric("Throughput", f"{info.get('throughput', 0.0):.0f}")
        kpi[2].metric("Avg Travel Time (s)", f"{info.get('avg_travel_time', 0.0):.2f}")

        # Side-by-side comparison when available
        if "interactive_baseline_info" in st.session_state:
            info_b = st.session_state["interactive_baseline_info"]
            dq = info.get("avg_queue", 0.0) - info_b.get("avg_queue", 0.0)
            dtt = info.get("avg_travel_time", 0.0) - info_b.get("avg_travel_time", 0.0)
            dth = info.get("throughput", 0.0) - info_b.get("throughput", 0.0)

            st.markdown("#### Comparison (AI vs Fixed-Time)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Queue — AI", f"{info.get('avg_queue', 0.0):.2f}", f"{dq:+.2f}")
                st.caption(f"Fixed-Time: {info_b.get('avg_queue', 0.0):.2f}")
            with c2:
                st.metric("Travel Time — AI", f"{info.get('avg_travel_time', 0.0):.2f}s", f"{dtt:+.2f}s")
                st.caption(f"Fixed-Time: {info_b.get('avg_travel_time', 0.0):.2f}s")
            with c3:
                st.metric("Throughput — AI", f"{info.get('throughput', 0.0):.0f}", f"{dth:+.0f}")
                st.caption(f"Fixed-Time: {info_b.get('throughput', 0.0):.0f}")

            # Compact table view
            st.markdown("#### Details")
            df_cmp = pd.DataFrame([
                {"Metric": "Avg Queue", "AI": round(info.get("avg_queue", 0.0), 2), "Fixed-Time": round(info_b.get("avg_queue", 0.0), 2), "Delta (AI - FT)": round(dq, 2)},
                {"Metric": "Avg Travel Time (s)", "AI": round(info.get("avg_travel_time", 0.0), 2), "Fixed-Time": round(info_b.get("avg_travel_time", 0.0), 2), "Delta (AI - FT)": round(dtt, 2)},
                {"Metric": "Throughput", "AI": int(info.get("throughput", 0.0)), "Fixed-Time": int(info_b.get("throughput", 0.0)), "Delta (AI - FT)": int(dth)},
            ])
            st.dataframe(df_cmp, width='stretch', hide_index=True)

    with tab_intersections:
        st.subheader("Per-Intersection Analysis")


# Episode graphs (moved from right column)
if metrics is None:
    st.info("No training history yet. Run a simulation to see charts.")
else:
    episodes = [m.get("episode", i) for i, m in enumerate(metrics)]
    avg_queue = [m.get("avg_queue", 0.0) for m in metrics]
    throughput = [m.get("throughput", 0.0) for m in metrics]
    avg_tt = [m.get("avg_travel_time", 0.0) for m in metrics]

    if chart_style == "Plotly":
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Average Queue Length", "Throughput", "Average Travel Time"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=avg_queue, mode='lines+markers', name='Avg Queue', line=dict(color='#ff6b6b')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=throughput, mode='lines+markers', name='Throughput', line=dict(color='#4ecdc4')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=avg_tt, mode='lines+markers', name='Avg Travel Time', line=dict(color='#45b7d1')),
            row=1, col=3
        )
        fig.update_layout(height=400, showlegend=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=1, col=3)
        fig.update_yaxes(title_text="Cars", row=1, col=1)
        fig.update_yaxes(title_text="Vehicles", row=1, col=2)
        fig.update_yaxes(title_text="Seconds", row=1, col=3)
        st.plotly_chart(fig, width='stretch', key="progress_plotly_main")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(episodes, avg_queue, label="Avg Queue", color='#ff6b6b', linewidth=2)
        axes[0].set_title("Average Queue Length", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cars")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(episodes, throughput, label="Throughput", color='#4ecdc4', linewidth=2)
        axes[1].set_title("Throughput", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Vehicles")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(episodes, avg_tt, label="Avg Travel Time", color='#45b7d1', linewidth=2)
        axes[2].set_title("Average Travel Time", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Seconds")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

st.subheader("Run History")
if metrics is None:
    st.info("No training history yet. Run a simulation to populate.")
else:
    # Build a concise history table from metrics.json (latest first)
    def _fmt(v, d=2):
        try:
            return round(float(v), d)
        except Exception:
            return v
    rows = []
    for rec in metrics:
        rows.append({
            "Episode": int(rec.get("episode", 0)) + 1,
            "Agents": int(rec.get("agents", 0)) if rec.get("agents") is not None else None,
            "Epsilon": _fmt(rec.get("epsilon", None), 3) if rec.get("epsilon") is not None else None,
            "Avg Queue": _fmt(rec.get("avg_queue", 0.0)),
            "Throughput": int(rec.get("throughput", 0.0)),
            "Avg Travel Time (s)": _fmt(rec.get("avg_travel_time", 0.0)),
            "Loss": _fmt(rec.get("loss", 0.0), 4),
            "Updates": int(rec.get("updates", 0)),
        })
    df_hist = pd.DataFrame(rows)
    df_hist = df_hist.sort_values(by=["Episode"], ascending=False)
    st.dataframe(df_hist, width='stretch', hide_index=True)
    with st.expander("What the columns mean", expanded=False):
        st.markdown("""
        - **Episode**: The episode index (1-based).
        - **Agents**: Number of learning agents (equals intersections `N`).
        - **Epsilon**: Exploration rate used that episode (lower means more exploitation).
        - **Avg Queue**: Average number of waiting cars per intersection (lower is better).
        - **Throughput**: Total vehicles that completed their journeys during the episode (higher is better).
        - **Avg Travel Time (s)**: Average time a vehicle spent in the network (lower is better).
        - **Loss**: Average DQN training loss for that episode (lower is generally better, but scale depends on training).
        - **Updates**: Number of gradient updates applied during the episode.
        """)

# Prefer the most recent interactive run if available; otherwise use live_metrics.json for per-intersection details
latest = st.session_state.get("interactive_info", live)


# Per-intersection details if present in latest live record
if latest and isinstance(latest.get("per_int_avg_queue"), list):
    st.subheader("Per-Intersection Analysis")
    per_q = latest.get("per_int_avg_queue", [])
    per_t = latest.get("per_int_throughput", [])
    
    # Create DataFrame for better table display
    df = pd.DataFrame({
        "Intersection": [f"Junction {i+1}" for i in range(len(per_q))],
        "Avg Queue": [f"{q:.2f}" for q in per_q],
        "Throughput": [int(t) if i < len(per_t) else 0 for i, t in enumerate(per_t)],
        "Status": ["Good" if q < np.mean(per_q) else "Moderate" if q < np.mean(per_q) * 1.5 else "Congested" for q in per_q]
    })
    
    st.dataframe(df, width='stretch')
    
    # Enhanced bar charts
    if chart_style == "Plotly":
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average Queue by Intersection", "Throughput by Intersection"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig2.add_trace(
            go.Bar(x=[f"J{i+1}" for i in range(len(per_q))], y=per_q, name="Avg Queue", marker_color='#ff6b6b'),
            row=1, col=1
        )
        fig2.add_trace(
            go.Bar(x=[f"J{i+1}" for i in range(len(per_t))], y=per_t, name="Throughput", marker_color='#4ecdc4'),
            row=1, col=2
        )
        
        fig2.update_layout(height=400, showlegend=False, template="plotly_dark")
        fig2.update_xaxes(title_text="Intersection", row=1, col=1)
        fig2.update_xaxes(title_text="Intersection", row=1, col=2)
        fig2.update_yaxes(title_text="Cars", row=1, col=1)
        fig2.update_yaxes(title_text="Vehicles", row=1, col=2)
        
        st.plotly_chart(fig2, width='stretch', key="per_intersection_plotly")
    else:
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 4))
        axes2[0].bar(range(len(per_q)), per_q, color='#ff6b6b', alpha=0.7)
        axes2[0].set_title("Average Queue by Intersection", fontweight='bold')
        axes2[0].set_xlabel("Intersection")
        axes2[0].set_ylabel("Cars")
        axes2[0].set_xticks(range(len(per_q)))
        axes2[0].set_xticklabels([f"J{i+1}" for i in range(len(per_q))])
        axes2[0].grid(True, axis='y', alpha=0.3)
        
        axes2[1].bar(range(len(per_t)), per_t, color='#4ecdc4', alpha=0.7)
        axes2[1].set_title("Throughput by Intersection", fontweight='bold')
        axes2[1].set_xlabel("Intersection")
        axes2[1].set_ylabel("Vehicles")
        axes2[1].set_xticks(range(len(per_t)))
        axes2[1].set_xticklabels([f"J{i+1}" for i in range(len(per_t))])
        axes2[1].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)


# Removed: previous baseline file comparison with emojis; comparison now shown above using fresh results

# Removed: sidebar quick comparison tool and greedy policy mode to simplify UI

# Manual refresh button and controlled auto-refresh
st.markdown("---")
if refresh_enabled:
    st.caption(f"Auto-refreshing every {refresh_seconds} seconds...")
    time.sleep(refresh_seconds)
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()