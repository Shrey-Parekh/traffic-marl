# Phase 3 Dashboard Specification - COMPLETED CHANGES

## Summary of Surgical Modifications

### ✅ COMPLETED: Sidebar Updates

1. **SUMO Connection Status** - Added at top of sidebar
   - Green dot (●) = SUMO Connected
   - Red dot (●) = SUMO Not Found
   - Error message with installation link if not found

2. **Environment Settings** - Updated
   - ❌ REMOVED: Number of Intersections slider
   - ✅ FIXED: 9 intersections (3×3 SUMO grid) with info banner
   - ✅ CHANGED: "Steps per Episode" → "SUMO Simulation Time (seconds)"
   - ✅ ADDED: Scenario selector (Uniform / Morning Peak / Evening Peak)
   - ✅ ADDED: Seeds multi-select [1,2,3,4,5] for statistical analysis
   - ❌ REMOVED: Single seed number input

3. **Training Settings** - Updated
   - ❌ REMOVED: Baseline Switch Period
   - ✅ KEPT: Episodes, Batch Size inputs
   - ✅ UPDATED: Time estimation for SUMO (~2.5 min per episode)

4. **Model Selection** - Updated
   - ❌ REMOVED: "Multi-Model Comparison" from radio buttons
   - ✅ KEPT: DQN, GNN-DQN, PPO-GNN, GAT-DQN, GNN-A2C
   - ✅ DEFAULT: GAT-DQN (our novel contribution)
   - ✅ UPDATED: GAT-DQN description shows "VehicleClassAttention"

5. **Advanced Options** - Updated
   - ❌ REMOVED: Neighbor observations toggle
   - ✅ KEPT: DQN epsilon settings, buffer size

6. **Refresh Settings** - Updated
   - ✅ CHANGED: Default refresh from 5s → 15s
   - ✅ CHANGED: Range from 3-30s → 5-60s
   - ✅ ADDED: Caption "SUMO training: ~2-3 min per episode"
   - ❌ REMOVED: Chart style selector (Matplotlib option)

7. **Imports** - Updated
   - ❌ REMOVED: MiniTrafficEnv, EnvConfig
   - ✅ ADDED: PuneSUMOEnv
   - ✅ ADDED: SCENARIOS, STATS_SEEDS from config
   - ✅ ADDED: SUMO availability checks (traci, sumolib)

## 🚧 REMAINING WORK: Main Panel 4-Tab Structure

The main panel needs to be rebuilt with 4 tabs. Here's the complete specification:

### Tab 1: "Training & Results"

**Top Section:**
- SUMO connection status banner (if not connected, show red warning)
- Scenario badge showing which scenario is running (Uniform/Morning Peak/Evening Peak)

**Training Flow** (keep existing subprocess approach):
- Progress bar
- Real-time metrics from live_metrics.json
- Episode counter
- Live queue (show BOTH raw and PCU)
- Throughput, loss, epsilon

**Post-Training Results:**
- Main metrics table with columns: Metric | Value | vs FixedTime | vs Webster | vs MaxPressure
- Metrics to show:
  - Queue (Raw)
  - Queue (PCU) ← PRIMARY METRIC
  - Throughput
  - Travel Time
  - Episode Reward
- Color-coded improvement badges (green >10%, yellow 5-10%, red <5%)

**Multi-Seed Results** (if len(seeds) > 1):
- Per-seed results table
- Mean ± Std row at bottom
- 95% CI shown in parentheses

### Tab 2: "Traffic Analysis"

**Chart 1: Vehicle Class Composition** (Stacked Bar)
- X-axis: Intersection ID (0-8)
- Y-axis: Queue length in PCU
- Stacks: two_wheeler, auto_rickshaw, car, pedestrian_group
- Colors: Orange, Yellow, Blue, Green (matching SUMO colors)
- Data source: Extract from metrics_all_seeds.json vehicle class counts

**Chart 2: PCU vs Raw Queue** (Dual-Axis Line)
- X-axis: Episode steps
- Y-axis Left: Raw queue count
- Y-axis Right: PCU equivalent
- Two lines showing the difference
- Demonstrates why PCU matters

**Chart 3: Peak Hour Effect** (Area Chart)
- X-axis: Simulation time (0-3600s)
- Y-axis: Queue length (PCU)
- Two areas: NS queue (blue), EW queue (red)
- Shaded bands: Morning peak (0-1200s), Evening peak (2400-3600s)
- Only show if scenario != "uniform"

**Graceful Handling:**
- If no training data: Show message "Run training first to see traffic analysis"
- Use st.session_state["training_results"] to check

### Tab 3: "Baselines Comparison"

**Top Section:**
- "Run All Baselines" button
  - Triggers: `python src/baseline.py --episodes 10 --scenario {scenario} --n_intersections 9`
  - Shows progress spinner
  - Loads baseline_metrics.json when complete

**Comparison Table:**
| Metric | FixedTime | Webster | MaxPressure | Your RL Model |
|--------|-----------|---------|-------------|---------------|
| Queue (PCU) | 8.5 | 6.8 | 5.2 | 4.4 ± 0.3 |
| Throughput | 125 | 148 | 168 | 185 ± 6 |
| Travel Time | 45.2 | 38.5 | 32.1 | 28.3 ± 2.1 |

- Bold the best value in each row
- Show RL with mean ± std if multi-seed

**Improvement Chart** (Horizontal Bar):
- Y-axis: Baseline names
- X-axis: % improvement over baseline
- Three bars per baseline (one per metric)
- Color: Green if RL better, Red if RL worse

**Graceful Handling:**
- If no baseline data: Show "Click 'Run All Baselines' to compare"
- If no RL data: Show "Run training first"

### Tab 4: "Publication Stats"

**Auto-Load Section:**
- Check if `outputs/statistical_summary.json` exists
- If yes, load and display automatically
- If no, show "Run multi-seed training to generate statistics"

**Statistics Table:**
| Metric | Mean | Std | 95% CI Lower | 95% CI Upper |
|--------|------|-----|--------------|--------------|
| Queue (PCU) | 4.42 | 0.28 | 4.14 | 4.70 |
| Throughput | 185.3 | 5.8 | 179.5 | 191.1 |
| Travel Time | 28.3 | 2.1 | 26.2 | 30.4 |

**LaTeX Table Generator:**
- Button: "Generate IEEE LaTeX Table"
- Outputs in st.code() block with copy button
- Format:
```latex
\begin{table}[h]
\centering
\caption{Performance Comparison: Baselines vs GAT-DQN with VehicleClassAttention}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Controller & Queue (PCU) & Throughput & Travel Time (s) \\
\midrule
FixedTime & 8.5 $\pm$ 0.8 & 125 $\pm$ 12 & 45.2 $\pm$ 3.1 \\
Webster & 6.8 $\pm$ 0.6 & 148 $\pm$ 10 & 38.5 $\pm$ 2.8 \\
MaxPressure & 5.2 $\pm$ 0.4 & 168 $\pm$ 8 & 32.1 $\pm$ 2.3 \\
\textbf{GAT-DQN (Ours)} & \textbf{4.4 $\pm$ 0.3} & \textbf{185 $\pm$ 6} & \textbf{28.3 $\pm$ 2.1} \\
\bottomrule
\end{tabular}
\end{table}
```
- Auto-bold best values
- Include paper title in caption

**Download Buttons:**
- metrics_all_seeds.json
- statistical_summary.json
- final_report.json
- comparison_results.json (if exists)

## Session State Keys

Use these for persistence across tabs:

```python
st.session_state["training_results"]  # Dict with RL results
st.session_state["baseline_results"]  # Dict with baseline results
st.session_state["statistical_summary"]  # Dict with stats
st.session_state["scenario"]  # Current scenario
st.session_state["seeds"]  # List of seeds used
st.session_state["model_type"]  # Model architecture
```

## Implementation Notes

1. **Subprocess Training** - Keep existing approach:
   ```python
   cmd = [
       sys.executable, "src/train.py",
       "--model_type", model_type,
       "--episodes", str(episodes),
       "--scenario", scenario,
       "--seeds", ",".join(map(str, seeds)),
       "--max_steps", str(max_steps),
   ]
   subprocess.Popen(cmd, ...)
   ```

2. **Live Metrics Reading** - Keep existing:
   ```python
   live_data = load_json(LIVE_METRICS_JSON)
   if live_data:
       current_episode = live_data.get("episode", 0)
       current_queue_pcu = live_data.get("avg_queue_pcu", 0.0)
       # Update progress bar
   ```

3. **Plotly Only** - All charts use plotly.graph_objects:
   ```python
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots
   
   fig = go.Figure()
   fig.add_trace(go.Bar(...))
   st.plotly_chart(fig, use_container_width=True)
   ```

4. **Error Handling** - Graceful degradation:
   ```python
   if not SUMO_AVAILABLE:
       st.error("SUMO not installed. Visit: https://sumo.dlr.de/docs/Installing/index.html")
       st.stop()
   ```

## Files to Update

1. ✅ **src/dashboard.py** - Sidebar modifications COMPLETE
2. 🚧 **src/dashboard.py** - Main panel 4-tab structure PENDING
3. 🚧 **src/train.py** - Add --seeds and --scenario arguments PENDING
4. 🚧 **src/train_comparison.py** - Update for PuneSUMOEnv PENDING

## Testing Checklist

After implementation:
- [ ] SUMO status indicator shows correct state
- [ ] Scenario selector works
- [ ] Multi-seed selection works
- [ ] Training subprocess launches correctly
- [ ] Live metrics update during training
- [ ] Tab 1 shows PCU and raw metrics
- [ ] Tab 2 charts render with data
- [ ] Tab 3 baseline comparison works
- [ ] Tab 4 LaTeX generator produces valid output
- [ ] All session state persists across tab switches
- [ ] Graceful handling when no data available

## Next Steps

1. Complete main panel 4-tab structure in dashboard.py
2. Update train.py for --seeds and --scenario arguments
3. Update train_comparison.py for PuneSUMOEnv
4. Test end-to-end workflow
5. Generate sample results for paper

---

**Status**: Sidebar modifications complete. Main panel 4-tab structure specification ready for implementation.
