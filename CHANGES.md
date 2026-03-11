# SUMO-Based Indian Mixed Traffic Transformation

## Overview
This document tracks the transformation from a generic MARL demo to a research-grade IEEE publication-ready system modeling Indian mixed traffic using SUMO.

## SUMO Network Setup

### Generate SUMO Network
Before running the system, compile the SUMO network:

```bash
cd sumo_config
netconvert -c pune_network.netccfg
```

This generates `pune_network.net.xml` from the node, edge, and type definitions.

## Major Changes

### 1. SUMO Configuration Files (sumo_config/)
- **pune_network.nod.xml**: 3×3 grid of 9 intersections (200m spacing)
- **pune_network.edg.xml**: Bidirectional edges, 2 lanes per direction, 50 km/h limit
- **pune_network.typ.xml**: Urban Pune road type definition
- **pune_vehicles.rou.xml**: 4 vehicle types (two_wheeler, auto_rickshaw, car, pedestrian_group)
- **pune_network.sumocfg**: Master SUMO configuration
- **pune_network.netccfg**: netconvert configuration

### 2. Environment Replacement
- **DELETED**: `src/env.py` (queue-based simulation)
- **CREATED**: `src/env_sumo.py` (SUMO-based PuneSUMOEnv)

### 3. Key Features of PuneSUMOEnv
- Full SUMO integration via traci
- 4 Indian vehicle classes with PCU weighting
- Peak hour asymmetry (morning/evening)
- Non-lane-based flow (two-wheeler lane-splitting)
- 3-phase signal logic (NS_GREEN, ALL_RED_CLEARANCE, EW_GREEN)
- 15-feature observation space per intersection
- PCU-based reward function

### 4. Agent Architecture Updates
- Added VehicleClassAttention module to GAT-DQN
- Dynamic observation space reading from env
- 3-action space (keep_phase, switch_phase, force_clearance)
- All models support new observation dimensions

### 5. Baseline Controllers
- **FixedTimeController**: Cyclic phase switching
- **WebsterOptimalController**: Webster's formula with dynamic adaptation
- **MaxPressureController**: Pressure-based reactive control
- All use PCU-equivalent queue values

### 6. Training Enhancements
- Multi-seed support (--seeds argument)
- Multi-scenario support (morning_peak, evening_peak, uniform, all)
- Statistical rigor: mean, std, 95% CI
- Outputs: metrics_all_seeds.json, statistical_summary.json

### 7. Dashboard Overhaul
- 5 tabs: Training, Traffic Analysis, Baselines Comparison, Model Comparison, Publication Stats
- Vehicle class breakdown visualization
- PCU vs raw queue comparison
- Peak hour visualization
- LaTeX table generator for IEEE papers
- SUMO connection status indicator

### 8. Configuration Centralization
- VEHICLE_CLASSES with PCU weights
- PEAK_HOUR_CONFIG for scenarios
- BASELINE_CONFIG for controller parameters
- SUMO_CONFIG for simulation settings
- OBS_FEATURES_PER_AGENT = 15

### 9. Requirements
Added:
- scipy (statistical analysis)
- traci (SUMO Python interface)
- sumolib (SUMO utilities)

## File Structure

```
traffic-marl/
├── sumo_config/
│   ├── pune_network.nod.xml
│   ├── pune_network.edg.xml
│   ├── pune_network.typ.xml
│   ├── pune_vehicles.rou.xml
│   ├── pune_network.sumocfg
│   ├── pune_network.netccfg
│   └── pune_network.net.xml (generated)
├── src/
│   ├── env_sumo.py (NEW - replaces env.py)
│   ├── agent.py (UPDATED)
│   ├── baseline.py (REWRITTEN)
│   ├── config.py (EXTENDED)
│   ├── train.py (UPDATED)
│   ├── train_comparison.py (UPDATED)
│   └── dashboard.py (REBUILT)
├── outputs/
│   ├── metrics_all_seeds.json
│   ├── statistical_summary.json
│   └── comparison_results.json
└── CHANGES.md (this file)
```

## Research Novelty

**Title**: "Heterogeneous Mixed-Traffic Signal Control using Graph Attention Networks: A Case Study on Pune Urban Intersections"

**Core Contribution**: Explicit modeling of Indian mixed traffic (two-wheelers, autos, pedestrians, cars) with non-lane-based flow dynamics in SUMO-based simulation — addressing a gap in existing MARL traffic literature.

## Usage

### 1. Setup SUMO Network
```bash
cd sumo_config
netconvert -c pune_network.netccfg
cd ..
```

### 2. Run Training
```bash
# Single model, single seed
python src/train.py --model_type GAT-DQN --episodes 50 --N 9

# Multi-seed statistical training
python src/train.py --model_type GAT-DQN --episodes 50 --seeds "1,2,3,4,5" --scenario all

# Multi-model comparison
python src/train_comparison.py --episodes 50
```

### 3. Run Dashboard
```bash
streamlit run src/dashboard.py
```

### 4. Run Baselines
```bash
python src/baseline.py --episodes 20
```

## Verification Checklist

- [x] SUMO config files created and valid
- [x] env.py deleted
- [x] env_sumo.py created with PuneSUMOEnv
- [x] agent.py updated for dynamic obs dims
- [x] baseline.py rewritten with 3 controllers
- [x] train.py supports multi-seed + multi-scenario
- [x] dashboard.py rebuilt with 5 tabs
- [x] config.py extended with all parameters
- [x] requirements.txt includes traci, sumolib, scipy
- [x] No env.py imports remain in codebase

## Notes

- SUMO must be installed and in PATH
- Run `netconvert -c sumo_config/pune_network.netccfg` before first use
- All traci calls wrapped in try/except for robustness
- Dashboard runs headless by default (render toggle available)
- Statistical outputs ready for IEEE paper submission
