
import os
import re
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────
EXPORTS_FOLDER = "output exports"
LAST_N_FRACTION = 0.15    # average over last 15% of episodes per seed
SEEDS_TO_USE = [1, 2, 3]  # which seeds to include

# Metrics to extract — map CSV column → paper label
METRICS = {
    "avg_queue":        "Avg Queue (PCU)",
    "avg_travel_time":  "Avg Travel Time (s)",
    "throughput":       "Throughput (veh)",
}

# Optional metric — included if present in CSV
OPTIONAL_METRICS = {
    "waiting_time":     "Avg Waiting Time (s)",
}

# Model display order for table
MODEL_ORDER = [
    "DQN",
    "GNN-DQN",
    "GAT-DQN-Base",
    "GAT-DQN",
    "ST-GAT",
    "Fed-ST-GAT",
]

# ── Helper functions ────────────────────────────────────────────────

def find_csv_files(exports_folder: str) -> dict:
    """
    Scans exports folder and returns dict:
    {model_name: {seed_num: csv_filepath}}
    
    Handles folder names like: DQN_1, GNN-DQN_2, Fed-ST-GAT_3
    """
    model_files = defaultdict(dict)
    
    if not os.path.exists(exports_folder):
        print(f"ERROR: Folder '{exports_folder}' not found.")
        print(f"Make sure you run this script from the project root.")
        return {}
    
    for folder_name in os.listdir(exports_folder):
        folder_path = os.path.join(exports_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Parse model name and seed from folder name
        # Pattern: MODEL_NAME_SEED (e.g. DQN_1, GNN-DQN_2, Fed-ST-GAT_3)
        match = re.match(r'^(.+)_(\d+)$', folder_name)
        if not match:
            continue
        
        model_name = match.group(1)
        seed_num   = int(match.group(2))
        
        # Find the metrics CSV in this folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"  WARNING: No CSV found in {folder_path}")
            continue
        
        # Use the metrics CSV (not any other CSV)
        metrics_csv = [f for f in csv_files if "metrics" in f.lower()]
        if metrics_csv:
            model_files[model_name][seed_num] = metrics_csv[0]
        else:
            model_files[model_name][seed_num] = csv_files[0]
    
    return dict(model_files)


def extract_last_n_mean(csv_path: str, last_n_fraction: float,
                         metrics: list) -> dict:
    """
    Reads CSV, takes last (last_n_fraction * total_episodes) rows,
    returns mean per metric.
    Returns None if file cannot be read.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return None

    if len(df) == 0:
        print(f"  WARNING: Empty CSV at {csv_path}")
        return None

    last_n = max(1, int(len(df) * last_n_fraction))
    last_n_df = df.tail(last_n)

    result = {}
    for metric in metrics:
        if metric in last_n_df.columns:
            result[metric] = float(last_n_df[metric].mean())
        else:
            result[metric] = None

    return result


def compute_model_stats(model_name: str,
                         seed_files: dict,
                         seeds_to_use: list,
                         last_n_fraction: float) -> dict:
    """
    For one model, computes mean ± std across seeds.
    Returns dict of {metric: {"mean": x, "std": x, "n_seeds": x}}
    """
    all_metrics = list(METRICS.keys()) + list(OPTIONAL_METRICS.keys())

    seed_values = defaultdict(list)
    seeds_used  = []

    for seed in seeds_to_use:
        if seed not in seed_files:
            print(f"  WARNING: {model_name} seed {seed} not found — skipping")
            continue

        csv_path = seed_files[seed]
        values   = extract_last_n_mean(csv_path, last_n_fraction, all_metrics)

        if values is None:
            continue

        seeds_used.append(seed)
        for metric, val in values.items():
            if val is not None:
                seed_values[metric].append(val)
    
    if not seeds_used:
        return None
    
    stats = {"n_seeds": len(seeds_used), "seeds_used": seeds_used}
    
    for metric in all_metrics:
        vals = seed_values.get(metric, [])
        if vals:
            stats[metric] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "values": vals,
            }
        else:
            stats[metric] = None
    
    return stats


def format_cell(stats: dict, metric: str) -> str:
    """Format a metric as 'mean ± std' or 'N/A'."""
    if stats is None or metric not in stats or stats[metric] is None:
        return "N/A"
    
    mean = stats[metric]["mean"]
    std  = stats[metric]["std"]
    n    = stats["n_seeds"]
    
    if metric == "throughput":
        return f"{mean:.0f} ± {std:.0f}"
    elif metric in ["avg_queue"]:
        return f"{mean:.2f} ± {std:.2f}"
    else:
        return f"{mean:.1f} ± {std:.1f}"


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PAPER TABLE IV METRIC EXTRACTOR")
    print(f"Exports folder: '{EXPORTS_FOLDER}'")
    print(f"Last {int(LAST_N_FRACTION * 100)}% of episodes per seed, seeds {SEEDS_TO_USE}")
    print("=" * 65)
    print()
    
    # Find all CSV files
    model_files = find_csv_files(EXPORTS_FOLDER)
    
    if not model_files:
        print("No model files found. Check folder structure.")
        return
    
    print(f"Found models: {sorted(model_files.keys())}")
    print()
    
    # Compute stats for each model
    all_results = {}
    
    for model_name in MODEL_ORDER:
        if model_name not in model_files:
            print(f"  {model_name}: NOT FOUND — skipping")
            continue
        
        print(f"Processing {model_name}...")
        seed_files = model_files[model_name]
        print(f"  Seeds available: {sorted(seed_files.keys())}")
        
        stats = compute_model_stats(
            model_name, seed_files, SEEDS_TO_USE, LAST_N_FRACTION
        )
        
        if stats is None:
            print(f"  {model_name}: No valid data found")
            continue
        
        all_results[model_name] = stats
        print(f"  Seeds used: {stats['seeds_used']}")
        
        for metric, label in METRICS.items():
            if stats.get(metric):
                m = stats[metric]["mean"]
                s = stats[metric]["std"]
                print(f"  {label}: {m:.3f} ± {s:.3f}")
        print()
    
    # ── Print formatted paper table ─────────────────────────────────
    metrics_list     = list(METRICS.keys())
    metrics_labels   = list(METRICS.values())
    
    # Check if waiting_time is available in any model
    has_waiting = any(
        r.get("waiting_time") is not None
        for r in all_results.values()
        if r is not None
    )
    
    if has_waiting:
        metrics_list.append("waiting_time")
        metrics_labels.append("Avg Waiting (s)")
    
    print()
    print("=" * 65)
    print("TABLE IV — RESULTS FOR PAPER (copy-paste ready)")
    print("=" * 65)
    
    # Header
    header = f"{'Model':<16}"
    for label in metrics_labels:
        header += f" {label:>18}"
    print(header)
    print("-" * (16 + 19 * len(metrics_list)))
    
    for model_name in MODEL_ORDER:
        if model_name not in all_results:
            continue
        stats = all_results[model_name]
        row   = f"{model_name:<16}"
        for metric in metrics_list:
            row += f" {format_cell(stats, metric):>18}"
        print(row)
    
    # ── LaTeX table output ───────────────────────────────────────────
    print()
    print("=" * 65)
    print("LATEX TABLE (paste into paper)")
    print("=" * 65)
    
    col_spec = "l" + "c" * len(metrics_list)
    
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\caption{Performance Comparison: All Models vs Baselines}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\centering")
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\hline")
    
    header_cols = ["\\textbf{Model}"] + \
                  [f"\\textbf{{{l}}}" for l in metrics_labels]
    latex.append(" & ".join(header_cols) + r" \\")
    latex.append(r"\hline")
    
    # Baseline placeholder rows
    latex.append(r"\multicolumn{" + str(len(metrics_list)+1) + 
                 r"}{l}{\textit{Rule-based baselines}} \\")
    latex.append(r"\hline")
    
    for b in ["Fixed-Time", "Webster", "MaxPressure"]:
        placeholders = " & ".join(["\\textit{see baseline\\_results.json}"] * 
                                   len(metrics_list))
        latex.append(f"{b} & {placeholders} \\\\")
    
    latex.append(r"\hline")
    latex.append(r"\multicolumn{" + str(len(metrics_list)+1) + 
                 r"}{l}{\textit{Learning-based models (mean $\pm$ std, 3 seeds)}} \\")
    latex.append(r"\hline")
    
    for model_name in MODEL_ORDER:
        if model_name not in all_results:
            continue
        stats = all_results[model_name]
        cells = [format_cell(stats, m) for m in metrics_list]
        
        # Bold the best model (Fed-ST-GAT)
        if model_name == "Fed-ST-GAT":
            row_latex = f"\\textbf{{{model_name}}} & " + \
                        " & ".join([f"\\textbf{{{c}}}" for c in cells]) + \
                        r" \\"
        else:
            row_latex = f"{model_name} & " + \
                        " & ".join(cells) + r" \\"
        latex.append(row_latex)
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    latex_str = "\n".join(latex)
    print(latex_str)
    
    # ── Save outputs ─────────────────────────────────────────────────
    with open("paper_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # CSV summary
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in all_results:
            continue
        stats = all_results[model_name]
        row = {"model": model_name, "n_seeds": stats["n_seeds"]}
        for metric in metrics_list:
            if stats.get(metric):
                row[f"{metric}_mean"] = round(stats[metric]["mean"], 3)
                row[f"{metric}_std"]  = round(stats[metric]["std"],  3)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"]  = None
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("paper_metrics.csv", index=False)
    
    print()
    print("Saved:")
    print("  paper_metrics.json  — full results with per-seed values")
    print("  paper_metrics.csv   — summary table")
    print()
    print("Next step: run run_baselines.py then add those numbers")
    print("to the LaTeX table above.")


if __name__ == "__main__":
    main()