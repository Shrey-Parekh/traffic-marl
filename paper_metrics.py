"""Extract paper metrics from outputs/ CSVs.

Usage:
    python paper_metrics.py            # last 37 episodes (exploitation phase)
    python paper_metrics.py --all      # all 250 episodes
"""
from collections import defaultdict
import json
import glob
import sys
import numpy as np
import pandas as pd

OUTPUTS_DIR = "outputs"
MODEL_ORDER = ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT"]
METRICS = ["avg_queue", "throughput"]


def find_csv_files() -> dict:
    """Scan outputs/ for *_metrics.csv. Returns {model: {seed: {scenario: path}}}."""
    found = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(f"{OUTPUTS_DIR}/*_metrics.csv"):
        fname = path.replace("\\", "/").split("/")[-1]
        stem = fname[: -len("_metrics.csv")]
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            print(f"  SKIP (unexpected name): {fname}")
            continue
        model, seed_str, scenario = parts
        if not seed_str.isdigit():
            print(f"  SKIP (non-numeric seed): {fname}")
            continue
        found[model][int(seed_str)][scenario] = path
    return {m: dict(seeds) for m, seeds in found.items()}


def compute_stats(paths: list, last_n: int | None) -> dict:
    """Concatenate last_n rows from each CSV (None = all rows), return mean per metric."""
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except OSError as e:
            print(f"  ERROR reading {path}: {e}")
            continue
        if len(df) == 0:
            continue
        frames.append(df if last_n is None else df.tail(last_n))

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    result = {}
    for metric in METRICS:
        if metric not in combined.columns:
            print(f"  WARNING: column '{metric}' not found")
            continue
        vals = combined[metric].dropna().values
        result[metric] = {"mean": float(np.mean(vals)), "n_rows": len(vals)}
    return result


def format_cell(stats: dict, metric: str) -> str:
    """Format metric value for display."""
    if metric not in stats:
        return "N/A"
    m = stats[metric]["mean"]
    return f"{m:.0f}" if metric == "throughput" else f"{m:.2f}"


def main(last_n: int | None) -> None:
    """Run metric extraction and print results table."""
    label = f"last {last_n} episodes" if last_n is not None else "all episodes"
    print("=" * 60)
    print(f"PAPER METRICS — {label}")
    print(f"Source: {OUTPUTS_DIR}/")
    print("=" * 60)

    csv_map = find_csv_files()
    if not csv_map:
        print("No *_metrics.csv files found in outputs/")
        return

    print(f"Models found: {sorted(csv_map.keys())}\n")

    all_results = {}

    for model in MODEL_ORDER:
        if model not in csv_map:
            print(f"  {model}: NOT FOUND — skipping")
            continue

        paths = [
            path
            for seed_dict in csv_map[model].values()
            for path in seed_dict.values()
        ]
        print(f"{model}: {len(paths)} CSV(s)")
        for p in paths:
            print(f"  {p}")

        stats = compute_stats(paths, last_n)
        if not stats:
            print("  No valid data\n")
            continue

        all_results[model] = stats
        for metric in METRICS:
            if metric in stats:
                print(f"  {metric}: {format_cell(stats, metric)}  (n={stats[metric]['n_rows']})")
        print()

    print("=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'Model':<16} {'Avg Queue (PCU)':>18} {'Throughput (veh)':>18}")
    print("-" * 54)
    for model in MODEL_ORDER:
        if model not in all_results:
            continue
        s = all_results[model]
        print(f"{model:<16} {format_cell(s, 'avg_queue'):>18} {format_cell(s, 'throughput'):>18}")

    suffix = "" if last_n is None else f"_last{last_n}"
    json_path = f"paper_metrics{suffix}.json"
    csv_path = f"paper_metrics{suffix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for model in MODEL_ORDER:
        if model not in all_results:
            continue
        s = all_results[model]
        row = {"model": model}
        for metric in METRICS:
            if metric in s:
                row[f"{metric}_mean"] = round(s[metric]["mean"], 3)
                row[f"{metric}_n"] = s[metric]["n_rows"]
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {json_path}, {csv_path}")


if __name__ == "__main__":
    use_all = "--all" in sys.argv
    main(last_n=None if use_all else 50)
