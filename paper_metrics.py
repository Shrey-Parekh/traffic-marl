"""Extract paper metrics from outputs/ and 'output exports/' CSVs.

Usage:
    python paper_metrics.py            # last 50 episodes (exploitation phase)
    python paper_metrics.py --all      # all episodes
"""
from collections import defaultdict
import json
import glob
import sys
import numpy as np
import pandas as pd

OUTPUTS_DIRS = ["outputs", "output exports"]
MODEL_ORDER = ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT"]
METRICS = ["avg_queue", "throughput"]


def find_csv_files() -> dict:
    """Scan all output dirs for *_metrics.csv.
    Returns {model: {seed: {scenario: path}}}
    """
    found = defaultdict(lambda: defaultdict(dict))
    for directory in OUTPUTS_DIRS:
        for path in glob.glob(f"{directory}/*_metrics.csv"):
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
            seed = int(seed_str)
            # Later directories override earlier ones for the same key
            found[model][seed][scenario] = path
    return {m: dict(seeds) for m, seeds in found.items()}


def seed_mean(path: str, last_n: int | None) -> dict[str, float]:
    """Return per-metric mean for a single seed CSV."""
    try:
        df = pd.read_csv(path)
    except OSError as e:
        print(f"  ERROR reading {path}: {e}")
        return {}
    if df.empty:
        return {}
    if last_n is not None:
        df = df.tail(last_n)
    result = {}
    for metric in METRICS:
        if metric not in df.columns:
            continue
        vals = df[metric].dropna().values
        if len(vals):
            result[metric] = float(np.mean(vals))
    return result


def compute_stats(seed_dict: dict, last_n: int | None) -> dict:
    """For each metric, collect per-seed means then compute mean ± SD across seeds."""
    # seed_dict: {seed: {scenario: path}}
    # We aggregate all scenarios within a seed into one mean first,
    # then aggregate across seeds.
    per_seed: dict[int, dict[str, list[float]]] = {}
    for seed, scenario_paths in seed_dict.items():
        seed_vals: dict[str, list[float]] = defaultdict(list)
        for path in scenario_paths.values():
            m = seed_mean(path, last_n)
            for metric, val in m.items():
                seed_vals[metric].append(val)
        if seed_vals:
            per_seed[seed] = {metric: float(np.mean(vals)) for metric, vals in seed_vals.items()}

    if not per_seed:
        return {}

    result = {}
    for metric in METRICS:
        vals = [per_seed[s][metric] for s in per_seed if metric in per_seed[s]]
        if not vals:
            continue
        result[metric] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
            "seed_means": {str(s): round(per_seed[s][metric], 4) for s in per_seed if metric in per_seed[s]},
        }
    return result


def format_cell(stats: dict, metric: str) -> str:
    if metric not in stats:
        return "N/A"
    m, sd = stats[metric]["mean"], stats[metric]["sd"]
    if metric == "throughput":
        return f"{m:.0f} ± {sd:.0f}"
    return f"{m:.2f} ± {sd:.2f}"


def main(last_n: int | None) -> None:
    label = f"last {last_n} episodes" if last_n is not None else "all episodes"
    print("=" * 70)
    print(f"PAPER METRICS — {label}")
    print(f"Sources: {OUTPUTS_DIRS}")
    print("=" * 70)

    csv_map = find_csv_files()
    if not csv_map:
        print("No *_metrics.csv files found.")
        return

    print(f"Models found: {sorted(csv_map.keys())}\n")

    all_results = {}

    for model in MODEL_ORDER:
        if model not in csv_map:
            print(f"  {model}: NOT FOUND — skipping")
            continue

        seed_dict = csv_map[model]
        print(f"{model}: {len(seed_dict)} seed(s) — {sorted(seed_dict.keys())}")
        for seed, scenarios in sorted(seed_dict.items()):
            for scenario, path in scenarios.items():
                print(f"  seed {seed} [{scenario}]: {path}")

        stats = compute_stats(seed_dict, last_n)
        if not stats:
            print("  No valid data\n")
            continue

        all_results[model] = stats
        for metric in METRICS:
            if metric in stats:
                s = stats[metric]
                print(f"  {metric}: {format_cell(stats, metric)}  "
                      f"(n_seeds={s['n_seeds']}, seed_means={s['seed_means']})")
        print()

    print("=" * 70)
    print("RESULTS TABLE  (mean ± SD across seeds)")
    print("=" * 70)
    print(f"{'Model':<16} {'Avg Queue (PCU)':>22} {'Throughput (veh)':>22}")
    print("-" * 62)
    for model in MODEL_ORDER:
        if model not in all_results:
            continue
        s = all_results[model]
        print(f"{model:<16} {format_cell(s, 'avg_queue'):>22} {format_cell(s, 'throughput'):>22}")

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
                row[f"{metric}_mean"] = round(s[metric]["mean"], 4)
                row[f"{metric}_sd"] = round(s[metric]["sd"], 4)
                row[f"{metric}_n_seeds"] = s[metric]["n_seeds"]
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {json_path}, {csv_path}")


if __name__ == "__main__":
    use_all = "--all" in sys.argv
    main(last_n=None if use_all else 50)
