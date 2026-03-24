"""
Baseline evaluation script for paper Table IV.

Runs Fixed-Time, Webster, and MaxPressure across multiple seeds on all 3 scenarios.

Configuration matches trained models exactly:
- Scenarios: uniform, morning_peak, evening_peak
- Max steps: 300
- N agents: 9
- Seeds: 1, 2, 3, 4, 5

Usage:
    python run_baselines.py

Output:
    outputs/baseline_results.json
"""

import json
from pathlib import Path
import numpy as np
from src.env_sumo import PuneSUMOEnv
from src.baseline import FixedTimeController, MaxPressureController, WebsterController
from src.config import INJECTION_CONFIG, PEAK_HOUR_CONFIG, BASELINE_CONFIG

SCENARIOS = ["uniform", "morning_peak", "evening_peak"]
MAX_STEPS = 300
N_AGENTS = 9
SEEDS = [1, 2, 3]
N_EVAL_EPISODES = 5
N_ROUTES_PER_DIRECTION = 6


def evaluate_baseline_seed(controller, seed: int, scenario: str = "uniform") -> dict:
    """Run controller for N_EVAL_EPISODES with given seed and return metrics."""
    env = PuneSUMOEnv({
        "render": False,
        "scenario": scenario,
        "n_intersections": N_AGENTS,
        "max_steps": MAX_STEPS,
        "seed": seed,
    })

    episode_metrics = {"avg_queue_pcu": [], "avg_travel_time": [], "throughput": []}

    for ep in range(N_EVAL_EPISODES):
        obs = env.reset()
        controller.reset()
        done = False
        step_queue_pcus = []
        while not done:
            obs, _, done, info = env.step(controller.act(obs))
            step_queue_pcus.append(info.get("avg_queue_pcu", 0))

        # Use episode-mean queue, not the final-step snapshot
        episode_metrics["avg_queue_pcu"].append(float(np.mean(step_queue_pcus)))
        episode_metrics["avg_travel_time"].append(info.get("avg_travel_time", 0))
        episode_metrics["throughput"].append(info.get("throughput", 0))

        print(f"    Seed {seed} Ep {ep+1}: "
              f"queue={info.get('avg_queue_pcu', 0):.2f} "
              f"travel={info.get('avg_travel_time', 0):.1f}s "
              f"throughput={info.get('throughput', 0)}")

    env.close()
    return episode_metrics


def main():
    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n{'='*55}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*55}")

        ns_flow_vph = INJECTION_CONFIG["base_rate"] * N_ROUTES_PER_DIRECTION * PEAK_HOUR_CONFIG[scenario]["NS_multiplier"] * 3600
        ew_flow_vph = INJECTION_CONFIG["base_rate"] * N_ROUTES_PER_DIRECTION * PEAK_HOUR_CONFIG[scenario]["EW_multiplier"] * 3600

        webster = WebsterController(n_agents=N_AGENTS)
        webster.compute_timing(ns_flow_vph, ew_flow_vph)
        print(f"Webster timing: NS={webster.ns_green}s EW={webster.ew_green}s cycle={webster.cycle_length}s")

        controllers = {
            "Fixed-Time": FixedTimeController(n_agents=N_AGENTS),
            "Webster": webster,
            "MaxPressure": MaxPressureController(n_agents=N_AGENTS),
        }

        scenario_results = {}

        for name, controller in controllers.items():
            print(f"\n  Evaluating {name}...")
            all_values = {"avg_queue_pcu": [], "avg_travel_time": [], "throughput": []}

            for seed in SEEDS:
                seed_metrics = evaluate_baseline_seed(controller, seed, scenario)
                for metric in all_values:
                    all_values[metric].extend(seed_metrics[metric])

            scenario_results[name] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_episodes": len(values),
                }
                for metric, values in all_values.items()
            }

            print(f"    {name}: queue={scenario_results[name]['avg_queue_pcu']['mean']:.2f} ± {scenario_results[name]['avg_queue_pcu']['std']:.2f}")

        all_results[scenario] = scenario_results

    output_path = Path("outputs") / "baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    print(f"\n{'='*70}")
    print("BASELINE RESULTS FOR TABLE 1")
    print(f"{'='*70}")
    for scenario in SCENARIOS:
        print(f"\n  {scenario}:")
        print(f"  {'Baseline':<15} {'Queue':>10} {'Travel':>10} {'Throughput':>12}")
        print(f"  {'-'*50}")
        for name, metrics in all_results[scenario].items():
            print(f"  {name:<15} "
                  f"{metrics['avg_queue_pcu']['mean']:>10.2f} "
                  f"{metrics['avg_travel_time']['mean']:>10.1f} "
                  f"{metrics['throughput']['mean']:>12.0f}")


if __name__ == "__main__":
    main()
