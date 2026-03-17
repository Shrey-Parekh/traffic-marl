"""
Baseline evaluation script for paper Table IV.

Runs Fixed-Time and Webster across multiple seeds on morning_peak scenario.

Configuration matches trained models exactly:
- Scenario: morning_peak
- Max steps: 300
- N agents: 9
- Seeds: 1, 2, 3

Usage:
    python run_baselines.py

Output:
    baseline_results.json
"""

import json
import numpy as np
from src.env_sumo import PuneSUMOEnv
from src.baseline import FixedTimeController, WebsterController, MaxPressureController
from src.config import INJECTION_CONFIG, PEAK_HOUR_CONFIG, BASELINE_CONFIG, SUMO_CONFIG

SCENARIO = "morning_peak"
MAX_STEPS = 300
N_AGENTS = 9
SEEDS = [1, 2, 3]
N_EVAL_EPISODES = 5


def evaluate_baseline_seed(controller, seed: int) -> dict:
    """Run controller for N_EVAL_EPISODES with given seed and return metrics."""
    env = PuneSUMOEnv({
        "render": False,
        "scenario": SCENARIO,
        "n_intersections": N_AGENTS,
        "max_steps": MAX_STEPS,
        "use_global_reward": True,
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
    ns_flow_vph = INJECTION_CONFIG["base_rate"] * PEAK_HOUR_CONFIG[SCENARIO]["NS_multiplier"] * 3600
    ew_flow_vph = INJECTION_CONFIG["base_rate"] * PEAK_HOUR_CONFIG[SCENARIO]["EW_multiplier"] * 3600

    webster = WebsterController(
        n_agents=N_AGENTS,
        saturation_flow=BASELINE_CONFIG["webster_saturation_flow"],
        lost_time_per_phase=BASELINE_CONFIG["webster_lost_time"],
    )
    webster.compute_timing(ns_flow_vph, ew_flow_vph)
    print(f"Webster timing: NS={webster.ns_green}s EW={webster.ew_green}s cycle={webster.cycle_length}s\n")

    baselines = {
        "Fixed-Time": FixedTimeController(
            n_agents=N_AGENTS,
            ns_green_duration=BASELINE_CONFIG["fixed_time_cycle"],
            ew_green_duration=BASELINE_CONFIG["fixed_time_cycle"],
            clearance_duration=SUMO_CONFIG["clearance_steps"],
        ),
        "Webster": webster,
        "MaxPressure": MaxPressureController(
            n_agents=N_AGENTS,
            min_green_steps=SUMO_CONFIG["min_green_steps"],
            pressure_threshold=3.0,
        ),
    }

    results = {}
    for name, controller in baselines.items():
        print(f"Evaluating {name}...")
        all_values = {"avg_queue_pcu": [], "avg_travel_time": [], "throughput": []}

        for seed in SEEDS:
            print(f"  Seed {seed}:")
            seed_metrics = evaluate_baseline_seed(controller, seed)
            for metric in all_values:
                all_values[metric].extend(seed_metrics[metric])

        results[name] = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n_episodes": len(values),
            }
            for metric, values in all_values.items()
        }

        print(f"  {name} overall (n={len(all_values['avg_queue_pcu'])} episodes):")
        for metric, stats in results[name].items():
            print(f"    {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print()

    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 55)
    print("BASELINE RESULTS FOR TABLE IV")
    print("=" * 55)
    print(f"{'Baseline':<15} {'Queue':>10} {'Travel':>10} {'Throughput':>12}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics['avg_queue_pcu']['mean']:>10.2f} "
              f"{metrics['avg_travel_time']['mean']:>10.1f} "
              f"{metrics['throughput']['mean']:>12.0f}")

    print(f"\n{N_EVAL_EPISODES} episodes × {len(SEEDS)} seeds = "
          f"{N_EVAL_EPISODES * len(SEEDS)} total episodes per baseline")
    print("Saved to baseline_results.json")


if __name__ == "__main__":
    main()
