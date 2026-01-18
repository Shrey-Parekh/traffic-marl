from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any


def run_scenarios(total_episodes: int, seeds: List[int], Ns: List[int]) -> None:
    """Run multiple short trainings across seeds and network sizes and write a report."""
    os.makedirs("outputs", exist_ok=True)

    scenarios: List[Dict[str, Any]] = []
    ep_left = total_episodes
    i = 0
    while ep_left > 0:
        seed = seeds[i % len(seeds)]
        N = Ns[i % len(Ns)]
        episodes = min(10, ep_left)  # chunked runs to keep each quick
        i += 1

        os.system(
            f"python -m src.train --episodes {episodes} --N {N} --seed {seed}"
        )

        live_path = os.path.join("outputs", "live_metrics.json")
        if os.path.exists(live_path):
            with open(live_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
                rec.update({"seed": seed, "N": N, "episodes": episodes})
                scenarios.append(rec)

        ep_left -= episodes

    report_path = os.path.join("outputs", "scenarios_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"runs": scenarios}, f, indent=2)
    print(f"Scenarios report saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run many diverse scenarios")
    parser.add_argument("--total_episodes", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--Ns", type=str, default="2,4,6")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    Ns = [int(s) for s in args.Ns.split(",") if s]
    run_scenarios(args.total_episodes, seeds, Ns)


if __name__ == "__main__":
    main()


