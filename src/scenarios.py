from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .config import SCENARIOS_REPORT_JSON, LIVE_METRICS_JSON, OUTPUTS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_scenarios(total_episodes: int, seeds: List[int], Ns: List[int]) -> None:
    """Run multiple short trainings across seeds and network sizes and write a report."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios: List[Dict[str, Any]] = []
    ep_left = total_episodes
    i = 0
    while ep_left > 0:
        seed = seeds[i % len(seeds)]
        N = Ns[i % len(Ns)]
        episodes = min(10, ep_left)  # chunked runs to keep each quick
        i += 1

        import os
        os.system(
            f"python -m src.train --episodes {episodes} --N {N} --seed {seed}"
        )

        if LIVE_METRICS_JSON.exists():
            with open(LIVE_METRICS_JSON, "r", encoding="utf-8") as f:
                rec = json.load(f)
                rec.update({"seed": seed, "N": N, "episodes": episodes})
                scenarios.append(rec)

        ep_left -= episodes

    with open(SCENARIOS_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"runs": scenarios}, f, indent=2)
    logger.info(f"Scenarios report saved to {SCENARIOS_REPORT_JSON}")


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
