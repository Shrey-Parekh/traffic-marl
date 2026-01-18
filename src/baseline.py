from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .config import BaselineConfig, BASELINE_METRICS_JSON, OUTPUTS_DIR
from .env import MiniTrafficEnv, EnvConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_fixed_time_episode(env: MiniTrafficEnv, switch_period: int) -> Dict[str, float]:
    obs = env.reset()
    done = False
    t = 0
    sum_reward = 0.0
    while not done:
        actions: Dict[str, int] = {}
        # Switch every switch_period, but env enforces min_green
        do_switch = 1 if (t % switch_period == 0 and t > 0) else 0
        for aid in obs.keys():
            actions[aid] = do_switch

        obs, rewards, done, info = env.step(actions)
        sum_reward += float(np.mean(list(rewards.values())))
        t += 1

    return {
        "avg_reward": sum_reward,
        "avg_queue": info.get("avg_queue", 0.0),
        "throughput": info.get("throughput", 0.0),
        "avg_travel_time": info.get("avg_travel_time", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-time baseline")
    parser.add_argument("--episodes", type=int, default=BaselineConfig.episodes)
    parser.add_argument("--N", type=int, default=BaselineConfig.num_intersections)
    parser.add_argument("--max_steps", type=int, default=BaselineConfig.max_steps)
    parser.add_argument("--switch_period", type=int, default=BaselineConfig.switch_period)
    parser.add_argument("--save_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--seed", type=int, default=BaselineConfig.seed)
    parser.add_argument("--neighbor_obs", action="store_true")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows (optional)")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid cols (optional)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = MiniTrafficEnv(
        EnvConfig(
            num_intersections=args.N, 
            max_steps=args.max_steps, 
            seed=args.seed,
            neighbor_obs=args.neighbor_obs,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
    )

    metrics = []
    for ep in range(args.episodes):
        m = run_fixed_time_episode(env, switch_period=args.switch_period)
        metrics.append({"episode": ep, **m})
        logger.info(
            f"Baseline Ep {ep}: avg_reward={m['avg_reward']:.2f} avg_queue={m['avg_queue']:.2f} "
            f"avg_travel_time={m['avg_travel_time']:.2f}s throughput={m['throughput']:.0f}"
        )

    out_path = save_dir / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Baseline metrics saved to {out_path}")


if __name__ == "__main__":
    main()
