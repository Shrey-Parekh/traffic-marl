from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np

from .env import MiniTrafficEnv, EnvConfig


def run_comprehensive_baseline(
    episodes: int,
    num_intersections: int,
    max_steps: int,
    switch_periods: List[int],
    seeds: List[int],
    save_dir: str = "outputs"
) -> Dict[str, Any]:
    """Generate baseline data across several switch periods and seeds."""
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    strategy_summaries = {}
    
    print(f"Generating comprehensive baseline data...")
    print(f"Testing {len(switch_periods)} strategies × {len(seeds)} seeds = {len(switch_periods) * len(seeds)} total runs")
    
    for period in switch_periods:
        period_results = []
        
        for seed in seeds:
            env = MiniTrafficEnv(
                EnvConfig(
                    num_intersections=num_intersections,
                    max_steps=max_steps,
                    seed=seed
                )
            )
            
            episode_metrics = []
            for ep in range(episodes):
                obs = env.reset()
                done = False
                step_count = 0
                episode_reward_sum = 0.0
                
                while not done:
                    # Fixed-time action: switch every 'period' steps
                    actions = {}
                    for i in range(num_intersections):
                        # Switch if it's time and minimum green has passed
                        if step_count % period == 0 and step_count > 0:
                            actions[f"int{i}"] = 1  # switch
                        else:
                            actions[f"int{i}"] = 0  # keep
                    
                    obs, rewards, done, info = env.step(actions)
                    episode_reward_sum += float(np.mean(list(rewards.values())))
                    step_count += 1
                
                episode_data = {
                    "episode": ep,
                    "seed": seed,
                    "switch_period": period,
                    "avg_reward": episode_reward_sum,
                    "avg_queue": info.get("avg_queue", 0.0),
                    "throughput": info.get("throughput", 0.0),
                    "avg_travel_time": info.get("avg_travel_time", 0.0),
                }
                episode_metrics.append(episode_data)
                all_results.append(episode_data)
            
            period_results.extend(episode_metrics)
            print(f"  Period {period}, Seed {seed}: Avg Queue = {np.mean([r['avg_queue'] for r in episode_metrics]):.2f}")
        
        period_queues = [r['avg_queue'] for r in period_results]
        period_throughput = [r['throughput'] for r in period_results]
        period_travel_time = [r['avg_travel_time'] for r in period_results]
        
        strategy_summaries[f"period_{period}"] = {
            "switch_period": period,
            "avg_queue_mean": float(np.mean(period_queues)),
            "avg_queue_std": float(np.std(period_queues)),
            "throughput_mean": float(np.mean(period_throughput)),
            "throughput_std": float(np.std(period_throughput)),
            "avg_travel_time_mean": float(np.mean(period_travel_time)),
            "avg_travel_time_std": float(np.std(period_travel_time)),
            "episodes": len(period_results)
        }
    
    all_queues = [r['avg_queue'] for r in all_results]
    all_throughput = [r['throughput'] for r in all_results]
    all_travel_time = [r['avg_travel_time'] for r in all_results]
    
    overall_summary = {
        "total_runs": len(all_results),
        "strategies_tested": len(switch_periods),
        "seeds_per_strategy": len(seeds),
        "episodes_per_run": episodes,
        "overall_avg_queue": float(np.mean(all_queues)),
        "overall_avg_queue_std": float(np.std(all_queues)),
        "overall_throughput": float(np.mean(all_throughput)),
        "overall_throughput_std": float(np.std(all_throughput)),
        "overall_avg_travel_time": float(np.mean(all_travel_time)),
        "overall_avg_travel_time_std": float(np.std(all_travel_time)),
        "best_strategy": min(strategy_summaries.keys(), 
                           key=lambda k: strategy_summaries[k]['avg_queue_mean']),
        "worst_strategy": max(strategy_summaries.keys(), 
                            key=lambda k: strategy_summaries[k]['avg_queue_mean'])
    }
    
    best_run = min(all_results, key=lambda r: r['avg_queue'])
    
    report = {
        "metadata": {
            "description": "Comprehensive baseline data for traffic light control",
            "methodology": "Fixed-time control with multiple switch periods and seeds",
            "generated_at": str(np.datetime64('now')),
            "parameters": {
                "num_intersections": num_intersections,
                "max_steps": max_steps,
                "switch_periods": switch_periods,
                "seeds": seeds,
                "episodes_per_run": episodes
            }
        },
        "overall_summary": overall_summary,
        "strategy_summaries": strategy_summaries,
        "best_run": best_run,
        "all_results": all_results
    }
    
    detailed_path = os.path.join(save_dir, "baseline_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    simple_results = []
    for result in all_results:
        simple_results.append({
            "episode": result["episode"],
            "avg_reward": result["avg_reward"],
            "avg_queue": result["avg_queue"],
            "throughput": result["throughput"],
            "avg_travel_time": result["avg_travel_time"]
        })
    
    simple_path = os.path.join(save_dir, "baseline_metrics.json")
    with open(simple_path, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, indent=2)
    
    summary_path = os.path.join(save_dir, "baseline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("COMPREHENSIVE BASELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total runs: {overall_summary['total_runs']}\n")
        f.write(f"Strategies tested: {overall_summary['strategies_tested']}\n")
        f.write(f"Seeds per strategy: {overall_summary['seeds_per_strategy']}\n")
        f.write(f"Episodes per run: {overall_summary['episodes_per_run']}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Average Queue: {overall_summary['overall_avg_queue']:.2f} ± {overall_summary['overall_avg_queue_std']:.2f}\n")
        f.write(f"  Throughput: {overall_summary['overall_throughput']:.1f} ± {overall_summary['overall_throughput_std']:.1f}\n")
        f.write(f"  Travel Time: {overall_summary['overall_avg_travel_time']:.2f}s ± {overall_summary['overall_avg_travel_time_std']:.2f}s\n\n")
        
        f.write("BEST STRATEGY:\n")
        best_key = overall_summary['best_strategy']
        best_data = strategy_summaries[best_key]
        f.write(f"  Switch Period: {best_data['switch_period']} steps\n")
        f.write(f"  Average Queue: {best_data['avg_queue_mean']:.2f} ± {best_data['avg_queue_std']:.2f}\n")
        f.write(f"  Throughput: {best_data['throughput_mean']:.1f} ± {best_data['throughput_std']:.1f}\n")
        f.write(f"  Travel Time: {best_data['avg_travel_time_mean']:.2f}s ± {best_data['avg_travel_time_std']:.2f}s\n\n")
        
        f.write("STRATEGY COMPARISON:\n")
        for key, data in strategy_summaries.items():
            f.write(f"  Period {data['switch_period']:2d}: Queue={data['avg_queue_mean']:5.2f}±{data['avg_queue_std']:4.2f}, "
                   f"Throughput={data['throughput_mean']:6.1f}±{data['throughput_std']:4.1f}\n")
    
    print(f"\nBaseline generation complete!")
    print(f"  Detailed results: {detailed_path}")
    print(f"  Dashboard data: {simple_path}")
    print(f"  Summary: {summary_path}")
    print(f"\nBest strategy: Switch every {strategy_summaries[overall_summary['best_strategy']]['switch_period']} steps")
    print(f"Overall performance: Queue={overall_summary['overall_avg_queue']:.2f}, "
          f"Throughput={overall_summary['overall_throughput']:.1f}, "
          f"Travel Time={overall_summary['overall_avg_travel_time']:.2f}s")
    
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive baseline data")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per run")
    parser.add_argument("--N", type=int, default=6, help="Number of intersections")
    parser.add_argument("--max_steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--switch_periods", type=str, default="10,15,20,25,30", 
                       help="Comma-separated switch periods to test")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5", 
                       help="Comma-separated seeds to test")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    switch_periods = [int(x.strip()) for x in args.switch_periods.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]
    
    run_comprehensive_baseline(
        episodes=args.episodes,
        num_intersections=args.N,
        max_steps=args.max_steps,
        switch_periods=switch_periods,
        seeds=seeds,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
