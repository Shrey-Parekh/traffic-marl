"""Multi-model comparison training for traffic control RL."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
import threading
import time

# Ensure project root is in Python path for imports when running as script
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch

# Handle both relative imports (when run as module) and absolute imports (when run as script)
try:
    from .config import TrainingConfig, OUTPUTS_DIR, ModelType
    from .env import MiniTrafficEnv, EnvConfig
    from .train import main as train_single_model
except ImportError:
    # Fallback for when running as script
    from src.config import TrainingConfig, OUTPUTS_DIR, ModelType
    from src.env import MiniTrafficEnv, EnvConfig
    from src.train import main as train_single_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_baseline(config: TrainingConfig, save_dir: Path) -> Dict[str, Any]:
    """Run baseline fixed-time controller."""
    logger.info("Running baseline fixed-time controller...")
    
    baseline_dir = save_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    env = MiniTrafficEnv(EnvConfig(
        num_intersections=config.num_intersections,
        max_steps=config.max_steps,
        seed=config.seed,
    ))
    
    all_results = []
    
    # Run multiple episodes for statistical significance
    for episode in range(config.episodes):
        obs = env.reset(seed=config.seed + episode)
        done = False
        step = 0
        episode_reward = 0.0
        
        while not done:
            # Fixed-time switching every 20 steps
            switch_period = 20
            do_switch = 1 if (step % switch_period == 0 and step > 0) else 0
            actions = {aid: do_switch for aid in obs.keys()}
            
            obs, rewards, done, info = env.step(actions)
            episode_reward += float(np.mean(list(rewards.values())))
            step += 1
        
        result = {
            "episode": episode,
            "model_type": "Baseline",
            "avg_reward": episode_reward,
            "avg_queue": info.get("avg_queue", 0.0),
            "throughput": info.get("throughput", 0.0),
            "avg_travel_time": info.get("avg_travel_time", 0.0),
            "updates": 0,
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }
        all_results.append(result)
        
        logger.info(f"Baseline Episode {episode+1}/{config.episodes}: "
                   f"Queue={result['avg_queue']:.2f}, "
                   f"Throughput={result['throughput']:.0f}, "
                   f"TravelTime={result['avg_travel_time']:.2f}s")
    
    # Save baseline results
    metrics_path = baseline_dir / "metrics.json"
    csv_path = baseline_dir / "metrics.csv"
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    # Write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    # Calculate averages
    avg_metrics = {
        "avg_reward": float(np.mean([r["avg_reward"] for r in all_results])),
        "avg_queue": float(np.mean([r["avg_queue"] for r in all_results])),
        "throughput": float(np.mean([r["throughput"] for r in all_results])),
        "avg_travel_time": float(np.mean([r["avg_travel_time"] for r in all_results])),
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
    }
    
    logger.info(f"Baseline completed: Queue={avg_metrics['avg_queue']:.2f}, "
               f"Throughput={avg_metrics['throughput']:.0f}, "
               f"TravelTime={avg_metrics['avg_travel_time']:.2f}s")
    
    return {
        "model_type": "Baseline",
        "episodes": len(all_results),
        "average_metrics": avg_metrics,
        "final_episode": all_results[-1] if all_results else {},
        "all_results": all_results,
    }


def run_single_model_training(model_type: str, config: TrainingConfig, save_dir: Path) -> Dict[str, Any]:
    """Run training for a single model type."""
    logger.info(f"Starting {model_type} training...")
    
    model_dir = save_dir / model_type.lower().replace("-", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for single model training
    args = [
        "--episodes", str(config.episodes),
        "--N", str(config.num_intersections),
        "--max_steps", str(config.max_steps),
        "--seed", str(config.seed),
        "--lr", str(config.learning_rate),
        "--batch_size", str(config.batch_size),
        "--gamma", str(config.gamma),
        "--model_type", model_type,
        "--save_dir", str(model_dir),
    ]
    
    # Add model-specific parameters
    if model_type == "PPO-GNN":
        args.extend(["--ppo_epochs", str(config.ppo_epochs)])
        args.extend(["--ppo_clip_ratio", str(config.ppo_clip_ratio)])
        args.extend(["--ppo_value_coef", str(config.ppo_value_coef)])
        args.extend(["--ppo_entropy_coef", str(config.ppo_entropy_coef)])
    elif model_type == "GNN-A2C":
        args.extend(["--a2c_value_coef", str(config.a2c_value_coef)])
        args.extend(["--a2c_entropy_coef", str(config.a2c_entropy_coef)])
    elif model_type == "GAT-DQN":
        args.extend(["--gat_n_heads", str(config.gat_n_heads)])
        args.extend(["--gat_dropout", str(config.gat_dropout)])
    
    # Add meta-learning parameters if enabled
    if config.use_meta_learning:
        args.append("--use_meta_learning")
        args.extend(["--meta_epsilon_min", str(config.meta_epsilon_min)])
        args.extend(["--meta_epsilon_max", str(config.meta_epsilon_max)])
        args.extend(["--meta_lr_scale_min", str(config.meta_lr_scale_min)])
        args.extend(["--meta_lr_scale_max", str(config.meta_lr_scale_max)])
        args.extend(["--meta_update_frequency", str(config.meta_update_frequency)])
    
    # Add DQN-specific parameters
    if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
        args.extend(["--epsilon_start", str(config.epsilon_start)])
        args.extend(["--epsilon_end", str(config.epsilon_end)])
        args.extend(["--epsilon_decay_steps", str(config.epsilon_decay_steps)])
        args.extend(["--update_target_steps", str(config.update_target_steps)])
    
    args.extend(["--min_buffer_size", str(config.min_buffer_size)])
    
    # Run training by calling the main function with modified sys.argv
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["train.py"] + args
        train_single_model()
    except SystemExit:
        pass  # Ignore sys.exit() calls from argparse
    finally:
        sys.argv = original_argv
    
    # Load results
    metrics_path = model_dir / "metrics.json"
    final_report_path = model_dir / "final_report.json"
    
    results = {"model_type": model_type, "episodes": 0, "average_metrics": {}, "final_episode": {}}
    
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            results["all_results"] = metrics
            results["episodes"] = len(metrics)
            if metrics:
                results["final_episode"] = metrics[-1]
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading metrics for {model_type}: {e}")
    
    if final_report_path.exists():
        try:
            with open(final_report_path, "r", encoding="utf-8") as f:
                final_report = json.load(f)
            results["average_metrics"] = final_report.get("average_metrics", {})
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading final report for {model_type}: {e}")
    
    logger.info(f"{model_type} training completed")
    return results


def run_multi_model_comparison(config: TrainingConfig) -> None:
    """Run all models and baseline for comparison."""
    save_dir = config.save_dir / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = ["DQN", "GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C"]
    all_results = {}
    
    logger.info("Starting Multi-Model Comparison")
    logger.info(f"Models to compare: {models_to_run} + Baseline")
    logger.info(f"Episodes per model: {config.episodes}")
    logger.info(f"Meta-learning: {'ENABLED' if config.use_meta_learning else 'DISABLED'}")
    
    # Run baseline first
    try:
        baseline_results = run_baseline(config, save_dir)
        all_results["Baseline"] = baseline_results
        logger.info("✅ Baseline completed")
    except Exception as e:
        logger.error(f"❌ Baseline failed: {e}")
        all_results["Baseline"] = {"error": str(e)}
    
    # Run all models sequentially (could be parallelized but might cause resource issues)
    for model_type in models_to_run:
        try:
            model_results = run_single_model_training(model_type, config, save_dir)
            all_results[model_type] = model_results
            logger.info(f"✅ {model_type} completed")
        except Exception as e:
            logger.error(f"❌ {model_type} failed: {e}")
            all_results[model_type] = {"error": str(e)}
    
    # Create comparison summary
    comparison_summary = {
        "comparison_mode": True,
        "models_compared": ["Baseline"] + models_to_run,
        "episodes_per_model": config.episodes,
        "meta_learning_enabled": config.use_meta_learning,
        "results": all_results,
        "ranking": {},
        "best_model": {},
    }
    
    # Calculate rankings
    metrics_to_rank = ["avg_queue", "throughput", "avg_travel_time"]
    rankings = {}
    
    for metric in metrics_to_rank:
        metric_values = []
        for model_name, results in all_results.items():
            if "error" not in results and "average_metrics" in results:
                avg_metrics = results["average_metrics"]
                if metric in avg_metrics:
                    metric_values.append((model_name, avg_metrics[metric]))
        
        if metric_values:
            # Sort based on metric (lower is better for queue and travel_time, higher for throughput)
            reverse_sort = metric == "throughput"
            sorted_values = sorted(metric_values, key=lambda x: x[1], reverse=reverse_sort)
            rankings[metric] = [{"model": name, "value": value, "rank": i+1} 
                              for i, (name, value) in enumerate(sorted_values)]
    
    comparison_summary["ranking"] = rankings
    
    # Determine best overall model (simple scoring: sum of ranks, lower is better)
    model_scores = {}
    for model_name in all_results.keys():
        if "error" not in all_results[model_name]:
            total_score = 0
            count = 0
            for metric, ranking in rankings.items():
                for entry in ranking:
                    if entry["model"] == model_name:
                        total_score += entry["rank"]
                        count += 1
                        break
            if count > 0:
                model_scores[model_name] = total_score / count
    
    if model_scores:
        best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x])
        comparison_summary["best_model"] = {
            "name": best_model_name,
            "score": model_scores[best_model_name],
            "metrics": all_results[best_model_name].get("average_metrics", {})
        }
    
    # Save comparison results
    comparison_path = save_dir / "comparison_results.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Save to main outputs directory for dashboard access
    main_comparison_path = OUTPUTS_DIR / "comparison_results.json"
    with open(main_comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info("🎉 Multi-Model Comparison completed!")
    logger.info(f"Results saved to: {comparison_path}")
    
    if comparison_summary["best_model"]:
        best = comparison_summary["best_model"]
        logger.info(f"🏆 Best performing model: {best['name']} (score: {best['score']:.2f})")


def main() -> None:
    """Main function for multi-model comparison."""
    parser = argparse.ArgumentParser(description="Multi-Model Comparison for Traffic Control RL")
    
    # Environment parameters
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--N", type=int, default=TrainingConfig.num_intersections)
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--save_dir", type=str, default=str(OUTPUTS_DIR))
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument("--replay_capacity", type=int, default=TrainingConfig.replay_capacity)
    parser.add_argument("--min_buffer_size", type=int, default=TrainingConfig.min_buffer_size)
    
    # DQN-specific parameters
    parser.add_argument("--epsilon_start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon_end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--epsilon_decay_steps", type=int, default=TrainingConfig.epsilon_decay_steps)
    parser.add_argument("--update_target_steps", type=int, default=TrainingConfig.update_target_steps)
    
    # PPO-specific parameters
    parser.add_argument("--ppo_epochs", type=int, default=TrainingConfig.ppo_epochs)
    parser.add_argument("--ppo_clip_ratio", type=float, default=TrainingConfig.ppo_clip_ratio)
    parser.add_argument("--ppo_value_coef", type=float, default=TrainingConfig.ppo_value_coef)
    parser.add_argument("--ppo_entropy_coef", type=float, default=TrainingConfig.ppo_entropy_coef)
    
    # A2C-specific parameters
    parser.add_argument("--a2c_value_coef", type=float, default=TrainingConfig.a2c_value_coef)
    parser.add_argument("--a2c_entropy_coef", type=float, default=TrainingConfig.a2c_entropy_coef)
    
    # GAT-specific parameters
    parser.add_argument("--gat_n_heads", type=int, default=TrainingConfig.gat_n_heads)
    parser.add_argument("--gat_dropout", type=float, default=TrainingConfig.gat_dropout)
    
    # Meta-learning arguments
    parser.add_argument("--use_meta_learning", action="store_true", help="Enable enhanced meta-learning")
    parser.add_argument("--meta_epsilon_min", type=float, default=TrainingConfig.meta_epsilon_min)
    parser.add_argument("--meta_epsilon_max", type=float, default=TrainingConfig.meta_epsilon_max)
    parser.add_argument("--meta_lr_scale_min", type=float, default=TrainingConfig.meta_lr_scale_min)
    parser.add_argument("--meta_lr_scale_max", type=float, default=TrainingConfig.meta_lr_scale_max)
    parser.add_argument("--meta_update_frequency", type=int, default=TrainingConfig.meta_update_frequency)
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        num_intersections=args.N,
        max_steps=args.max_steps,
        episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        min_buffer_size=args.min_buffer_size,
        model_type="Multi-Model Comparison",
        comparison_mode=True,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        update_target_steps=args.update_target_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_ratio=args.ppo_clip_ratio,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        a2c_value_coef=args.a2c_value_coef,
        a2c_entropy_coef=args.a2c_entropy_coef,
        gat_n_heads=args.gat_n_heads,
        gat_dropout=args.gat_dropout,
        use_meta_learning=args.use_meta_learning,
        meta_epsilon_min=args.meta_epsilon_min,
        meta_epsilon_max=args.meta_epsilon_max,
        meta_lr_scale_min=args.meta_lr_scale_min,
        meta_lr_scale_max=args.meta_lr_scale_max,
        meta_update_frequency=args.meta_update_frequency,
        seed=args.seed,
        save_dir=Path(args.save_dir),
    )
    
    run_multi_model_comparison(config)


if __name__ == "__main__":
    main()