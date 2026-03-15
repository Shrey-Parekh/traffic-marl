"""
Parallel multi-seed training wrapper for traffic control experiments.

Usage:
    python train_parallel.py --model_type GAT-DQN --episodes 120 --seeds 1,2,3 --scenario morning_peak

This spawns multiple training processes in parallel, each with:
- Unique seed
- Unique SUMO port (8813, 8814, 8815, ...)
- Unique output directory: output exports/{model}_{seed}/
- Unique filenames: {model}_{seed}_{episodes}_{scenario}_{type}.ext
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-Seed Training")
    
    # Required arguments
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=str, required=True,
                       help="Comma-separated seeds (e.g., '1,2,3')")
    parser.add_argument("--scenario", type=str, required=True,
                       choices=["uniform", "morning_peak", "evening_peak"])
    
    # Optional arguments
    parser.add_argument("--N", type=int, default=9)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gat_n_heads", type=int, default=4)
    parser.add_argument("--gat_dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Parse seeds
    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    logger.info(f"Starting parallel training for {len(seed_list)} seeds: {seed_list}")
    logger.info(f"Model: {args.model_type}, Episodes: {args.episodes}, Scenario: {args.scenario}")
    
    # Create output exports directory
    output_exports = Path("output exports")
    output_exports.mkdir(parents=True, exist_ok=True)
    
    # Spawn parallel processes
    processes = []
    for idx, seed in enumerate(seed_list):
        port = 8813 + idx
        seed_dir = output_exports / f"{args.model_type}_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, "-m", "src.train",
            "--model_type", args.model_type,
            "--episodes", str(args.episodes),
            "--scenario", args.scenario,
            "--seed", str(seed),
            "--N", str(args.N),
            "--max_steps", str(args.max_steps),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--gamma", str(args.gamma),
            "--gat_n_heads", str(args.gat_n_heads),
            "--gat_dropout", str(args.gat_dropout),
            "--save_dir", str(seed_dir),
            "--port", str(port),  # Custom argument for SUMO port
        ]
        
        logger.info(f"Starting seed {seed} on port {port} -> {seed_dir}")
        
        # Redirect output to log file
        log_file = seed_dir / f"{args.model_type}_{seed}_{args.episodes}_{args.scenario}_train.log"
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        processes.append((seed, proc, log_file))
    
    # Wait for all to complete
    logger.info(f"\nWaiting for {len(processes)} processes to complete...")
    results = []
    for seed, proc, log_file in processes:
        returncode = proc.wait()
        status = "SUCCESS" if returncode == 0 else "FAILED"
        results.append((seed, status, log_file))
        
        if status == "SUCCESS":
            logger.info(f"✓ Seed {seed}: SUCCESS")
        else:
            logger.error(f"✗ Seed {seed}: FAILED (see {log_file})")
    
    # Report summary
    logger.info("\n" + "="*70)
    logger.info("PARALLEL TRAINING COMPLETE")
    logger.info("="*70)
    
    successful = sum(1 for _, s, _ in results if s == "SUCCESS")
    failed = len(results) - successful
    
    logger.info(f"\nResults: {successful} successful, {failed} failed out of {len(results)} total")
    
    for seed, status, log_file in results:
        symbol = "✓" if status == "SUCCESS" else "✗"
        seed_dir = output_exports / f"{args.model_type}_{seed}"
        logger.info(f"{symbol} Seed {seed}: {status} - Output: {seed_dir}")
    
    if failed > 0:
        logger.warning(f"\n{failed} seed(s) failed. Check log files for details.")
        sys.exit(1)
    else:
        logger.info("\nAll seeds completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
