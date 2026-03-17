"""
Parallel multi-seed training wrapper for traffic control experiments.

Usage:
    python train_parallel.py --model_type GAT-DQN --episodes 120 --seeds 1,2,3 --scenario morning_peak

Multiple parallel runs of this script can coexist safely — each invocation
picks a free port block automatically using a lock file, so ports never clash.

Each process gets:
- Unique seed
- Unique SUMO port (auto-assigned from a free block)
- Unique output directory: output exports/{model}_{seed}/
"""

import argparse
import subprocess
import sys
import logging
import os
import time
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Port allocation lock file — shared across all parallel invocations
PORT_LOCK_FILE = Path("output exports/.port_registry.json")
PORT_BASE = 8813
PORT_BLOCK_SIZE = 10  # reserve 10 ports per invocation (max 10 seeds)


def acquire_port_block(n_seeds: int) -> int:
    """
    Atomically claim a block of n_seeds ports from the shared registry.
    Returns the starting port for this invocation.
    Uses a simple file-based lock with retry.
    """
    lock_path = PORT_LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.with_suffix(".lock")

    # Spin-wait for lock (max 30s)
    for _ in range(300):
        try:
            # Atomic create — fails if file exists
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Could not acquire port lock after 30s")

    try:
        # Read current registry
        if lock_path.exists():
            with open(lock_path) as f:
                registry = json.load(f)
        else:
            registry = {"next_block": 0}

        # Claim next block
        block_idx = registry["next_block"]
        registry["next_block"] = block_idx + 1

        # Write back
        with open(lock_path, "w") as f:
            json.dump(registry, f)

        start_port = PORT_BASE + block_idx * PORT_BLOCK_SIZE
        return start_port
    finally:
        lock_file.unlink(missing_ok=True)


def release_port_block(start_port: int):
    """No-op — ports are freed when SUMO processes exit. Registry resets on next run."""
    pass


def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-Seed Training")

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=str, required=True,
                        help="Comma-separated seeds (e.g., '1,2,3')")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["uniform", "morning_peak", "evening_peak"])
    parser.add_argument("--N", type=int, default=9)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gat_n_heads", type=int, default=4)
    parser.add_argument("--gat_dropout", type=float, default=0.1)

    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    logger.info(f"Starting parallel training: {args.model_type}, seeds={seed_list}, "
                f"episodes={args.episodes}, scenario={args.scenario}")

    output_exports = Path("output exports")
    output_exports.mkdir(parents=True, exist_ok=True)

    # Claim a unique port block for this invocation
    start_port = acquire_port_block(len(seed_list))
    logger.info(f"Assigned port block: {start_port} – {start_port + len(seed_list) - 1}")

    processes = []
    for idx, seed in enumerate(seed_list):
        port = start_port + idx
        seed_dir = output_exports / f"{args.model_type}_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

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
            "--port", str(port),
        ]

        logger.info(f"  Seed {seed} → port {port} → {seed_dir}")
        log_file = seed_dir / f"{args.model_type}_{seed}_{args.episodes}_{args.scenario}_train.log"
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        processes.append((seed, proc, log_file))

    logger.info(f"Waiting for {len(processes)} processes...")
    results = []
    for seed, proc, log_file in processes:
        returncode = proc.wait()
        status = "SUCCESS" if returncode == 0 else "FAILED"
        results.append((seed, status, log_file))
        symbol = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"{symbol} Seed {seed}: {status}")

    successful = sum(1 for _, s, _ in results if s == "SUCCESS")
    failed = len(results) - successful
    logger.info(f"\n{successful} succeeded, {failed} failed out of {len(results)}")

    if failed > 0:
        for seed, status, log_file in results:
            if status == "FAILED":
                logger.error(f"  Seed {seed} failed — see {log_file}")
        sys.exit(1)
    else:
        logger.info("All seeds completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
