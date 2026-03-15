"""
Patch to add to train.py for parallel multi-seed training.

Add this at the start of main() function, right after args = parser.parse_args()
"""

# After: args = parser.parse_args()
# Add this block:

# Handle parallel multi-seed training
if args.seeds is not None:
    import subprocess
    import sys
    
    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    logger.info(f"Starting parallel training for {len(seed_list)} seeds: {seed_list}")
    logger.info(f"Each seed will use ports starting from 8813")
    
    # Spawn parallel processes
    processes = []
    for idx, seed in enumerate(seed_list):
        port = 8813 + idx
        
        # Build command for this seed
        cmd = [
            sys.executable, "-m", "src.train",
            "--model_type", args.model_type,
            "--episodes", str(args.episodes),
            "--scenario", args.scenario,
            "--seed", str(seed),
            "--port", str(port),  # Pass port as argument
            "--N", str(args.N),
            "--max_steps", str(args.max_steps),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            # Add other args as needed
        ]
        
        logger.info(f"Starting seed {seed} on port {port}")
        proc = subprocess.Popen(cmd)
        processes.append((seed, proc))
    
    # Wait for all to complete
    results = []
    for seed, proc in processes:
        returncode = proc.wait()
        status = "SUCCESS" if returncode == 0 else "FAILED"
        results.append((seed, status))
        logger.info(f"Seed {seed}: {status}")
    
    # Report summary
    logger.info("\n" + "="*60)
    logger.info("PARALLEL TRAINING COMPLETE")
    logger.info("="*60)
    for seed, status in results:
        symbol = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"{symbol} Seed {seed}: {status}")
    
    successful = sum(1 for _, s in results if s == "SUCCESS")
    logger.info(f"\nCompleted: {successful}/{len(seed_list)} seeds successful")
    
    sys.exit(0)  # Exit after parallel training

# Continue with single-seed training...
# (rest of main() continues normally)
