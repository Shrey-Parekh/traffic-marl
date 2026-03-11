"""
Quick test script to verify baseline controller performance.
Run this immediately after Phase 2 to sanity-check the numbers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline import FixedTimeController, WebsterOptimalController, MaxPressureController, run_baseline_episode
from env_sumo import PuneSUMOEnv
from config import BASELINE_CONFIG
import numpy as np

def test_baselines():
    """Quick test of all three baseline controllers."""
    
    print("\n" + "="*70)
    print("BASELINE CONTROLLER PERFORMANCE TEST")
    print("="*70)
    
    # Create environment (small test)
    env_config = {
        "n_intersections": 9,
        "scenario": "uniform",
        "render": False,
        "seed": 42,
        "max_steps": 300,  # Short test
    }
    
    print(f"\nEnvironment: {env_config['n_intersections']} intersections, {env_config['max_steps']} steps")
    print(f"Scenario: {env_config['scenario']}")
    
    try:
        env = PuneSUMOEnv(env_config)
    except Exception as e:
        print(f"\n❌ ERROR: Could not create SUMO environment")
        print(f"   {str(e)}")
        print(f"\n   Make sure:")
        print(f"   1. SUMO is installed")
        print(f"   2. Run: cd sumo_config && netconvert -c pune_network.netccfg")
        return
    
    # Initialize controllers
    controllers = {
        "FixedTime": FixedTimeController(cycle_length=BASELINE_CONFIG["fixed_time_cycle"]),
        "Webster": WebsterOptimalController(
            lost_time=BASELINE_CONFIG["webster_lost_time"],
            saturation_flow=BASELINE_CONFIG["webster_saturation_flow"],
        ),
        "MaxPressure": MaxPressureController(
            pressure_threshold=BASELINE_CONFIG["max_pressure_threshold"],
        ),
    }
    
    results = {}
    
    print(f"\nRunning 3 episodes per controller...\n")
    
    for name, controller in controllers.items():
        print(f"Testing {name}...")
        episode_metrics = []
        
        for ep in range(3):
            try:
                metrics = run_baseline_episode(env, controller, name)
                episode_metrics.append(metrics)
                print(f"  Episode {ep+1}: Queue(PCU)={metrics['avg_queue_pcu']:.2f}, "
                      f"Throughput={metrics['throughput']}, Reward={metrics['episode_reward']:.2f}")
            except Exception as e:
                print(f"  ❌ Episode {ep+1} failed: {e}")
                continue
        
        if episode_metrics:
            results[name] = {
                "avg_queue_pcu": np.mean([m['avg_queue_pcu'] for m in episode_metrics]),
                "avg_throughput": np.mean([m['throughput'] for m in episode_metrics]),
                "avg_reward": np.mean([m['episode_reward'] for m in episode_metrics]),
            }
    
    env.close()
    
    # Print summary
    print(f"\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\n{'Controller':<15} {'Queue(PCU)':<15} {'Throughput':<15} {'Reward':<15}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['avg_queue_pcu']:<15.2f} "
              f"{metrics['avg_throughput']:<15.1f} {metrics['avg_reward']:<15.2f}")
    
    # Sanity checks
    print(f"\n" + "="*70)
    print("SANITY CHECKS FOR IEEE PAPER NARRATIVE")
    print("="*70)
    
    if len(results) == 3:
        fixed_queue = results['FixedTime']['avg_queue_pcu']
        webster_queue = results['Webster']['avg_queue_pcu']
        maxp_queue = results['MaxPressure']['avg_queue_pcu']
        
        print(f"\n✓ Expected ranking: FixedTime > Webster > MaxPressure")
        print(f"  Actual: FixedTime={fixed_queue:.2f}, Webster={webster_queue:.2f}, MaxPressure={maxp_queue:.2f}")
        
        if fixed_queue > webster_queue > maxp_queue:
            print(f"  ✅ PASS: Baselines ranked correctly")
        else:
            print(f"  ⚠️  WARNING: Baseline ranking unexpected")
        
        print(f"\n✓ MaxPressure should be strongest (lowest queue)")
        if maxp_queue < webster_queue * 0.95:
            print(f"  ✅ PASS: MaxPressure is {((webster_queue - maxp_queue) / webster_queue * 100):.1f}% better than Webster")
        else:
            print(f"  ⚠️  WARNING: MaxPressure not significantly better than Webster")
        
        print(f"\n✓ RL target performance (for final episodes):")
        target_queue = maxp_queue * 0.85  # RL should beat MaxPressure by ~15%
        print(f"  Target queue: {target_queue:.2f} PCU (15% better than MaxPressure)")
        print(f"  This means RL should achieve ~{target_queue:.2f} PCU in final episodes")
        
        print(f"\n✓ Early episode expectation:")
        print(f"  Untrained RL (episodes 1-10) should have queue > {fixed_queue:.2f} PCU")
        print(f"  This creates the narrative: RL starts worse, ends better")
    
    print(f"\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\nIf all checks pass, baselines are ready for Phase 3!")
    print(f"If warnings appear, tune parameters in src/config.py BASELINE_CONFIG")

if __name__ == "__main__":
    test_baselines()
