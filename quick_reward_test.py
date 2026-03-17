"""Quick test to validate reward scale is reasonable for learning."""

from src.env_sumo import PuneSUMOEnv
from src.baseline import MaxPressureController
from src.config import SUMO_CONFIG
import numpy as np

SCENARIO = "morning_peak"
N_AGENTS = 9
MAX_STEPS = 100
SEED = 1


def make_env():
    return PuneSUMOEnv({
        "render": False,
        "scenario": SCENARIO,
        "n_intersections": N_AGENTS,
        "max_steps": MAX_STEPS,
        "seed": SEED,
    })


def test_reward_scale():
    """Test reward magnitude and variance across three policies."""

    # Test 1: Random policy
    print("=" * 60)
    print("TEST 1: Random Policy (baseline for comparison)")
    print("=" * 60)
    env = make_env()
    obs = env.reset()
    random_rewards = []
    for _ in range(MAX_STEPS):
        actions = [np.random.choice([0, 1]) if env.steps_since_switch[i] >= 5 else 0
                   for i in range(N_AGENTS)]
        obs, rewards, done, info = env.step(actions)
        random_rewards.append(np.mean(rewards))
    print(f"Random policy reward: {np.mean(random_rewards):.4f} ± {np.std(random_rewards):.4f}")
    print(f"Range: [{np.min(random_rewards):.4f}, {np.max(random_rewards):.4f}]")
    env.close()

    # Test 2: MaxPressure policy (should be best adaptive baseline)
    print("\n" + "=" * 60)
    print("TEST 2: MaxPressure Policy (should be better than random)")
    print("=" * 60)
    mp = MaxPressureController(
        n_agents=N_AGENTS,
        min_green_steps=SUMO_CONFIG["min_green_steps"],
        pressure_threshold=3.0,
    )
    env = make_env()
    obs = env.reset()
    mp.reset()
    mp_rewards = []
    for _ in range(MAX_STEPS):
        actions = mp.act(obs)
        obs, rewards, done, info = env.step(actions)
        mp_rewards.append(np.mean(rewards))
    print(f"MaxPressure reward: {np.mean(mp_rewards):.4f} ± {np.std(mp_rewards):.4f}")
    print(f"Range: [{np.min(mp_rewards):.4f}, {np.max(mp_rewards):.4f}]")
    env.close()

    # Test 3: Keep-all policy (worst case)
    print("\n" + "=" * 60)
    print("TEST 3: Keep-All Policy (should be worst)")
    print("=" * 60)
    env = make_env()
    env.reset()
    keep_rewards = []
    for _ in range(MAX_STEPS):
        _, rewards, done, info = env.step([0] * N_AGENTS)
        keep_rewards.append(np.mean(rewards))
    print(f"Keep-all reward: {np.mean(keep_rewards):.4f} ± {np.std(keep_rewards):.4f}")
    print(f"Range: [{np.min(keep_rewards):.4f}, {np.max(keep_rewards):.4f}]")
    env.close()

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    random_mean = np.mean(random_rewards)
    mp_mean = np.mean(mp_rewards)
    keep_mean = np.mean(keep_rewards)

    print(f"\nReward differences:")
    print(f"  MaxPressure vs Random:   {mp_mean - random_mean:.4f} "
          f"({((mp_mean - random_mean) / abs(random_mean) * 100):.1f}%)")
    print(f"  MaxPressure vs Keep-all: {mp_mean - keep_mean:.4f}")
    print(f"  Random vs Keep-all:      {random_mean - keep_mean:.4f}")

    reward_range = max(mp_mean, random_mean, keep_mean) - min(mp_mean, random_mean, keep_mean)
    print(f"\nReward scale assessment:")
    print(f"  Total reward range: {reward_range:.4f}")

    if reward_range < 0.01:
        print("  ❌ TOO SMALL - rewards too weak for learning")
    elif reward_range < 0.1:
        print("  ⚠️  MARGINAL - may work but slow convergence")
    elif reward_range < 1.0:
        print("  ✓ GOOD - reasonable reward scale for DQN")
    else:
        print("  ⚠️  LARGE - may need gradient clipping")

    if mp_mean > random_mean and mp_mean > keep_mean:
        print("  ✓ MaxPressure is best rule-based policy (as expected)")
    else:
        print("  ❌ WARNING: MaxPressure should have highest reward among rule-based policies")

    avg_std = np.mean([np.std(random_rewards), np.std(mp_rewards), np.std(keep_rewards)])
    if reward_range > 0 and avg_std / reward_range > 2.0:
        print(f"  ⚠️  High variance (std={avg_std:.4f}) - may need more stable rewards")
    else:
        print(f"  ✓ Variance is reasonable (std={avg_std:.4f})")

    return reward_range > 0.01 and mp_mean > random_mean


if __name__ == "__main__":
    success = test_reward_scale()
    print("\n" + "=" * 60)
    if success:
        print("✓ REWARD SCALE IS REASONABLE - proceed with training")
    else:
        print("❌ REWARD SCALE NEEDS ADJUSTMENT")
    print("=" * 60)
