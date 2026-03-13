"""Verification tests for the 3 literature-grounded fixes."""

# Test 1 — Epsilon scheduler
print("=" * 60)
print("TEST 1: Epsilon Scheduler")
print("=" * 60)

from src.agent import EpsilonScheduler
from src.config import EPSILON_CONFIG

s = EpsilonScheduler(total_episodes=80, max_steps_per_episode=150,
                     model_type="GNN-DQN")
eps_values = [s.step() for _ in range(80 * 150)]

assert eps_values[0] > 0.98,         "START: epsilon must begin near 1.0"
assert eps_values[5000] > 0.35,      "MIDDLE: epsilon must still be above 0.35 at step 5000"
assert eps_values[-1] <= 0.05,       "END: epsilon must reach 0.05"
assert eps_values[100] < eps_values[0], "DECAY: epsilon must be decreasing"
print(f"Epsilon at step 0:    {eps_values[0]:.3f}")
print(f"Epsilon at step 5000: {eps_values[5000]:.3f}")
print(f"Epsilon at step 12000:{eps_values[-1]:.3f}")
print("✓ EPSILON TEST PASSED\n")

# Test 2 — PER buffer
print("=" * 60)
print("TEST 2: Prioritized Experience Replay")
print("=" * 60)

from src.agent import PrioritizedReplayBuffer
import numpy as np

buf = PrioritizedReplayBuffer(capacity=1000)
for i in range(200):
    buf.add(
        np.zeros(15), 0, -0.5, np.zeros(15), False
    )
samples, indices, weights = buf.sample(32, beta=0.4)
assert len(samples) == 32,           "SAMPLE: must return 32 transitions"
assert len(indices) == 32,           "INDICES: must return 32 indices"
assert weights.shape == (32,),       "WEIGHTS: must return 32 weights"
assert weights.max() <= 1.0,         "WEIGHTS: must be normalized to max 1.0"
buf.update_priorities(indices, np.random.rand(32))
samples2, _, _ = buf.sample(32, beta=0.4)
assert len(samples2) == 32,          "UPDATE: sampling after priority update must work"
print("✓ PER TEST PASSED\n")

# Test 3 — Reward range
print("=" * 60)
print("TEST 3: Reward Function")
print("=" * 60)

from src.env_sumo import PuneSUMOEnv
env = PuneSUMOEnv({"render": False, "scenario": "uniform"})
obs = env.reset()
all_rewards = []
for _ in range(30):
    obs, rewards, done, info = env.step([0] * 9)
    all_rewards.extend(rewards)
env.close()
assert all(r >= -1.0 for r in all_rewards), "REWARD: must be >= -1.0"
assert all(r <= 0.20 for r in all_rewards), "REWARD: must be <= 0.20"
print(f"Reward range: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
print("✓ REWARD TEST PASSED\n")

print("=" * 60)
print("ALL 3 TESTS PASSED — ready to train")
print("=" * 60)
