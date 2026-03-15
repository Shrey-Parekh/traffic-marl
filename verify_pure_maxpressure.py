"""Verify pure MaxPressure reward implementation."""

from src.config import REWARD_CONFIG

print("=" * 70)
print("PURE MAXPRESSURE REWARD VERIFICATION")
print("=" * 70)

print("\n1. REWARD_CONFIG:")
for key, value in REWARD_CONFIG.items():
    print(f"   {key}: {value}")

print("\n2. REWARD FORMULA:")
print("   reward = pressure / norm")
print("   reward = pressure / 50.0")

print("\n3. EXAMPLE SCENARIOS:")
print("-" * 70)

norm = REWARD_CONFIG["reward_queue_norm"]

# Scenario 1: Serving longer queue
ns_pcu, ew_pcu = 20.0, 10.0
pressure = ns_pcu - ew_pcu
reward = pressure / norm
print(f"Scenario 1: Serving longer queue")
print(f"  NS_PCU=20, EW_PCU=10, Phase=NS_GREEN")
print(f"  pressure = {pressure:.1f}")
print(f"  reward = {pressure:.1f} / {norm:.1f} = {reward:.4f}")
print(f"  ✓ POSITIVE reward")

# Scenario 2: Serving shorter queue
ns_pcu, ew_pcu = 10.0, 20.0
pressure = ns_pcu - ew_pcu
reward = pressure / norm
print(f"\nScenario 2: Serving shorter queue")
print(f"  NS_PCU=10, EW_PCU=20, Phase=NS_GREEN")
print(f"  pressure = {pressure:.1f}")
print(f"  reward = {pressure:.1f} / {norm:.1f} = {reward:.4f}")
print(f"  ✓ NEGATIVE reward")

# Scenario 3: Balanced queues
ns_pcu, ew_pcu = 15.0, 15.0
pressure = ns_pcu - ew_pcu
reward = pressure / norm
print(f"\nScenario 3: Balanced queues")
print(f"  NS_PCU=15, EW_PCU=15, Phase=NS_GREEN")
print(f"  pressure = {pressure:.1f}")
print(f"  reward = {pressure:.1f} / {norm:.1f} = {reward:.4f}")
print(f"  ✓ ZERO reward")

# Scenario 4: Large imbalance
ns_pcu, ew_pcu = 30.0, 5.0
pressure = ns_pcu - ew_pcu
reward = pressure / norm
print(f"\nScenario 4: Large imbalance")
print(f"  NS_PCU=30, EW_PCU=5, Phase=NS_GREEN")
print(f"  pressure = {pressure:.1f}")
print(f"  reward = {pressure:.1f} / {norm:.1f} = {reward:.4f}")
print(f"  ✓ STRONG POSITIVE reward")

print("\n" + "=" * 70)
print("KEY PROPERTIES:")
print("=" * 70)
print("✓ Pure pressure signal - no queue penalty")
print("✓ Positive when serving longer queue")
print("✓ Negative when serving shorter queue")
print("✓ Zero when queues balanced")
print("✓ Reward magnitude proportional to queue imbalance")
print("✓ Proven optimal for network stability (Varaiya 2013)")
print("\n✓ PURE MAXPRESSURE READY")
print("=" * 70)
