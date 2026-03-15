"""Verify phase 2 bug is fixed."""
from src.env_sumo import PuneSUMOEnv
import numpy as np

env = PuneSUMOEnv({
    'render': False,
    'scenario': 'morning_peak',
    'n_intersections': 9,
    'max_steps': 600,
    'seed': 42
})

obs = env.reset()
rewards_by_phase = {0: [], 1: [], 2: []}

for _ in range(500):
    obs, r, d, info = env.step([1] * 9)
    if d:
        break
    for i in range(9):
        phase = int(obs[i][4])
        rewards_by_phase[phase].append(r[i])

env.close()

print("=" * 70)
print("PHASE 2 BUG VERIFICATION")
print("=" * 70)

for ph, name in [(0, 'NS green'), (1, 'clearance'), (2, 'EW green')]:
    vals = rewards_by_phase[ph]
    if vals:
        mean_val = np.mean(vals)
        nonzero = sum(1 for v in vals if abs(v) > 0.0001)
        print(f'Phase {ph} ({name:12s}): mean={mean_val:7.4f} nonzero={nonzero:4d}/{len(vals):4d}')
    else:
        print(f'Phase {ph} ({name:12s}): NO DATA')

print("\n" + "=" * 70)
if rewards_by_phase[2]:
    nonzero_ew = sum(1 for v in rewards_by_phase[2] if abs(v) > 0.0001)
    if nonzero_ew > 0:
        print("✓ PHASE 2 BUG FIXED - EW green has non-zero rewards")
    else:
        print("✗ PHASE 2 BUG STILL PRESENT - EW green all zeros")
else:
    print("✗ Phase 2 never reached in simulation")
print("=" * 70)
