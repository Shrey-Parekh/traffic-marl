from src.env_sumo import PuneSUMOEnv
import numpy as np


def smart(obs, n=9):
    actions = []
    for i in range(n):
        o = obs[i]
        ns, ew, phase, steps = o[2], o[3], int(o[4]), o[5]
        if steps < 5:
            actions.append(0)
        elif phase == 0 and ew > ns * 1.3:
            actions.append(1)
        elif phase == 2 and ns > ew * 1.3:
            actions.append(1)
        else:
            actions.append(0)
    return actions


env = PuneSUMOEnv({'render': False, 'scenario': 'morning_peak'})
obs = env.reset()
keep_r = []
for _ in range(300):
    obs, r, d, i = env.step([0] * 9)
    keep_r.append(np.mean(r))
env.close()

env = PuneSUMOEnv({'render': False, 'scenario': 'morning_peak'})
obs = env.reset()
smart_r = []
for _ in range(300):
    obs, r, d, i = env.step(smart(obs))
    smart_r.append(np.mean(r))
env.close()

diff = np.mean(smart_r) - np.mean(keep_r)
print(f"Keep:  {np.mean(keep_r):.4f}")
print(f"Smart: {np.mean(smart_r):.4f}")
print(f"Diff:  {diff:.4f}")
print("PASS" if diff > 0 else "FAIL")
