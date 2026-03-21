from src.env_sumo import PuneSUMOEnv
import numpy as np

env = PuneSUMOEnv({'render': False, 'scenario': 'morning_peak'})
obs = env.reset()
ns_pcus, ew_pcus = [], []

for _ in range(300):
    obs, r, done, info = env.step([0] * 9)
    for i in range(9):
        ns_pcus.append(float(obs[i][2]))
        ew_pcus.append(float(obs[i][3]))

env.close()
print(f"Avg NS PCU:        {np.mean(ns_pcus):.2f}")
print(f"Avg EW PCU:        {np.mean(ew_pcus):.2f}")
print(f"Max NS PCU:        {np.max(ns_pcus):.2f}")
print(f"Avg total:         {np.mean(ns_pcus) + np.mean(ew_pcus):.2f}")
print(f"Pressure diff avg: {np.mean(np.abs(np.array(ns_pcus) - np.array(ew_pcus))):.2f}")
