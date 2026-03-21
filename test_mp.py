# save as test_mp.py and run: python test_mp.py

from src.env_sumo import PuneSUMOEnv
import numpy as np

thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]

for thresh in thresholds:
    env = PuneSUMOEnv({'render': False, 'scenario': 'morning_peak'})
    obs = env.reset()
    switches = 0
    queues = []

    for _ in range(300):
        actions = []
        for i in range(9):
            ns_pcu, ew_pcu = env.get_raw_queue_pcu(i)
            phase = env.current_phases[i]
            steps = env.steps_since_switch[i]

            if phase == 1:
                actions.append(0)
            elif steps < env.min_green_steps:
                actions.append(0)
            elif phase == 0:
                actions.append(1 if (ew_pcu - ns_pcu) > thresh else 0)
            else:
                actions.append(1 if (ns_pcu - ew_pcu) > thresh else 0)

        switches += sum(actions)
        obs, r, done, info = env.step(actions)
        queues.append(info['avg_queue_pcu'])

    env.close()
    print(
        f"thresh={thresh:.1f} "
        f"switches={switches/9:.0f}/agent "
        f"queue={np.mean(queues):.2f} "
        f"travel={info['avg_travel_time']:.1f} "
        f"throughput={info['throughput']}"
    )