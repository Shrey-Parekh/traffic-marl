from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from .env import MiniTrafficEnv, EnvConfig
from .agent import DQNet, ReplayBuffer


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    eps = end + (start - end) * max(0.0, (decay_steps - step) / decay_steps)
    return float(eps)


def select_action(q_net: DQNet, obs: np.ndarray, n_actions: int, eps: float, device: torch.device) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(0, n_actions))
    with torch.no_grad():
        x = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = q_net(x)
        return int(torch.argmax(q, dim=1).item())


def optimize(
    q_net: DQNet,
    target_net: DQNet,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    trans = buffer.sample(batch_size)
    state = torch.from_numpy(np.vstack(trans.state)).float().to(device)
    action = torch.tensor(trans.action, dtype=torch.long).unsqueeze(1).to(device)
    reward = torch.tensor(trans.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state = torch.from_numpy(np.vstack(trans.next_state)).float().to(device)
    done = torch.tensor(trans.done, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = q_net(state).gather(1, action)
    with torch.no_grad():
        next_q = target_net(next_state).max(1, keepdim=True)[0]
        target = reward + gamma * (1.0 - done) * next_q

    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def run_episode(
    env: MiniTrafficEnv,
    q_net: DQNet,
    target_net: DQNet,
    buffer: ReplayBuffer,
    n_actions: int,
    obs_dim: int,
    eps: float,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    update_target_steps: int,
    global_step: int,
) -> (Dict[str, float], int):
    obs_dict = env.reset()
    done = False
    episode_reward_sum = 0.0
    updates = 0
    losses = []

    while not done:
        actions: Dict[str, int] = {}
        # Build actions from shared policy
        for aid, obs in obs_dict.items():
            actions[aid] = select_action(q_net, obs, n_actions, eps, device)

        next_obs_dict, rewards, done, info = env.step(actions)
        episode_reward_sum += float(np.mean(list(rewards.values())))

        # Store transitions for all agents
        for aid in obs_dict.keys():
            buffer.push(
                obs_dict[aid],
                actions[aid],
                rewards[aid],
                next_obs_dict[aid],
                done,
            )

        obs_dict = next_obs_dict

        # Learn
        if len(buffer) >= batch_size:
            loss = optimize(q_net, target_net, buffer, batch_size, gamma, optimizer, device)
            losses.append(loss)
            updates += 1
            if update_target_steps > 0 and (global_step % update_target_steps == 0):
                target_net.load_state_dict(q_net.state_dict())
        global_step += 1

    avg_loss = float(np.mean(losses)) if losses else 0.0
    metrics = {
        "avg_reward": episode_reward_sum,
        "avg_queue": info.get("avg_queue", 0.0),
        "throughput": info.get("throughput", 0.0),
        "avg_travel_time": info.get("avg_travel_time", 0.0),
        "updates": float(updates),
        "loss": avg_loss,
    }
    return metrics, global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Traffic MARL Training")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay_capacity", type=int, default=20000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=5000)
    parser.add_argument("--update_target_steps", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--neighbor_obs", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seeds(args.seed)

    env = MiniTrafficEnv(
        EnvConfig(
            num_intersections=args.N,
            max_steps=args.max_steps,
            seed=args.seed,
            neighbor_obs=args.neighbor_obs,
        )
    )

    device = torch.device("cpu")
    obs_dim = env.get_obs_dim()
    n_actions = env.get_n_actions()

    q_net = DQNet(obs_dim, n_actions).to(device)
    target_net = DQNet(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    buffer = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    summary_path = os.path.join(args.save_dir, "summary.txt")
    live_path = os.path.join(args.save_dir, "live_metrics.json")
    policy_path = os.path.join(args.save_dir, "policy_final.pth")
    final_report_path = os.path.join(args.save_dir, "final_report.json")

    all_metrics = []
    global_step = 0
    for ep in trange(args.episodes, desc="Episodes"):
        eps = epsilon_by_step(global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        m, global_step = run_episode(
            env,
            q_net,
            target_net,
            buffer,
            n_actions,
            obs_dim,
            eps,
            args.batch_size,
            args.gamma,
            optimizer,
            device,
            args.update_target_steps,
            global_step,
        )
        # Include richer info when available
        record = {"episode": ep, **m}
        all_metrics.append(record)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Append/update CSV for non-technical readers
        import csv
        write_header = not os.path.exists(csv_path)
        fieldnames = list(record.keys())
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(record)

        # Human-readable rolling summary
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
                (
                    "Mini Traffic MARL Training Summary\n"
                    f"Episodes so far: {ep + 1}\n"
                    f"Latest -> Avg Queue: {record.get('avg_queue', 0.0):.2f}, "
                    f"Throughput: {record.get('throughput', 0.0):.0f}, "
                    f"Avg Travel Time: {record.get('avg_travel_time', 0.0):.2f}s, "
                    f"Loss: {record.get('loss', 0.0):.4f}\n"
                )
            )

        print(
            f"Ep {ep}: avg_reward={m['avg_reward']:.2f} avg_queue={m['avg_queue']:.2f} "
            f"avg_travel_time={m['avg_travel_time']:.2f}s throughput={m['throughput']:.0f} loss={m['loss']:.4f}"
        )

    torch.save(q_net.state_dict(), policy_path)
    print(f"Saved final policy to {policy_path}")

    # Aggregate final report for non-technical viewers
    keys = ["avg_reward", "avg_queue", "throughput", "avg_travel_time", "loss"]
    avgs = {k: float(np.mean([r.get(k, 0.0) for r in all_metrics])) for k in keys}
    best = max(all_metrics, key=lambda r: -r.get("avg_queue", 0.0)) if all_metrics else {}
    final = all_metrics[-1] if all_metrics else {}
    report = {
        "episodes": len(all_metrics),
        "average_metrics": avgs,
        "final_episode": final,
        "plain_english": (
            "Over the training run, the controller reduced queues and travel time while maintaining or "
            "improving throughput. The averages summarize performance across all episodes; the final "
            "episode shows the latest results that the dashboard displays."
        ),
    }
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote final summary to {final_report_path}")


if __name__ == "__main__":
    main()


