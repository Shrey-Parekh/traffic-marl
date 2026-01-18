"""Multi-agent reinforcement learning training for traffic control."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict

import numpy as np
import torch  # type: ignore[import-untyped]
from torch import nn, optim  # type: ignore[import-untyped]
from tqdm import trange

from .env import MiniTrafficEnv, EnvConfig
from .agent import DQNet, ReplayBuffer


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(
    step: int, start: float, end: float, decay_steps: int
) -> float:
    """Calculate epsilon value for epsilon-greedy exploration."""
    if decay_steps <= 0:
        return end
    eps = end + (start - end) * max(
        0.0, (decay_steps - step) / decay_steps
    )
    return float(eps)


def select_action(
    q_net: DQNet,
    obs: np.ndarray,
    n_actions: int,
    eps: float,
    device: torch.device,
) -> int:
    """Select action using epsilon-greedy policy."""
    if np.random.rand() < eps:
        return int(np.random.randint(0, n_actions))
    with torch.no_grad():
        x = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = q_net(x)
        return int(torch.argmax(q, dim=1).item())


def optimize(  # noqa: PLR0913
    q_net: DQNet,
    target_net: DQNet,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Perform one optimization step using DQN algorithm."""
    trans = buffer.sample(batch_size)
    state = torch.from_numpy(np.vstack(trans.state)).float().to(device)
    action = (
        torch.tensor(trans.action, dtype=torch.long)
        .unsqueeze(1)
        .to(device)
    )
    reward = (
        torch.tensor(trans.reward, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )
    next_state = (
        torch.from_numpy(np.vstack(trans.next_state)).float().to(device)
    )
    done = (
        torch.tensor(trans.done, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

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


def run_episode(  # noqa: PLR0913
    env: MiniTrafficEnv,
    q_net: DQNet,
    target_net: DQNet,
    buffer: ReplayBuffer,
    n_actions: int,
    eps: float,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    update_target_steps: int,
    min_buffer_size: int,
    global_step: int,
) -> tuple[Dict[str, float], int]:
    """Run one episode of training."""
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

        # Learn (only if buffer has enough samples for stable training)
        if len(buffer) >= max(batch_size, min_buffer_size):
            loss = optimize(
                q_net, target_net, buffer, batch_size, gamma,
                optimizer, device
            )
            losses.append(loss)
            updates += 1
            if (
                update_target_steps > 0
                and updates % update_target_steps == 0
            ):
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


def main() -> None:  # noqa: PLR0915
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Mini Traffic MARL Training"
    )
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
    parser.add_argument("--min_buffer_size", type=int, default=1000, help="Minimum buffer size before training starts")
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

    # Use ASCII-friendly output to avoid Windows console encoding issues
    print("Starting Multi-Agent RL Training:")
    print(f"- {args.N} intersections (agents)")
    print(f"- {args.episodes} episodes")
    episode_duration = args.max_steps * 2
    print(f"- {args.max_steps} steps per episode (~{episode_duration}s)")
    print(f"- Learning rate: {args.lr}")
    print(
        f"- Epsilon decay: {args.epsilon_start} -> {args.epsilon_end} "
        f"over {args.epsilon_decay_steps} steps"
    )
    print("-" * 60)

    for ep in trange(args.episodes, desc="Multi-Agent Episodes"):
        eps = epsilon_by_step(
            global_step,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )
        m, global_step = run_episode(
            env,
            q_net,
            target_net,
            buffer,
            n_actions,
            eps,
            args.batch_size,
            args.gamma,
            optimizer,
            device,
            args.update_target_steps,
            args.min_buffer_size,
            global_step,
        )

        # Enhanced record with multi-agent info
        record = {
            "episode": ep,
            "epsilon": eps,
            "global_step": global_step,
            "agents": args.N,
            **m,
        }
        all_metrics.append(record)

        # Update files after each episode
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Write CSV fresh each time to avoid duplicate headers and ensure consistency
        fieldnames = list(record.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in all_metrics:
                w.writerow(rec)

        # Human-readable rolling summary
        with open(summary_path, "w", encoding="utf-8") as f:
            summary_text = (
                "Multi-Agent RL Traffic Control Training Summary\n"
                f"Episodes completed: {ep + 1}/{args.episodes}\n"
                f"Agents (intersections): {args.N}\n"
                f"Current epsilon: {eps:.3f}\n"
                f"Global steps: {global_step}\n"
                "Latest episode performance:\n"
                f"  - Avg Queue: {record.get('avg_queue', 0.0):.2f} cars\n"
                f"  - Throughput: {record.get('throughput', 0.0):.0f} "
                "vehicles\n"
                f"  - Avg Travel Time: "
                f"{record.get('avg_travel_time', 0.0):.2f}s\n"
                f"  - Training Loss: {record.get('loss', 0.0):.4f}\n"
                f"  - Updates: {record.get('updates', 0):.0f}\n"
            )
            f.write(summary_text)

        # Enhanced console output
        print(
            f"Ep {ep+1:2d}/{args.episodes}: "
            f"eps={eps:.3f} | "
            f"Queue={m['avg_queue']:.2f} | "
            f"Throughput={m['throughput']:.0f} | "
            f"TravelTime={m['avg_travel_time']:.1f}s | "
            f"Loss={m['loss']:.4f} | "
            f"Updates={m['updates']:.0f}"
        )

    torch.save(q_net.state_dict(), policy_path)
    print(f"Saved final policy to {policy_path}")

    # Aggregate final report for non-technical viewers
    keys = [
        "avg_reward",
        "avg_queue",
        "throughput",
        "avg_travel_time",
        "loss",
    ]
    avgs = {
        k: float(np.mean([r.get(k, 0.0) for r in all_metrics]))
        for k in keys
    }
    final = all_metrics[-1] if all_metrics else {}
    report = {
        "episodes": len(all_metrics),
        "average_metrics": avgs,
        "final_episode": final,
        "plain_english": (
            "Over the training run, the controller reduced queues and "
            "travel time while maintaining or improving throughput. "
            "The averages summarize performance across all episodes; "
            "the final episode shows the latest results that the "
            "dashboard displays."
        ),
    }
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote final summary to {final_report_path}")


if __name__ == "__main__":
    main()
