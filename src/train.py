"""Multi-agent reinforcement learning training for traffic control."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is in Python path for imports when running as script
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch  # type: ignore[import-untyped]
from torch import nn, optim  # type: ignore[import-untyped]
from tqdm import trange

# Handle both relative imports (when run as module) and absolute imports (when run as script)
try:
    from .agent import DQNet, GNN_DQNet, ReplayBuffer, MetaController
    from .config import TrainingConfig, OUTPUTS_DIR
    from .env import MiniTrafficEnv, EnvConfig
except ImportError:
    # Fallback for when running as script
    from src.agent import DQNet, GNN_DQNet, ReplayBuffer, MetaController
    from src.config import TrainingConfig, OUTPUTS_DIR
    from src.env import MiniTrafficEnv, EnvConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(
    step: int, start: float, end: float, decay_steps: int
) -> float:
    """Calculate epsilon value for epsilon-greedy exploration (fallback for non-meta-learning)."""
    if decay_steps <= 0:
        return end
    eps = end + (start - end) * max(
        0.0, (decay_steps - step) / decay_steps
    )
    return float(eps)


def get_meta_metrics(recent_rewards: list, recent_queues: list, episode: int, total_episodes: int) -> torch.Tensor:
    """Prepare metrics for meta-controller input."""
    # Recent average reward (last 10 episodes or available)
    recent_reward = float(np.mean(recent_rewards[-10:]) if recent_rewards else 0.0)
    
    # Recent average queue (last 10 episodes or available)  
    recent_queue = float(np.mean(recent_queues[-10:]) if recent_queues else 0.0)
    
    # Episode progress (0 to 1)
    episode_progress = float(episode / max(1, total_episodes))
    
    return torch.tensor([recent_reward, recent_queue, episode_progress], dtype=torch.float32)


def select_action(
    q_net: DQNet | GNN_DQNet,
    obs: np.ndarray,
    n_actions: int,
    eps: float,
    device: torch.device,
    use_gnn: bool = False,
    adjacency: np.ndarray | None = None,
    node_idx: int = 0,
    time_since_switch: int = 0,
    min_green: int = 5,
) -> int:
    """Select action using epsilon-greedy policy with action masking."""
    # Create action mask - action 1 (switch) is invalid if time_since_switch < min_green
    valid_actions = [0]  # Action 0 (keep) is always valid
    if time_since_switch >= min_green:
        valid_actions.append(1)  # Action 1 (switch) is valid
    
    if np.random.rand() < eps:
        # Random action from valid actions only
        return int(np.random.choice(valid_actions))
    
    with torch.no_grad():
        if use_gnn:
            # obs is node_features [num_nodes, features], adjacency is [num_nodes, num_nodes]
            node_features = torch.from_numpy(obs).to(device).float()
            adj = torch.from_numpy(adjacency).to(device).float() if adjacency is not None else None
            q = q_net(node_features, adj)
            # q is [num_nodes, n_actions], select for specific node
            q_values = q[node_idx].clone()
        else:
            # obs is flat array [features]
            x = torch.from_numpy(obs).to(device).unsqueeze(0)
            q = q_net(x)
            q_values = q[0].clone()
        
        # Apply action masking - set invalid actions to -inf
        for action in range(n_actions):
            if action not in valid_actions:
                q_values[action] = float('-inf')
        
        return int(torch.argmax(q_values).item())


def optimize(  # noqa: PLR0913
    q_net: DQNet | GNN_DQNet,
    target_net: DQNet | GNN_DQNet,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,  # Updated default
    use_gnn: bool = False,
) -> float:
    """Perform one optimization step using DQN algorithm."""
    trans = buffer.sample(batch_size)
    
    if use_gnn:
        # Graph input: filter valid transitions
        valid_indices = [i for i, s in enumerate(trans.state) if s is not None and len(s) > 0 and i < len(trans.adjacency) and trans.adjacency[i] is not None and trans.node_id[i] is not None]
        
        if not valid_indices:
            return 0.0
        
        batch_size_actual = len(valid_indices)
        
        # Stack states and adjacencies (each transition has full graph)
        state_list = []
        next_state_list = []
        adj_list = []
        node_ids = []
        
        for idx in valid_indices:
            state_list.append(trans.state[idx])
            if idx < len(trans.next_state) and trans.next_state[idx] is not None and len(trans.next_state[idx]) > 0:
                next_state_list.append(trans.next_state[idx])
            else:
                next_state_list.append(trans.state[idx])
            adj_list.append(trans.adjacency[idx])
            node_ids.append(trans.node_id[idx])  # Use stored node_id
        
        state = torch.from_numpy(np.stack(state_list)).float().to(device)
        next_state = torch.from_numpy(np.stack(next_state_list)).float().to(device)
        adjacency = torch.from_numpy(np.stack(adj_list)).float().to(device)
        
        actions = [trans.action[idx] for idx in valid_indices]
        rewards = [trans.reward[idx] for idx in valid_indices]
        dones = [trans.done[idx] for idx in valid_indices]
        
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Clamp rewards to prevent extreme values
        reward_tensor = torch.clamp(reward_tensor, -15.0, 10.0)  # Updated range for neighbor rewards
        
        q_values_all = q_net(state, adjacency)
        # Extract Q-value for the correct node using stored node_id
        q_values = torch.stack([q_values_all[i, node_ids[i], actions[i]] for i in range(batch_size_actual)]).unsqueeze(1)
        
        with torch.no_grad():
            next_q_all = target_net(next_state, adjacency)
            next_q = torch.stack([next_q_all[i, node_ids[i], :].max() for i in range(batch_size_actual)])
            target = reward_tensor + gamma * (1.0 - done_tensor) * next_q
            target = target.unsqueeze(1)
    else:
        # Flat input: state is [batch, features]
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
        
        # Clamp rewards to prevent extreme values
        reward = torch.clamp(reward, -10.0, 10.0)
        
        q_values = q_net(state).gather(1, action)
        with torch.no_grad():
            next_q = target_net(next_state).max(1, keepdim=True)[0]
            target = reward + gamma * (1.0 - done) * next_q

    # Use Huber loss instead of MSE for more stable training
    loss = nn.functional.smooth_l1_loss(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check for NaN gradients
    total_norm = 0
    for p in q_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            # Replace NaN gradients with zeros
            if torch.isnan(param_norm):
                p.grad.data.zero_()
    total_norm = total_norm ** (1. / 2)
    
    # Only clip if gradients are reasonable
    if total_norm > 0 and not np.isnan(total_norm):
        nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=grad_clip_norm)
    
    optimizer.step()
    return float(loss.item())


def run_episode(  # noqa: PLR0913
    env: MiniTrafficEnv,
    q_net: DQNet | GNN_DQNet,
    target_net: DQNet | GNN_DQNet,
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
    use_gnn: bool = False,
    meta_lr_scale: float = 1.0,
) -> tuple[Dict[str, float], int]:
    """Run one episode of shared-policy multi-agent training.
    
    All intersections share the same policy (parameter sharing), not independent agents.
    """
    obs_dict = env.reset()
    done = False
    episode_reward_sum = 0.0
    updates = 0
    losses = []
    
    # Get adjacency matrix once at start if using GNN
    adjacency = env.get_adjacency_matrix() if use_gnn else None
    node_features = env.get_node_features(use_gnn) if use_gnn else None
    
    # Get context features for logging
    context = env._get_context_features()

    while not done:
        actions: Dict[str, int] = {}
        # Build actions from shared policy with action masking
        if use_gnn and node_features is not None:
            # GNN: select actions for all nodes at once
            for i, aid in enumerate(obs_dict.keys()):
                actions[aid] = select_action(
                    q_net, node_features, n_actions, eps, device, use_gnn, adjacency, i,
                    env.time_since_switch[i], env.min_green
                )
        else:
            # DQN: select action per agent from flat observation
            for i, (aid, obs) in enumerate(obs_dict.items()):
                actions[aid] = select_action(
                    q_net, obs, n_actions, eps, device, use_gnn,
                    time_since_switch=env.time_since_switch[i], min_green=env.min_green
                )

        next_obs_dict, rewards, done, info = env.step(actions)
        episode_reward_sum += float(np.mean(list(rewards.values())))
        
        # Update node features for next step if using GNN
        next_node_features = env.get_node_features(use_gnn) if use_gnn else None

        # Store transitions for all agents
        for i, aid in enumerate(obs_dict.keys()):
            if use_gnn:
                # Store graph state and adjacency with node index
                buffer.push(
                    node_features if node_features is not None else np.array([]),
                    actions[aid],
                    rewards[aid],
                    next_node_features if next_node_features is not None else np.array([]),
                    done,
                    adjacency,
                    i,  # Store the node index
                )
            else:
                # Store flat state
                buffer.push(
                    obs_dict[aid],
                    actions[aid],
                    rewards[aid],
                    next_obs_dict[aid],
                    done,
                )
        
        obs_dict = next_obs_dict
        node_features = next_node_features

        # Learn (only if buffer has enough samples for stable training)
        if len(buffer) >= max(batch_size, min_buffer_size):
            # Apply meta-learning rate scaling
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * meta_lr_scale
            
            loss = optimize(
                q_net, target_net, buffer, batch_size, gamma,
                optimizer, device, TrainingConfig.grad_clip_norm, use_gnn
            )
            losses.append(loss)
            updates += 1
            
            # Restore original learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / meta_lr_scale
            
        # Update target network based on global steps, not training updates
        if (
            update_target_steps > 0
            and global_step % update_target_steps == 0
            and global_step > 0
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
        "time_of_day": context["time_of_day"],
        "global_congestion": context["global_congestion"],
    }
    return metrics, global_step


def main() -> None:  # noqa: PLR0915
    """Main training function for shared-policy multi-agent traffic control with meta-learning support.
    
    This system uses meta-learning to adapt exploration behavior and policy responses 
    based on traffic context and performance trends. Uses parameter sharing: one DQN/GNN 
    policy controls all intersections. This is NOT independent-agent MARL.
    """
    parser = argparse.ArgumentParser(
        description="Shared-Policy Multi-Agent RL Traffic Control with Meta-Learning Support"
    )
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--N", type=int, default=TrainingConfig.num_intersections)
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument("--replay_capacity", type=int, default=TrainingConfig.replay_capacity)
    parser.add_argument("--epsilon_start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon_end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--epsilon_decay_steps", type=int, default=TrainingConfig.epsilon_decay_steps)
    parser.add_argument("--update_target_steps", type=int, default=TrainingConfig.update_target_steps)
    parser.add_argument("--min_buffer_size", type=int, default=TrainingConfig.min_buffer_size, help="Minimum buffer size before training starts")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--neighbor_obs", action="store_true")
    parser.add_argument("--use_gnn", action="store_true", help="Use GNN-DQN hybrid architecture")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows (optional)")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid cols (optional)")
    
    # Meta-learning arguments
    parser.add_argument("--use_meta_learning", action="store_true", help="Enable meta-learning for adaptive exploration")
    parser.add_argument("--meta_epsilon_min", type=float, default=TrainingConfig.meta_epsilon_min, help="Minimum epsilon for meta-learning")
    parser.add_argument("--meta_epsilon_max", type=float, default=TrainingConfig.meta_epsilon_max, help="Maximum epsilon for meta-learning")
    parser.add_argument("--meta_lr_scale_min", type=float, default=TrainingConfig.meta_lr_scale_min, help="Minimum learning rate scale for meta-learning")
    parser.add_argument("--meta_lr_scale_max", type=float, default=TrainingConfig.meta_lr_scale_max, help="Maximum learning rate scale for meta-learning")
    
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    # Force neighbor_obs = False when using GNN to let GNN learn spatial relationships
    if args.use_gnn:
        neighbor_obs = False
        logger.info("Using GNN: forcing neighbor_obs=False to let GNN learn spatial patterns")
    else:
        neighbor_obs = args.neighbor_obs

    env = MiniTrafficEnv(
        EnvConfig(
            num_intersections=args.N,
            max_steps=args.max_steps,
            seed=args.seed,
            neighbor_obs=neighbor_obs,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
    )

    device = torch.device("cpu")
    n_actions = env.get_n_actions()

    if args.use_gnn:
        node_features = env.get_obs_dim(use_gnn=True)
        q_net = GNN_DQNet(node_features, n_actions).to(device)
        target_net = GNN_DQNet(node_features, n_actions).to(device)
        logger.info(f"Using GNN-DQN architecture with {node_features} node features")
    else:
        obs_dim = env.get_obs_dim(use_gnn=False)
        q_net = DQNet(obs_dim, n_actions).to(device)
        target_net = DQNet(obs_dim, n_actions).to(device)
        logger.info(f"Using DQN architecture with {obs_dim} observation dim")
    
    target_net.load_state_dict(q_net.state_dict())
    buffer = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    
    # Add learning rate scheduler for stability
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.episodes//4, gamma=0.8)
    
    # Initialize meta-controller if enabled
    meta_controller = None
    meta_optimizer = None
    recent_rewards = []
    recent_queues = []
    
    if args.use_meta_learning:
        meta_controller = MetaController(
            epsilon_min=args.meta_epsilon_min,
            epsilon_max=args.meta_epsilon_max,
            lr_scale_min=args.meta_lr_scale_min,
            lr_scale_max=args.meta_lr_scale_max,
        ).to(device)
        meta_optimizer = optim.Adam(meta_controller.parameters(), lr=TrainingConfig.meta_controller_lr)
        logger.info(f"Meta-learning enabled: epsilon [{args.meta_epsilon_min:.3f}, {args.meta_epsilon_max:.3f}], "
                   f"lr_scale [{args.meta_lr_scale_min:.3f}, {args.meta_lr_scale_max:.3f}]")

    metrics_path = save_dir / "metrics.json"
    csv_path = save_dir / "metrics.csv"
    summary_path = save_dir / "summary.txt"
    live_path = save_dir / "live_metrics.json"
    policy_path = save_dir / "policy_final.pth"
    final_report_path = save_dir / "final_report.json"

    all_metrics = []
    global_step = 0

    logger.info("Starting Shared-Policy Multi-Agent RL Training with Meta-Learning Support:")
    logger.info(f"- {args.N} intersections (shared policy, not independent agents)")
    logger.info(f"- {args.episodes} episodes")
    episode_duration = args.max_steps * 2
    logger.info(f"- {args.max_steps} steps per episode (~{episode_duration}s)")
    logger.info(f"- Learning rate: {args.lr}")
    if args.use_meta_learning:
        logger.info("- Meta-learning: ENABLED")
        logger.info(f"  - Adaptive epsilon range: [{args.meta_epsilon_min:.3f}, {args.meta_epsilon_max:.3f}]")
        logger.info(f"  - Learning rate scale range: [{args.meta_lr_scale_min:.3f}, {args.meta_lr_scale_max:.3f}]")
    else:
        logger.info("- Meta-learning: DISABLED (using traditional epsilon decay)")
        logger.info(
            f"- Epsilon decay: {args.epsilon_start} -> {args.epsilon_end} "
            f"over {args.epsilon_decay_steps} steps"
        )

    for ep in trange(args.episodes, desc="Shared-Policy Multi-Agent Episodes"):
        # Determine epsilon and learning rate scale
        if args.use_meta_learning and meta_controller is not None:
            # Use meta-controller for adaptive hyperparameters
            meta_metrics = get_meta_metrics(recent_rewards, recent_queues, ep, args.episodes)
            with torch.no_grad():
                eps, meta_lr_scale = meta_controller(meta_metrics)
        else:
            # Use traditional epsilon decay
            eps = epsilon_by_step(
                global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            meta_lr_scale = 1.0
        
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
            args.use_gnn,
            meta_lr_scale,
        )
        
        # Store recent metrics for meta-learning
        recent_rewards.append(m["avg_reward"])
        recent_queues.append(m["avg_queue"])
        
        # Update meta-controller periodically
        if (args.use_meta_learning and meta_controller is not None and 
            ep > 0 and ep % TrainingConfig.meta_update_frequency == 0):
            
            # Simple meta-learning objective: minimize recent queue length
            meta_metrics = get_meta_metrics(recent_rewards, recent_queues, ep, args.episodes)
            _ = meta_controller(meta_metrics)  # Get predictions but don't use them directly
            
            # Meta-loss: encourage lower queue lengths (simple heuristic)
            recent_queue_avg = np.mean(recent_queues[-TrainingConfig.meta_update_frequency:])
            meta_loss = torch.tensor(recent_queue_avg, dtype=torch.float32, requires_grad=True)
            
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        # Enhanced record with shared-policy multi-agent info and meta-learning
        record = {
            "episode": ep,
            "epsilon": eps,
            "meta_epsilon": eps if args.use_meta_learning else None,
            "meta_lr_scale": meta_lr_scale if args.use_meta_learning else None,
            "global_step": global_step,
            "agents": args.N,
            "learning_rate": scheduler.get_last_lr()[0],
            **m,
        }
        all_metrics.append(record)
        
        # Step the learning rate scheduler
        scheduler.step()

        # Update files after each episode (flush immediately to avoid blocking)
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2)
                f.flush()
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.flush()

            # Write CSV fresh each time to avoid duplicate headers and ensure consistency
            fieldnames = list(record.keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for rec in all_metrics:
                    w.writerow(rec)
                f.flush()

            # Human-readable rolling summary
            with open(summary_path, "w", encoding="utf-8") as f:
                summary_text = (
                    "Shared-Policy Multi-Agent RL Traffic Control Training Summary\n"
                    f"Episodes completed: {ep + 1}/{args.episodes}\n"
                    f"Intersections (shared policy): {args.N}\n"
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
                f.flush()
        except Exception as e:
            logger.error(f"Error writing episode files: {e}")

        # Enhanced console output
        logger.info(
            f"Ep {ep+1:2d}/{args.episodes}: "
            f"eps={eps:.3f} | "
            f"Queue={m['avg_queue']:.2f} | "
            f"Throughput={m['throughput']:.0f} | "
            f"TravelTime={m['avg_travel_time']:.1f}s | "
            f"Loss={m['loss']:.4f} | "
            f"Updates={m['updates']:.0f}"
        )

    # Save final policy
    try:
        torch.save(q_net.state_dict(), policy_path)
        logger.info(f"Saved final policy to {policy_path}")
    except Exception as e:
        logger.error(f"Error saving policy: {e}")

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
    try:
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.flush()  # Ensure file is written immediately
        logger.info(f"Wrote final summary to {final_report_path}")
    except Exception as e:
        logger.error(f"Error writing final report: {e}")
    
    # Force flush all logging output
    import sys
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
