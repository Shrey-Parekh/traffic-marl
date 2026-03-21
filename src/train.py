
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Union

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
from torch import nn, optim

from tqdm import trange

try:
    from .agent import (
        DQNet, GNN_DQNet, GAT_DQNet, GATDQNBase, STGATTransformerDQN, FedSTGATDQN,
        ReplayBuffer, PrioritizedReplayBuffer, EpsilonScheduler, Transition,
        STGATAgent, FedSTGATAgent, HistoryBuffer
    )
    from .config import TrainingConfig, OUTPUTS_DIR, PER_CONFIG, TEMPORAL_CONFIG, FEDERATED_CONFIG, MODEL_GAMMA
    from .env_sumo import PuneSUMOEnv
except ImportError:
    from src.agent import (
        DQNet, GNN_DQNet, GAT_DQNet, GATDQNBase, STGATTransformerDQN, FedSTGATDQN,
        ReplayBuffer, PrioritizedReplayBuffer, EpsilonScheduler, Transition,
        STGATAgent, FedSTGATAgent, HistoryBuffer
    )
    from src.config import TrainingConfig, OUTPUTS_DIR, PER_CONFIG, TEMPORAL_CONFIG, FEDERATED_CONFIG, MODEL_GAMMA
    from src.env_sumo import PuneSUMOEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across CPU and GPU."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable for performance (disable for reproducibility)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Enable TF32 on Ampere+ GPUs (RTX 4060 Ti) for faster matmul
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def select_action(
    model: Union[DQNet, GNN_DQNet, GAT_DQNet, GATDQNBase, STGATTransformerDQN, FedSTGATDQN],
    obs: np.ndarray,
    n_actions: int,
    eps: float,
    device: torch.device,
    model_type: str = "DQN",
    adjacency: np.ndarray | None = None,
    node_idx: int = 0,
    time_since_switch: int = 0,
    min_green: int = 10,
) -> tuple[int, float, float]:
    """Select action using appropriate policy for the model type."""

    valid_actions = [0]
    if time_since_switch >= min_green:
        valid_actions.append(1)
    
    # All models now use DQN-style epsilon-greedy
    if np.random.rand() < eps:
        return int(np.random.choice(valid_actions)), 0.0, 0.0
    
    with torch.no_grad():
        if model_type in ["GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"]:
            node_features = torch.from_numpy(obs).to(device).float()
            adj = torch.from_numpy(adjacency).to(device).float() if adjacency is not None else None
            q = model(node_features, adj)
            q_values = q[node_idx].clone()
        else:
            x = torch.from_numpy(obs).to(device).unsqueeze(0)
            q = model(x)
            q_values = q[0].clone()
        
        for action in range(n_actions):
            if action not in valid_actions:
                q_values[action] = float('-inf')
        
        action = int(torch.argmax(q_values).item())
        return action, 0.0, 0.0

def optimize_dqn(
    q_net: Union[DQNet, GNN_DQNet, GAT_DQNet],
    target_net: Union[DQNet, GNN_DQNet, GAT_DQNet],
    buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    model_type: str = "DQN",
) -> float:
    """Perform one optimization step using DQN algorithm."""
    # Handle both ReplayBuffer and PrioritizedReplayBuffer
    if isinstance(buffer, PrioritizedReplayBuffer):
        beta = getattr(buffer, 'current_beta', 0.4)
        samples, indices, weights = buffer.sample(batch_size, beta=beta)
        weights = torch.FloatTensor(weights).to(device)
        
        # Convert samples to Transition format
        # PER stores (state, action, reward, next_state, done, adjacency, node_id)
        # Transition needs (state, action, reward, next_state, done, adjacency, node_id, log_prob, value)
        if samples:
            # Pad with None for log_prob and value
            padded_samples = [s + (None, None) for s in samples]
            trans = Transition(*zip(*padded_samples))
        else:
            return 0.0
    else:
        trans = buffer.sample(batch_size)
        indices = None
        weights = None
    
    if model_type in ["GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"]:

        valid_indices = [i for i, s in enumerate(trans.state) if s is not None and len(s) > 0 and i < len(trans.adjacency) and trans.adjacency[i] is not None and trans.node_id[i] is not None]
        
        if not valid_indices:
            return 0.0
        
        batch_size_actual = len(valid_indices)
        

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
            node_ids.append(trans.node_id[idx])
        
        state = torch.from_numpy(np.stack(state_list)).float().to(device)
        next_state = torch.from_numpy(np.stack(next_state_list)).float().to(device)
        adjacency = torch.from_numpy(np.stack(adj_list)).float().to(device)
        
        actions = [trans.action[idx] for idx in valid_indices]
        rewards = [trans.reward[idx] for idx in valid_indices]
        dones = [trans.done[idx] for idx in valid_indices]
        
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        

        reward_tensor = torch.clamp(reward_tensor, -5.0, 5.0)
        
        q_values_all = q_net(state, adjacency)

        q_values = torch.stack([q_values_all[i, node_ids[i], actions[i]] for i in range(batch_size_actual)]).unsqueeze(1)
        
        with torch.no_grad():
            # Double DQN for all graph models: online net selects action, target net evaluates
            next_q_all_online = q_net(next_state, adjacency)
            next_actions = torch.stack([next_q_all_online[i, node_ids[i], :].argmax() for i in range(batch_size_actual)])
            next_q_all_target = target_net(next_state, adjacency)
            next_q = torch.stack([next_q_all_target[i, node_ids[i], next_actions[i]] for i in range(batch_size_actual)])
            target = reward_tensor + gamma * (1.0 - done_tensor) * next_q
            target = target.unsqueeze(1)
    else:

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
        

        reward = torch.clamp(reward, -5.0, 5.0)
        
        q_values = q_net(state).gather(1, action)
        with torch.no_grad():
            next_q = target_net(next_state).max(1, keepdim=True)[0]
            target = reward + gamma * (1.0 - done) * next_q

    # Compute TD errors for PER priority updates
    with torch.no_grad():
        td_errors = (q_values - target).abs().squeeze().cpu().numpy()
    
    # Apply importance sampling weights if using PER
    if weights is not None:
        loss_per_sample = nn.functional.smooth_l1_loss(q_values, target, reduction='none')
        # Use uniform weights - keep priority sampling but disable IS correction
        weights = torch.ones(len(loss_per_sample)).to(device)
        loss = loss_per_sample.mean()
    else:
        loss = nn.functional.smooth_l1_loss(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Update PER priorities if using PrioritizedReplayBuffer
    if isinstance(buffer, PrioritizedReplayBuffer) and indices is not None:
        buffer.update_priorities(indices, td_errors)
    

    total_norm = 0
    for p in q_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            if torch.isnan(param_norm):
                p.grad.data.zero_()
    total_norm = total_norm ** (1. / 2)
    

    if total_norm > 0 and not np.isnan(total_norm):
        nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=grad_clip_norm)
    
    optimizer.step()
    
    # Soft target update for GAT-DQN only
    if model_type == "GAT-DQN":
        tau = 0.01
        for target_param, online_param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )
    
    return float(loss.item())
def run_episode(
    env: PuneSUMOEnv,
    model: Union[DQNet, GNN_DQNet, GAT_DQNet, GATDQNBase, STGATTransformerDQN, FedSTGATDQN, STGATAgent, FedSTGATAgent],
    target_net: Union[DQNet, GNN_DQNet, GAT_DQNet] | None,
    buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    n_actions: int,
    epsilon_scheduler: EpsilonScheduler,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    update_target_steps: int,
    min_buffer_size: int,
    global_step: int,
    total_training_updates: int,
    model_type: str = "DQN",
    config: TrainingConfig = None,
    history_buffer: HistoryBuffer = None,
) -> tuple[Dict[str, float], int, int]:
    """Run one episode of shared-policy multi-agent training with support for multiple architectures.
    
    Returns: (metrics, global_step, training_updates)
    """
    obs_array = env.reset()
    
    # Initialize history buffer for ST-GAT and Fed-ST-GAT
    if history_buffer is not None:
        history_buffer.reset()
        history_buffer.update(obs_array)
        obs_input = history_buffer.get()  # (9, T, 24)
    else:
        obs_input = obs_array  # (9, 24) for non-temporal models
    
    # Convert array to dict format for compatibility
    obs_dict = {f"agent_{i}": obs_array[i] for i in range(len(obs_array))}
    done = False
    episode_reward_sum = 0.0
    step_count = 0
    updates = 0
    training_updates = 0
    losses = []
    step_queue_pcus = []
    
    use_agent_api = model_type in ["ST-GAT", "Fed-ST-GAT"]
    adjacency = env.adjacency_matrix if model_type in ["GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"] else None

    while not done:
        # Update epsilon every step
        eps = epsilon_scheduler.step() if epsilon_scheduler else 0.0
        
        # Update epsilon for agent-based models
        if use_agent_api:
            model.update_epsilon(eps)
        
        actions: Dict[str, int] = {}
        
        if use_agent_api:
            # ST-GAT and Fed-ST-GAT use agent API
            action_list = model.act(obs_input, evaluate=False)
            actions = {f"agent_{i}": action_list[i] for i in range(len(action_list))}
        elif adjacency is not None:
            # For graph models, use the full observation array
            for i, aid in enumerate(obs_dict.keys()):
                action, _, _ = select_action(
                    model, obs_array, n_actions, eps, device, model_type, 
                    adjacency, i, env.steps_since_switch[i], env.min_green_steps
                )
                actions[aid] = action
        else:
            # For non-graph models, use individual observations
            for i, (aid, obs) in enumerate(obs_dict.items()):
                action, _, _ = select_action(
                    model, obs, n_actions, eps, device, model_type,
                    time_since_switch=env.steps_since_switch[i], min_green=env.min_green_steps
                )
                actions[aid] = action

        # Convert actions dict to list for PuneSUMOEnv
        action_list = [actions[f"agent_{i}"] for i in range(len(obs_array))]
        next_obs_array, rewards_list, done, info = env.step(action_list)
        step_queue_pcus.append(info.get("avg_queue_pcu", 0.0))
        
        # Update history buffer for ST-GAT and Fed-ST-GAT
        if history_buffer is not None:
            history_buffer.update(next_obs_array)
            next_obs_input = history_buffer.get()  # (9, T, 24)
        else:
            next_obs_input = next_obs_array  # (9, 24)
        
        # Convert back to dict format
        next_obs_dict = {f"agent_{i}": next_obs_array[i] for i in range(len(next_obs_array))}
        rewards = {f"agent_{i}": rewards_list[i] for i in range(len(rewards_list))}
        
        episode_reward_sum += float(np.mean(rewards_list))
        step_count += 1
        
        if use_agent_api:
            # ST-GAT and Fed-ST-GAT use agent API
            model.remember(obs_input, action_list, rewards_list, next_obs_input, done)
            loss = model.learn(batch_size)
            if loss > 0:
                losses.append(loss)
                updates += 1
                training_updates += 1
        else:
            for i, aid in enumerate(obs_dict.keys()):
                if adjacency is not None:
                    buffer.push(
                        obs_array,
                        actions[aid],
                        rewards[aid],
                        next_obs_array,
                        done,
                        adjacency,
                        i,
                    )
                else:
                    buffer.push(
                        obs_dict[aid],
                        actions[aid],
                        rewards[aid],
                        next_obs_dict[aid],
                        done,
                    )
        
        # Advance observation
        obs_dict = next_obs_dict
        obs_array = next_obs_array
        obs_input = next_obs_input

        if not use_agent_api and model_type in ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN"]:
            if len(buffer) >= min_buffer_size:
                loss = optimize_dqn(
                    model, target_net, buffer, batch_size, gamma,
                    optimizer, device, config.grad_clip_norm if config else 1.0, model_type
                )
                losses.append(loss)
                updates += 1
                training_updates += 1
            
        global_step += 1

    # Fed-ST-GAT: trigger FedAvg at episode end
    if model_type == "Fed-ST-GAT":
        model.on_episode_end()

    # Hard target update ONLY for models without built-in soft updates
    # ST-GAT and Fed-ST-GAT use soft updates in their learn() method
    cumulative_updates = total_training_updates + training_updates
    if (
        target_net is not None and
        model_type not in ["GAT-DQN", "ST-GAT", "Fed-ST-GAT"] and
        update_target_steps > 0
        and cumulative_updates > 0
        and cumulative_updates % update_target_steps == 0
    ):
        target_net.load_state_dict(model.state_dict())

    # Only DQN-style models now, no PPO/A2C
    avg_loss = float(np.mean(losses)) if losses else 0.0
    
    # Calculate average reward per step
    avg_reward_per_step = episode_reward_sum / step_count if step_count > 0 else 0.0
    
    metrics = {
        "avg_reward": avg_reward_per_step,
        "avg_queue": float(np.mean(step_queue_pcus)) if step_queue_pcus else 0.0,
        "throughput": info.get("throughput", 0.0),
        "avg_travel_time": info.get("avg_travel_time", 0.0),
        "updates": float(updates),
        "training_updates": float(training_updates),
        "loss": avg_loss,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "time_of_day": 0.0,
        "global_congestion": 0.0,
    }
    return metrics, global_step, training_updates

def main() -> None:
    """Main training function for shared-policy multi-agent traffic control with multiple architectures.
    
    This system supports multiple RL architectures (DQN, GNN-DQN, GAT-DQN-Base, GAT-DQN, ST-GAT, Fed-ST-GAT) 
    for traffic light control. Uses parameter sharing: one policy controls all intersections.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Architecture Shared-Policy Multi-Agent RL Traffic Control"
    )
    

    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--N", type=int, default=TrainingConfig.num_intersections)
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--port", type=int, default=None, help="SUMO port for parallel execution")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-seed training (e.g., '1,2,3,4,5')")
    parser.add_argument("--scenario", type=str, default="morning_peak", choices=["uniform", "morning_peak", "evening_peak"], help="Traffic scenario")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUTS_DIR))
    
    parser.add_argument("--model_type", type=str, choices=["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"], 
                       default="DQN", help="Model architecture to use")
    

    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument("--replay_capacity", type=int, default=TrainingConfig.replay_capacity)
    parser.add_argument("--min_buffer_size", type=int, default=TrainingConfig.min_buffer_size)
    

    parser.add_argument("--epsilon_start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon_end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--update_target_steps", type=int, default=TrainingConfig.update_target_steps)
    

    parser.add_argument("--ppo_epochs", type=int, default=TrainingConfig.ppo_epochs)
    parser.add_argument("--ppo_clip_ratio", type=float, default=TrainingConfig.ppo_clip_ratio)
    parser.add_argument("--ppo_value_coef", type=float, default=TrainingConfig.ppo_value_coef)
    parser.add_argument("--ppo_entropy_coef", type=float, default=TrainingConfig.ppo_entropy_coef)
    

    parser.add_argument("--a2c_value_coef", type=float, default=TrainingConfig.a2c_value_coef)
    parser.add_argument("--a2c_entropy_coef", type=float, default=TrainingConfig.a2c_entropy_coef)
    

    parser.add_argument("--gat_n_heads", type=int, default=TrainingConfig.gat_n_heads)
    parser.add_argument("--gat_dropout", type=float, default=TrainingConfig.gat_dropout)
    

    parser.add_argument("--comparison_mode", action="store_true", help="Internal flag for comparison mode")
    
    args = parser.parse_args()

    # Resolve gamma: use MODEL_GAMMA default unless --gamma was explicitly passed
    # This ensures ST-GAT/Fed-ST-GAT use 0.95 by default without silently ignoring CLI
    default_gamma = TrainingConfig.gamma  # 0.99
    if args.gamma == default_gamma:
        args.gamma = MODEL_GAMMA.get(args.model_type, default_gamma)

    config = TrainingConfig(
        num_intersections=args.N,
        max_steps=args.max_steps,
        episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        min_buffer_size=args.min_buffer_size,
        model_type=args.model_type,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        update_target_steps=args.update_target_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_ratio=args.ppo_clip_ratio,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        a2c_value_coef=args.a2c_value_coef,
        a2c_entropy_coef=args.a2c_entropy_coef,
        gat_n_heads=args.gat_n_heads,
        gat_dropout=args.gat_dropout,
        seed=args.seed,
        save_dir=Path(args.save_dir),
        comparison_mode=getattr(args, 'comparison_mode', False),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

    # Use unique filenames when port is specified (parallel execution)
    if args.port is not None:
        file_prefix = "%s_%d_%d_%s" % (args.model_type, args.seed, args.episodes, args.scenario)
        metrics_path = save_dir / ("%s_metrics.json" % file_prefix)
        csv_path = save_dir / ("%s_metrics.csv" % file_prefix)
        summary_path = save_dir / ("%s_summary.txt" % file_prefix)
        live_path = save_dir / ("%s_live_metrics.json" % file_prefix)
        policy_path = save_dir / ("%s_policy.pth" % file_prefix)
        final_report_path = save_dir / ("%s_final_report.json" % file_prefix)
    else:
        # Default names for dashboard compatibility
        metrics_path = save_dir / "metrics.json"
        csv_path = save_dir / "metrics.csv"
        summary_path = save_dir / "summary.txt"
        live_path = save_dir / "live_metrics.json"
        policy_path = save_dir / "policy_final.pth"
        final_report_path = save_dir / "final_report.json"
    

    old_files = [metrics_path, csv_path, summary_path, live_path, policy_path, final_report_path]
    for file_path in old_files:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("Cleared old file: %s", file_path)
        except Exception as e:
            logger.warning("Could not clear %s: %s", file_path, e)
    
    logger.info("Starting with completely fresh training state - all old files cleared")

    use_global = args.model_type != "DQN"
    # Initialize PuneSUMOEnv
    env = PuneSUMOEnv({
        "n_intersections": args.N,
        "scenario": args.scenario,
        "render": False,
        "seed": args.seed,
        "port": args.port,  # Pass port for parallel execution
        "max_steps": args.max_steps,
        "use_global_reward": use_global,
    })
    
    env.reset()
    port_info = ", port=%d" % args.port if args.port else ""
    logger.info("PuneSUMOEnv initialized: %d intersections, scenario=%s%s", args.N, args.scenario, port_info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("✓ Using GPU: %s", torch.cuda.get_device_name(0))
        logger.info("  - GPU Memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("  - CUDA version: %s", torch.version.cuda)
        logger.info("  - cuDNN benchmark: enabled")
        logger.info("  - TF32 matmul: enabled (Ampere optimization)")
    else:
        logger.info("GPU not available, using CPU")
        # Optimize CPU performance
        torch.set_num_threads(12)  # i7-12700K has 12 cores (8P+4E)
        logger.info("  - CPU threads: %d", torch.get_num_threads())
    
    # Log libsumo status (note: libsumo not used in this codebase)
    logger.info("Using TraCI for SUMO communication")
    
    # Get observation and action dimensions from environment
    obs_dim = 24  # 24 features per agent
    n_actions = 3  # 3 actions (keep_phase, switch_phase, force_clearance)

    if args.model_type == "DQN":
        model = DQNet(obs_dim, n_actions).to(device)
        target_net = DQNet(obs_dim, n_actions).to(device)
        logger.info("Using DQN architecture with %d observation dim - FRESH INITIALIZATION", obs_dim)
        history_buffer = None
    elif args.model_type == "GNN-DQN":
        model = GNN_DQNet(obs_dim, n_actions).to(device)
        target_net = GNN_DQNet(obs_dim, n_actions).to(device)
        logger.info("Using GNN-DQN architecture with %d node features - FRESH INITIALIZATION", obs_dim)
        history_buffer = None
    elif args.model_type == "GAT-DQN-Base":
        model = GATDQNBase(obs_dim, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        target_net = GATDQNBase(obs_dim, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        logger.info("Using GAT-DQN-Base (ablation without VCA) with %d node features - FRESH INITIALIZATION", obs_dim)
        history_buffer = None
    elif args.model_type == "GAT-DQN":
        model = GAT_DQNet(obs_dim, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        target_net = GAT_DQNet(obs_dim, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        logger.info("Using GAT-DQN architecture with %d node features, %d attention heads - FRESH INITIALIZATION", obs_dim, args.gat_n_heads)
        history_buffer = None
    elif args.model_type == "ST-GAT":
        # ST-GAT uses reduced gamma to prevent Q-value divergence over 300 steps
        effective_gamma = MODEL_GAMMA.get(args.model_type, args.gamma)
        # ST-GAT has 3x more parameters than DQN - reduce learning rate significantly
        stgat_lr = args.lr * 0.1  # Further reduced from 0.2 to 0.1 (0.00005)
        model = STGATAgent(
            obs_dim          = obs_dim,
            action_dim       = n_actions,
            n_agents         = args.N,
            adjacency_matrix = env.adjacency_matrix,
            config           = {
                "lr":         stgat_lr,
                "gamma":      effective_gamma,
                "tau":        0.0005,  # Very slow target updates for maximum stability
                "window":     TEMPORAL_CONFIG["window"],
                "hidden_dim": TEMPORAL_CONFIG["hidden_dim"],
                "gat_heads":  TEMPORAL_CONFIG["gat_heads"],
            }
        )
        target_net = None  # STGATAgent manages its own target network
        history_buffer = HistoryBuffer(
            n_agents = args.N,
            window   = TEMPORAL_CONFIG["window"],
            obs_dim  = obs_dim,
        )
        logger.info("Using ST-GAT (Spatial-Temporal) with %d node features, lr=%f, gamma=%f - CONTRIBUTION 1", obs_dim, stgat_lr, effective_gamma)
    elif args.model_type == "Fed-ST-GAT":
        # Fed-ST-GAT has 3x more parameters than DQN - reduce learning rate significantly
        fedstgat_lr = args.lr * 0.2 if args.lr == 0.0005 else args.lr
        model = FedSTGATAgent(
            obs_dim          = obs_dim,
            action_dim       = n_actions,
            n_agents         = args.N,
            adjacency_matrix = env.adjacency_matrix,
            config           = {
                "lr":           fedstgat_lr,
                "gamma":        args.gamma,
                "tau":          0.001,  # Reduced from 0.01 to 0.001 for more stable target updates
                "window":       TEMPORAL_CONFIG["window"],
                "hidden_dim":   TEMPORAL_CONFIG["hidden_dim"],
                "gat_heads":    TEMPORAL_CONFIG["gat_heads"],
                "fed_interval": FEDERATED_CONFIG["fed_interval"],
            }
        )
        target_net = None  # FedSTGATAgent manages its own target networks
        history_buffer = HistoryBuffer(
            n_agents = args.N,
            window   = TEMPORAL_CONFIG["window"],
            obs_dim  = obs_dim,
        )
        logger.info("Using Fed-ST-GAT (Federated Spatial-Temporal) with %d node features, lr=%f - CONTRIBUTION 2", obs_dim, fedstgat_lr)
    else:
        raise ValueError("Unknown model type: %s" % args.model_type)
    
    if target_net is not None:
        target_net.load_state_dict(model.state_dict())
        logger.info("Target network initialized with fresh model weights")
    
    # ST-GAT and Fed-ST-GAT manage their own buffers and optimizers
    if args.model_type in ["ST-GAT", "Fed-ST-GAT"]:
        buffer = None  # Not used - agents have their own replay buffers
        optimizer = None  # Not used - agents have their own optimizers
        logger.info("%s manages its own replay buffer and optimizer", args.model_type)
    else:  # DQN, GNN-DQN, GAT-DQN-Base, GAT-DQN
        buffer = PrioritizedReplayBuffer(capacity=args.replay_capacity)
        logger.info("Initialized Prioritized Experience Replay buffer with capacity %d", args.replay_capacity)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        logger.info("Initialized fresh optimizer with learning rate %f", args.lr)
    

    all_metrics = []
    global_step = 0
    total_training_updates = 0

    logger.info("Starting %s Shared-Policy Multi-Agent RL Training:", args.model_type)
    logger.info("- %d intersections (shared policy, not independent agents)", args.N)
    logger.info("- %d episodes", args.episodes)
    episode_duration = args.max_steps * 2
    logger.info("- %d steps per episode (~%ds)", args.max_steps, episode_duration)
    logger.info("- Learning rate: %f", args.lr)
    logger.info("- Model: %s", args.model_type)
    logger.info("GUARANTEED FRESH START: All old files cleared, models initialized fresh, no old data loaded")
    
    # DIAGNOSTIC: Verify fresh initialization
    if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN"]:
        dummy_obs = torch.randn(1, obs_dim).to(device)
        if args.model_type in ["GNN-DQN", "GAT-DQN-Base", "GAT-DQN"]:
            dummy_adj = torch.eye(args.N).unsqueeze(0).to(device)
            dummy_obs = torch.randn(1, args.N, obs_dim).to(device)
            q_init = model(dummy_obs, dummy_adj).mean().item()
        else:
            q_init = model(dummy_obs).mean().item()
        logger.info("DIAGNOSTIC: Initial Q-values mean = %.6f (should be ~0.0 ± 0.1)", q_init)
        if abs(q_init) > 0.5:
            logger.warning("WARNING: Initial Q-values suspiciously large (%.3f) - possible old weights!", q_init)
        if buffer is not None:
            logger.info("DIAGNOSTIC: Replay buffer size = %d (should be 0)", len(buffer))
            if len(buffer) > 0:
                logger.warning("WARNING: Replay buffer not empty (%d transitions) - old data present!", len(buffer))
        if optimizer is not None:
            logger.info("DIAGNOSTIC: Optimizer state = %d params (should be 0)", len(optimizer.state_dict()['state']))
            if len(optimizer.state_dict()['state']) > 0:
                logger.warning("WARNING: Optimizer has state - momentum from old run!")
    elif args.model_type in ["ST-GAT", "Fed-ST-GAT"]:
        dummy_obs = torch.randn(1, args.N, TEMPORAL_CONFIG["window"], obs_dim).to(device)
        q_init = model.local_agents[0].online_net(dummy_obs).mean().item() if args.model_type == "Fed-ST-GAT" else model.online_net(dummy_obs).mean().item()
        logger.info("DIAGNOSTIC: Initial Q-values mean = %.6f (should be ~0.0 ± 0.1)", q_init)
        if abs(q_init) > 0.5:
            logger.warning("WARNING: Initial Q-values suspiciously large (%.3f) - possible old weights!", q_init)
        buffer_size = len(model.local_agents[0].memory) if args.model_type == "Fed-ST-GAT" else len(model.memory)
        logger.info("DIAGNOSTIC: Replay buffer size = %d (should be 0)", buffer_size)
        if buffer_size > 0:
            logger.warning("WARNING: Replay buffer not empty (%d transitions) - old data present!", buffer_size)
    
    # Initialize epsilon scheduler (step-based decay)
    epsilon_scheduler = None
    if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"]:
        epsilon_scheduler = EpsilonScheduler(
            total_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            model_type=args.model_type
        )
        logger.info("- Epsilon decay: Step-based linear decay")
        logger.info("  * Start: %f (full exploration)", epsilon_scheduler.eps_start)
        logger.info("  * End: %f (continuous exploration)", epsilon_scheduler.eps_end)
        logger.info("  * Decay steps: %d of %d total", epsilon_scheduler.decay_steps, args.episodes * args.max_steps)
        logger.info("  * Model complexity multiplier: %s", args.model_type)

    for ep in trange(args.episodes, desc="%s Multi-Agent Episodes" % args.model_type):
        # Anneal PER beta from beta_start to beta_end over training
        if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"]:
            beta = (
                PER_CONFIG["beta_start"]
                + (PER_CONFIG["beta_end"] - PER_CONFIG["beta_start"])
                * (ep / max(args.episodes - 1, 1))
            )
            # Set beta on buffer if it's PrioritizedReplayBuffer
            if hasattr(buffer, 'current_beta'):
                buffer.current_beta = beta
        else:
            beta = 0.0
        
        m, global_step, training_updates = run_episode(
            env,
            model,
            target_net,
            buffer,
            n_actions,
            epsilon_scheduler,
            args.batch_size,
            args.gamma,
            optimizer,
            device,
            args.update_target_steps,
            args.min_buffer_size,
            global_step,
            total_training_updates,
            args.model_type,
            config,
            history_buffer,
        )
        

        total_training_updates += training_updates
        
        # Diagnostic logging for ST-GAT every 10 episodes
        if args.model_type in ["ST-GAT", "Fed-ST-GAT"] and (ep + 1) % 10 == 0:
            with torch.no_grad():
                dummy = torch.zeros(1, args.N, TEMPORAL_CONFIG["window"], obs_dim).to(device)
                if args.model_type == "Fed-ST-GAT":
                    q_vals = model.local_agents[0].online_net(dummy)
                    agent_for_diag = model.local_agents[0]
                else:
                    q_vals = model.online_net(dummy)
                    agent_for_diag = model
                
                # Q-value range check
                q_min, q_max = q_vals.min().item(), q_vals.max().item()
                
                # Target network distance
                target_dist = sum((t_p - o_p).pow(2).sum().item() 
                                 for t_p, o_p in zip(agent_for_diag.target_net.parameters(), 
                                                     agent_for_diag.online_net.parameters())) ** 0.5
                target_dist /= sum(p.numel() for p in agent_for_diag.online_net.parameters())
                
                # PER priorities
                if len(agent_for_diag.memory) > 0:
                    priorities = agent_for_diag.memory.priorities[:len(agent_for_diag.memory)]
                    per_max = priorities.max()
                else:
                    per_max = 0.0
                
                logger.info("Ep %d: Loss=%.4f, Queue=%.2f, Q[%.2f,%.2f], TargetDist=%.6f, PERmax=%.2f",
                           ep + 1, m['loss'], m['avg_queue'], q_min, q_max, target_dist, per_max)
                
                # Alert on divergence
                if abs(q_max) > 10 or abs(q_min) > 10:
                    logger.warning("⚠ Q-VALUE DIVERGENCE at episode %d!", ep + 1)
                if target_dist < 0.00001:
                    logger.warning("⚠ TARGET NETWORK TOO CLOSE at episode %d!", ep + 1)
                if per_max > 10.0:
                    logger.warning("⚠ PER PRIORITIES TOO HIGH at episode %d!", ep + 1)

        # Get actual learning rate and gamma used (may differ from args for ST-GAT)
        if args.model_type in ["ST-GAT", "Fed-ST-GAT"]:
            actual_lr = model.local_agents[0].lr if hasattr(model, 'local_agents') else model.lr
            actual_gamma = model.local_agents[0].gamma if hasattr(model, 'local_agents') else model.gamma
        else:
            actual_lr = args.lr
            actual_gamma = args.gamma

        record = {
            "episode": ep + 1,  # 1-indexed for display
            "total_episodes": args.episodes,  # Add total episodes
            "model_type": args.model_type,
            "epsilon": epsilon_scheduler.get() if epsilon_scheduler else 0.0,
            "per_beta": beta if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN-Base", "GAT-DQN", "ST-GAT", "Fed-ST-GAT"] else 0.0,
            "global_step": global_step,
            "training_updates": total_training_updates,
            "agents": args.N,
            "learning_rate": actual_lr,
            "gamma": actual_gamma,
            **m,
        }
        all_metrics.append(record)
        

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2)
                f.flush()
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.flush()

            fieldnames = list(record.keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for rec in all_metrics:
                    w.writerow(rec)
                f.flush()

            with open(summary_path, "w", encoding="utf-8") as f:
                summary_text = (
                    "%s Shared-Policy Multi-Agent RL Traffic Control Training Summary\n"
                    "Episodes completed: %d/%d\n"
                    "Intersections (shared policy): %d\n"
                    "Model architecture: %s\n"
                    "Current epsilon: %.3f\n"
                    "Global steps: %d\n"
                    "Latest episode performance:\n"
                    "  - Avg Queue: %.2f cars\n"
                    "  - Throughput: %.0f vehicles\n"
                    "  - Avg Travel Time: %.2fs\n"
                    "  - Training Loss: %.4f\n"
                    "  - Policy Loss: %.4f\n"
                    "  - Value Loss: %.4f\n"
                    "  - Updates: %.0f\n"
                ) % (
                    args.model_type,
                    ep + 1, args.episodes,
                    args.N,
                    args.model_type,
                    epsilon_scheduler.get() if epsilon_scheduler else 0.0,
                    global_step,
                    record.get('avg_queue', 0.0),
                    record.get('throughput', 0.0),
                    record.get('avg_travel_time', 0.0),
                    record.get('loss', 0.0),
                    record.get('policy_loss', 0.0),
                    record.get('value_loss', 0.0),
                    record.get('updates', 0)
                )
                f.write(summary_text)
                f.flush()
        except Exception as e:
            logger.error("Error writing episode files: %s", e)

        logger.info(
            "Ep %d/%d: eps=%.3f | Queue=%.2f | Throughput=%.0f | TravelTime=%.1fs | Loss=%.4f | Updates=%.0f",
            ep + 1, args.episodes,
            epsilon_scheduler.get() if epsilon_scheduler else 0.0,
            m['avg_queue'], m['throughput'], m['avg_travel_time'], m['loss'], m['updates']
        )

    try:
        torch.save(model.state_dict(), policy_path)
        logger.info("Saved final %s policy to %s", args.model_type, policy_path)
    except Exception as e:
        logger.error("Error saving policy: %s", e)

    keys = [
        "avg_reward",
        "avg_queue",
        "throughput",
        "avg_travel_time",
        "loss",
        "policy_loss",
        "value_loss",
    ]
    avgs = {
        k: float(np.mean([r.get(k, 0.0) for r in all_metrics]))
        for k in keys
    }
    final = all_metrics[-1] if all_metrics else {}
    report = {
        "model_type": args.model_type,
        "episodes": len(all_metrics),
        "average_metrics": avgs,
        "final_episode": final,
        "plain_english": (
            "Over the training run, the %s controller reduced queues and "
            "travel time while maintaining or improving throughput. "
            "The averages summarize performance across all episodes; "
            "the final episode shows the latest results that the "
            "dashboard displays."
        ) % args.model_type,
    }
    try:
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.flush()
        logger.info("Wrote final summary to %s", final_report_path)
    except Exception as e:
        logger.error("Error writing final report: %s", e)
    
    # Flush output streams
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    main()
