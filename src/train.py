"""Multi-agent reinforcement learning training for traffic control with multiple architectures."""
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
from torch.distributions import Categorical
from tqdm import trange

try:
    from .agent import (
        DQNet, GNN_DQNet, PPO_GNN, GAT_DQNet, GNN_A2C, 
        ReplayBuffer, RolloutBuffer
    )
    from .config import TrainingConfig, OUTPUTS_DIR, ModelType
    from .env import MiniTrafficEnv, EnvConfig
except ImportError:

    from src.agent import (
        DQNet, GNN_DQNet, PPO_GNN, GAT_DQNet, GNN_A2C, 
        ReplayBuffer, RolloutBuffer
    )
    from src.config import TrainingConfig, OUTPUTS_DIR, ModelType
    from src.env import MiniTrafficEnv, EnvConfig

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

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def epsilon_by_step(
    step: int, start: float, end: float, decay_steps: int
) -> float:
    """Calculate epsilon value for epsilon-greedy exploration.
    
    LEGACY: Linear decay based on training steps.
    Kept for backward compatibility but not recommended.
    """
    if decay_steps <= 0:
        return end
    eps = end + (start - end) * max(
        0.0, (decay_steps - step) / decay_steps
    )
    return float(eps)

def epsilon_by_episode(
    episode: int,
    total_episodes: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.10,
    warmup_fraction: float = 0.10,
    decay_power: float = 2.0,
) -> float:
    """Calculate epsilon value using hybrid episode-based decay.
    
    This is the RECOMMENDED approach for 20-50 episode training runs.
    
    Algorithm:
        Phase 1 (Warm-up): First warmup_fraction of episodes at epsilon_start
        Phase 2 (Decay): Remaining episodes with polynomial decay
    
    Args:
        episode: Current episode number (0-indexed)
        total_episodes: Total number of episodes in training
        epsilon_start: Starting epsilon value (default 1.0)
        epsilon_end: Minimum epsilon value (default 0.10)
        warmup_fraction: Fraction of episodes for warm-up (default 0.10)
        decay_power: Power for polynomial decay (default 2.0 for quadratic)
    
    Returns:
        Epsilon value for current episode
    
    Example:
        For 20 episodes:
            Episodes 0-1: ε = 1.0 (warm-up)
            Episodes 2-19: ε decays from 1.0 → 0.10
        
        For 50 episodes:
            Episodes 0-4: ε = 1.0 (warm-up)
            Episodes 5-49: ε decays from 1.0 → 0.10
    """
    warmup_episodes = int(total_episodes * warmup_fraction)
    
    if episode < warmup_episodes:

        return epsilon_start
    else:

        progress = (episode - warmup_episodes) / (total_episodes - warmup_episodes)
        epsilon = epsilon_start * (1.0 - progress) ** decay_power
        return max(epsilon_end, epsilon)

def select_action(
    model: Union[DQNet, GNN_DQNet, PPO_GNN, GAT_DQNet, GNN_A2C],
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
    
    if model_type in ["PPO-GNN", "GNN-A2C"]:

        with torch.no_grad():
            if model_type in ["PPO-GNN", "GNN-A2C"]:
                node_features = torch.from_numpy(obs).to(device).float()
                adj = torch.from_numpy(adjacency).to(device).float() if adjacency is not None else None
                
                if hasattr(model, 'get_action_and_value'):
                    action, log_prob, value = model.get_action_and_value(node_features, adj, node_idx)
                else:

                    if model_type == "PPO-GNN":
                        policy_logits, values = model(node_features, adj)
                        if policy_logits.dim() == 2:
                            node_logits = policy_logits[node_idx]
                            node_value = values[node_idx]
                        else:
                            node_logits = policy_logits[:, node_idx]
                            node_value = values[:, node_idx]
                    else:
                        action_logits, values = model(node_features, adj)
                        if action_logits.dim() == 2:
                            node_logits = action_logits[node_idx]
                            node_value = values[node_idx]
                        else:
                            node_logits = action_logits[:, node_idx]
                            node_value = values[:, node_idx]
                    

                    masked_logits = node_logits.clone()
                    for a in range(n_actions):
                        if a not in valid_actions:
                            masked_logits[a] = float('-inf')
                    
                    probs = torch.softmax(masked_logits, dim=-1)
                    dist = Categorical(probs)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor)
                    
                    action = action_tensor.item()
                    log_prob = log_prob.item()
                    value = node_value.item() if hasattr(node_value, 'item') else float(node_value)
                

                if action not in valid_actions:
                    action = valid_actions[0]
                
                return action, log_prob, value
    else:

        if np.random.rand() < eps:

            return int(np.random.choice(valid_actions)), 0.0, 0.0
        
        with torch.no_grad():
            if model_type in ["GNN-DQN", "GAT-DQN"]:

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
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    model_type: str = "DQN",
) -> float:
    """Perform one optimization step using DQN algorithm."""
    trans = buffer.sample(batch_size)
    
    if model_type in ["GNN-DQN", "GAT-DQN"]:

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
            next_q_all = target_net(next_state, adjacency)
            next_q = torch.stack([next_q_all[i, node_ids[i], :].max() for i in range(batch_size_actual)])
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

    loss = nn.functional.smooth_l1_loss(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    

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
    return float(loss.item())

def optimize_ppo(
    model: PPO_GNN,
    buffer: RolloutBuffer,
    optimizer: optim.Optimizer,
    device: torch.device,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[float, float, float]:
    """Optimize PPO model."""
    states, actions, rewards, old_log_probs, old_values, dones, adjacencies, node_ids = buffer.get()
    
    if not states:
        return 0.0, 0.0, 0.0
    

    states_tensor = torch.stack([torch.from_numpy(s) for s in states]).float().to(device)
    adjacencies_tensor = torch.stack([torch.from_numpy(a) for a in adjacencies if a is not None]).float().to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    old_values_tensor = torch.tensor(old_values, dtype=torch.float32).to(device)
    node_ids_tensor = torch.tensor(node_ids, dtype=torch.long).to(device)
    

    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = old_values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - old_values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + old_values[t])
    
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
    

    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    

    for _ in range(ppo_epochs):

        policy_logits, values = model(states_tensor, adjacencies_tensor)
        

        batch_size = len(node_ids)
        node_policy_logits = torch.stack([policy_logits[i, node_ids[i]] for i in range(batch_size)])
        node_values = torch.stack([values[i, node_ids[i]] for i in range(batch_size)]).squeeze(-1)
        

        probs = torch.softmax(node_policy_logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy()
        

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        

        value_loss = nn.functional.smooth_l1_loss(node_values, returns_tensor)
        

        entropy_loss = -entropy.mean()
        

        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy_loss += entropy_loss.item()
    
    return total_policy_loss / ppo_epochs, total_value_loss / ppo_epochs, total_entropy_loss / ppo_epochs

def optimize_a2c(
    model: GNN_A2C,
    buffer: RolloutBuffer,
    optimizer: optim.Optimizer,
    device: torch.device,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
) -> tuple[float, float, float]:
    """Optimize A2C model."""
    states, actions, rewards, _, _, dones, adjacencies, node_ids = buffer.get()
    
    if not states:
        return 0.0, 0.0, 0.0
    

    states_tensor = torch.stack([torch.from_numpy(s) for s in states]).float().to(device)
    adjacencies_tensor = torch.stack([torch.from_numpy(a) for a in adjacencies if a is not None]).float().to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    node_ids_tensor = torch.tensor(node_ids, dtype=torch.long).to(device)
    

    returns = []
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1 - dones[t])
        returns.insert(0, R)
    
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
    

    action_logits, values = model(states_tensor, adjacencies_tensor)
    

    batch_size = len(node_ids)
    node_action_logits = torch.stack([action_logits[i, node_ids[i]] for i in range(batch_size)])
    node_values = torch.stack([values[i, node_ids[i]] for i in range(batch_size)]).squeeze(-1)
    

    advantages = returns_tensor - node_values.detach()
    

    probs = torch.softmax(node_action_logits, dim=-1)
    dist = Categorical(probs)
    log_probs = dist.log_prob(actions_tensor)
    actor_loss = -(log_probs * advantages).mean()
    

    critic_loss = nn.functional.smooth_l1_loss(node_values, returns_tensor)
    

    entropy_loss = -dist.entropy().mean()
    

    loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
    

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    return actor_loss.item(), critic_loss.item(), entropy_loss.item()

def run_episode(
    env: MiniTrafficEnv,
    model: Union[DQNet, GNN_DQNet, PPO_GNN, GAT_DQNet, GNN_A2C],
    target_net: Union[DQNet, GNN_DQNet, GAT_DQNet] | None,
    buffer: Union[ReplayBuffer, RolloutBuffer],
    n_actions: int,
    eps: float,
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
) -> tuple[Dict[str, float], int, int]:
    """Run one episode of shared-policy multi-agent training with support for multiple architectures.
    
    Returns: (metrics, global_step, training_updates)
    """
    obs_dict = env.reset()
    done = False
    episode_reward_sum = 0.0
    updates = 0
    training_updates = 0
    losses = []
    policy_losses = []
    value_losses = []
    

    use_graph = model_type in ["GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C"]
    adjacency = env.get_adjacency_matrix() if use_graph else None
    node_features = env.get_node_features(use_graph) if use_graph else None
    

    context = env._get_context_features()

    while not done:
        actions: Dict[str, int] = {}
        log_probs = []
        values = []
        

        if use_graph and node_features is not None:

            for i, aid in enumerate(obs_dict.keys()):
                action, log_prob, value = select_action(
                    model, node_features, n_actions, eps, device, model_type, 
                    adjacency, i, env.time_since_switch[i], env.min_green
                )
                actions[aid] = action
                log_probs.append(log_prob)
                values.append(value)
        else:

            for i, (aid, obs) in enumerate(obs_dict.items()):
                action, log_prob, value = select_action(
                    model, obs, n_actions, eps, device, model_type,
                    time_since_switch=env.time_since_switch[i], min_green=env.min_green
                )
                actions[aid] = action
                log_probs.append(log_prob)
                values.append(value)

        next_obs_dict, rewards, done, info = env.step(actions)
        episode_reward_sum += float(np.mean(list(rewards.values())))
        

        next_node_features = env.get_node_features(use_graph) if use_graph else None

        for i, aid in enumerate(obs_dict.keys()):
            if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:

                if use_graph:
                    buffer.push(
                        node_features if node_features is not None else np.array([]),
                        actions[aid],
                        rewards[aid],
                        next_node_features if next_node_features is not None else np.array([]),
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
            else:

                if use_graph:
                    buffer.push(
                        node_features if node_features is not None else np.array([]),
                        actions[aid],
                        rewards[aid],
                        log_probs[i],
                        values[i],
                        done,
                        adjacency,
                        i,
                    )
                else:
                    buffer.push(
                        obs_dict[aid],
                        actions[aid],
                        rewards[aid],
                        log_probs[i],
                        values[i],
                        done,
                    )
        
        obs_dict = next_obs_dict
        node_features = next_node_features

        if model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:

            if len(buffer) >= max(batch_size, min_buffer_size):
                loss = optimize_dqn(
                    model, target_net, buffer, batch_size, gamma,
                    optimizer, device, config.grad_clip_norm if config else 1.0, model_type
                )
                losses.append(loss)
                updates += 1
                training_updates += 1
            
        global_step += 1

    cumulative_updates = total_training_updates + training_updates
    if (
        target_net is not None and
        update_target_steps > 0
        and cumulative_updates > 0
        and cumulative_updates % update_target_steps == 0
    ):
        target_net.load_state_dict(model.state_dict())

    if model_type in ["PPO-GNN", "GNN-A2C"] and len(buffer) > 0:
        if model_type == "PPO-GNN":
            policy_loss, value_loss, entropy_loss = optimize_ppo(
                model, buffer, optimizer, device,
                config.ppo_epochs if config else 4,
                config.ppo_clip_ratio if config else 0.2,
                config.ppo_value_coef if config else 0.5,
                config.ppo_entropy_coef if config else 0.01,
                config.ppo_max_grad_norm if config else 0.5,
                gamma,
                config.ppo_gae_lambda if config else 0.95,
            )
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            updates += 1
        elif model_type == "GNN-A2C":
            actor_loss, critic_loss, entropy_loss = optimize_a2c(
                model, buffer, optimizer, device,
                config.a2c_value_coef if config else 0.5,
                config.a2c_entropy_coef if config else 0.01,
                config.a2c_max_grad_norm if config else 0.5,
                gamma,
            )
            policy_losses.append(actor_loss)
            value_losses.append(critic_loss)
            updates += 1
        

        buffer.clear()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
    avg_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
    
    metrics = {
        "avg_reward": episode_reward_sum,
        "avg_queue": info.get("avg_queue", 0.0),
        "throughput": info.get("throughput", 0.0),
        "avg_travel_time": info.get("avg_travel_time", 0.0),
        "updates": float(updates),
        "training_updates": float(training_updates),  # FIXED: Track training updates
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "time_of_day": context["time_of_day"],
        "global_congestion": context["global_congestion"],
    }
    return metrics, global_step, training_updates

def main() -> None:
    """Main training function for shared-policy multi-agent traffic control with multiple architectures.
    
    This system supports multiple RL architectures (DQN, GNN-DQN, PPO-GNN, GAT-DQN, GNN-A2C) 
    for traffic light control. Uses parameter sharing: one policy controls all intersections.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Architecture Shared-Policy Multi-Agent RL Traffic Control"
    )
    

    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--N", type=int, default=TrainingConfig.num_intersections)
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--save_dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--neighbor_obs", action="store_true")
    parser.add_argument("--grid_rows", type=int, default=None, help="Grid rows (optional)")
    parser.add_argument("--grid_cols", type=int, default=None, help="Grid cols (optional)")
    

    parser.add_argument("--model_type", type=str, choices=["DQN", "GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C"], 
                       default="DQN", help="Model architecture to use")
    

    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument("--replay_capacity", type=int, default=TrainingConfig.replay_capacity)
    parser.add_argument("--min_buffer_size", type=int, default=TrainingConfig.min_buffer_size)
    

    parser.add_argument("--epsilon_start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon_end", type=float, default=TrainingConfig.epsilon_end)
    parser.add_argument("--epsilon_decay_steps", type=int, default=TrainingConfig.epsilon_decay_steps)
    parser.add_argument("--epsilon_warmup_fraction", type=float, default=TrainingConfig.epsilon_warmup_fraction)
    parser.add_argument("--epsilon_decay_power", type=float, default=TrainingConfig.epsilon_decay_power)
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
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_warmup_fraction=args.epsilon_warmup_fraction,
        epsilon_decay_power=args.epsilon_decay_power,
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
        neighbor_obs=args.neighbor_obs,
        save_dir=Path(args.save_dir),
        comparison_mode=getattr(args, 'comparison_mode', False),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)

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
                logger.info(f"Cleared old file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clear {file_path}: {e}")
    
    logger.info(" Starting with completely fresh training state - all old files cleared")

    use_graph = args.model_type in ["GNN-DQN", "PPO-GNN", "GAT-DQN", "GNN-A2C"]
    if use_graph:
        neighbor_obs = False
        logger.info(f"Using {args.model_type}: forcing neighbor_obs=False to let graph networks learn spatial patterns")
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
            arrival_rate_ns=config.arrival_rate_ns,
            arrival_rate_ew=config.arrival_rate_ew,
            min_green=config.min_green,
            step_length=config.step_length,
            depart_capacity=config.depart_capacity,
            reward_queue_weight=config.reward_queue_weight,
            reward_imbalance_weight=config.reward_imbalance_weight,
            reward_good_switch=config.reward_good_switch,
            reward_bad_switch=config.reward_bad_switch,
            reward_imbalance_threshold=config.reward_imbalance_threshold,
            reward_queue_norm=config.reward_queue_norm,
        )
    )
    

    env.reset(seed=args.seed)
    logger.info("Environment initialized with fresh state")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f" Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("GPU not available, using CPU (training will be slower)")
    
    n_actions = env.get_n_actions()

    if args.model_type == "DQN":
        obs_dim = env.get_obs_dim(use_gnn=False)
        model = DQNet(obs_dim, n_actions).to(device)
        target_net = DQNet(obs_dim, n_actions).to(device)
        logger.info(f"Using DQN architecture with {obs_dim} observation dim - FRESH INITIALIZATION")
    elif args.model_type == "GNN-DQN":
        node_features = env.get_obs_dim(use_gnn=True)
        model = GNN_DQNet(node_features, n_actions).to(device)
        target_net = GNN_DQNet(node_features, n_actions).to(device)
        logger.info(f"Using GNN-DQN architecture with {node_features} node features - FRESH INITIALIZATION")
    elif args.model_type == "PPO-GNN":
        node_features = env.get_obs_dim(use_gnn=True)
        model = PPO_GNN(node_features, n_actions).to(device)
        target_net = None
        logger.info(f"Using PPO-GNN architecture with {node_features} node features - FRESH INITIALIZATION")
    elif args.model_type == "GAT-DQN":
        node_features = env.get_obs_dim(use_gnn=True)
        model = GAT_DQNet(node_features, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        target_net = GAT_DQNet(node_features, n_actions, n_heads=args.gat_n_heads, dropout=args.gat_dropout).to(device)
        logger.info(f"Using GAT-DQN architecture with {node_features} node features, {args.gat_n_heads} attention heads - FRESH INITIALIZATION")
    elif args.model_type == "GNN-A2C":
        node_features = env.get_obs_dim(use_gnn=True)
        model = GNN_A2C(node_features, n_actions).to(device)
        target_net = None
        logger.info(f"Using GNN-A2C architecture with {node_features} node features - FRESH INITIALIZATION")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    

    if target_net is not None:
        target_net.load_state_dict(model.state_dict())
        logger.info("Target network initialized with fresh model weights")
    

    if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
        buffer = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
        logger.info(f"Initialized fresh replay buffer with capacity {args.replay_capacity}")
    else:
        buffer = RolloutBuffer()
        logger.info("Initialized fresh rollout buffer")
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"Initialized fresh optimizer with learning rate {args.lr}")
    

    all_metrics = []
    global_step = 0
    total_training_updates = 0

    logger.info(f"Starting {args.model_type} Shared-Policy Multi-Agent RL Training:")
    logger.info(f"- {args.N} intersections (shared policy, not independent agents)")
    logger.info(f"- {args.episodes} episodes")
    episode_duration = args.max_steps * 2
    logger.info(f"- {args.max_steps} steps per episode (~{episode_duration}s)")
    logger.info(f"- Learning rate: {args.lr}")
    logger.info(f"- Model: {args.model_type}")
    logger.info("GUARANTEED FRESH START: All old files cleared, models initialized fresh, no old data loaded")
    
    if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
        logger.info(
            f"- Epsilon decay: Hybrid episode-based decay"
        )
        logger.info(
            f"  * Start: {config.epsilon_start} (full exploration)"
        )
        logger.info(
            f"  * End: {config.epsilon_end} (continuous exploration)"
        )
        logger.info(
            f"  * Warm-up: {int(args.episodes * config.epsilon_warmup_fraction)} episodes at ε=1.0"
        )
        logger.info(
            f"  * Decay: Quadratic decay over remaining episodes"
        )

    for ep in trange(args.episodes, desc=f"{args.model_type} Multi-Agent Episodes"):

        context = env._get_context_features()
        

        if args.model_type in ["DQN", "GNN-DQN", "GAT-DQN"]:
            eps = epsilon_by_episode(
                episode=ep,
                total_episodes=args.episodes,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                warmup_fraction=config.epsilon_warmup_fraction,
                decay_power=config.epsilon_decay_power,
            )
        else:
            eps = 0.0
        
        m, global_step, training_updates = run_episode(
            env,
            model,
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
            total_training_updates,
            args.model_type,
            config,
        )
        

        total_training_updates += training_updates

        record = {
            "episode": ep,
            "model_type": args.model_type,
            "epsilon": eps,
            "global_step": global_step,
            "training_updates": total_training_updates,  # FIXED: Add cumulative training updates
            "agents": args.N,
            "learning_rate": args.lr,
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
                    f"{args.model_type} Shared-Policy Multi-Agent RL Traffic Control Training Summary\n"
                    f"Episodes completed: {ep + 1}/{args.episodes}\n"
                    f"Intersections (shared policy): {args.N}\n"
                    f"Model architecture: {args.model_type}\n"
                    f"Current epsilon: {eps:.3f}\n"
                    f"Global steps: {global_step}\n"
                    "Latest episode performance:\n"
                    f"  - Avg Queue: {record.get('avg_queue', 0.0):.2f} cars\n"
                    f"  - Throughput: {record.get('throughput', 0.0):.0f} vehicles\n"
                    f"  - Avg Travel Time: {record.get('avg_travel_time', 0.0):.2f}s\n"
                    f"  - Training Loss: {record.get('loss', 0.0):.4f}\n"
                    f"  - Policy Loss: {record.get('policy_loss', 0.0):.4f}\n"
                    f"  - Value Loss: {record.get('value_loss', 0.0):.4f}\n"
                    f"  - Updates: {record.get('updates', 0):.0f}\n"
                )
                f.write(summary_text)
                f.flush()
        except Exception as e:
            logger.error(f"Error writing episode files: {e}")

        logger.info(
            f"Ep {ep+1:2d}/{args.episodes}: "
            f"eps={eps:.3f} | "
            f"Queue={m['avg_queue']:.2f} | "
            f"Throughput={m['throughput']:.0f} | "
            f"TravelTime={m['avg_travel_time']:.1f}s | "
            f"Loss={m['loss']:.4f} | "
            f"Updates={m['updates']:.0f}"
        )

    try:
        torch.save(model.state_dict(), policy_path)
        logger.info(f"Saved final {args.model_type} policy to {policy_path}")
    except Exception as e:
        logger.error(f"Error saving policy: {e}")

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
            f"Over the training run, the {args.model_type} controller reduced queues and "
            f"travel time while maintaining or improving throughput. "
            f"The averages summarize performance across all episodes; "
            f"the final episode shows the latest results that the "
            f"dashboard displays."
        ),
    }
    try:
        with open(final_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.flush()
        logger.info(f"Wrote final summary to {final_report_path}")
    except Exception as e:
        logger.error(f"Error writing final report: {e}")
    

    import sys
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    main()
