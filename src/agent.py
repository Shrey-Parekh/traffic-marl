from __future__ import annotations

from collections import namedtuple, deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

try:
    from torch_geometric.nn import TransformerConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    TransformerConv = None

try:
    from .config import EPSILON_CONFIG, PER_CONFIG
except ImportError:
    from config import EPSILON_CONFIG, PER_CONFIG

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "adjacency", "node_id", "log_prob", "value"),
)


class EpsilonScheduler:
    """
    Step-based linear epsilon decay.
    
    Decay is computed over total training STEPS, not episodes.
    This ensures consistent exploration regardless of:
    - Episode length changes (max_steps parameter)
    - Number of episodes
    - Step length changes in SUMO config
    
    Model complexity multiplier stretches decay window for graph models
    that require more environment steps to learn spatial coordination.
    
    Usage:
        scheduler = EpsilonScheduler(total_episodes, max_steps, model_type)
        # Inside step loop:
        epsilon = scheduler.step()
        agent.epsilon = epsilon
    """
    
    def __init__(
        self,
        total_episodes: int,
        max_steps_per_episode: int,
        model_type: str = "DQN",
    ):
        cfg = EPSILON_CONFIG
        complexity = cfg["model_complexity"].get(model_type, 1.0)
        
        self.eps_start = cfg["start"]
        self.eps_end   = cfg["end"]
        
        total_steps = total_episodes * max_steps_per_episode
        
        # Decay window stretched by model complexity
        # Capped at 95% of total steps so decay always completes
        self.decay_steps = min(
            int(total_steps * cfg["decay_fraction"] * complexity),
            int(total_steps * 0.95)
        )
        
        self.current_step = 0
        self.current_eps  = cfg["start"]
    
    def step(self) -> float:
        """
        Call once per environment step inside the training loop.
        Returns current epsilon value.
        """
        if self.current_step >= self.decay_steps:
            self.current_eps = self.eps_end
        else:
            progress = self.current_step / self.decay_steps
            self.current_eps = (
                self.eps_start
                - (self.eps_start - self.eps_end) * progress
            )
        self.current_step += 1
        return float(self.current_eps)
    
    def get(self) -> float:
        """Returns current epsilon without advancing the step counter."""
        return float(self.current_eps)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.buffer: deque = deque(maxlen=capacity)
        random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        adjacency: np.ndarray | None = None,
        node_id: int | None = None,
        log_prob: float | None = None,
        value: float | None = None,
    ) -> None:
        self.buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                np.asarray(adjacency, dtype=np.float32) if adjacency is not None else None,
                int(node_id) if node_id is not None else None,
                float(log_prob) if log_prob is not None else None,
                float(value) if value is not None else None,
            )
        )

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions proportional to their TD error magnitude.
    High TD error = transition was surprising = more learning signal.
    
    Importance sampling weights correct for the introduced sampling bias.
    Beta anneals from beta_start to 1.0 over training to fully correct
    bias by the end of training.
    
    Reference: Schaul et al. 2016, empirically validated for traffic
    signal DQN to produce faster convergence vs uniform replay.
    
    Usage:
        buffer = PrioritizedReplayBuffer(capacity=10000)
        buffer.add(state, action, reward, next_state, done)
        samples, indices, weights = buffer.sample(batch_size, beta)
        # After computing loss:
        buffer.update_priorities(indices, td_errors)
    """
    
    def __init__(self, capacity: int, alpha: float = None):
        cfg = PER_CONFIG
        self.capacity     = capacity
        self.alpha        = alpha if alpha is not None else cfg["alpha"]
        self.eps          = cfg["epsilon"]
        self.buffer       = []
        self.priorities   = np.zeros(capacity, dtype=np.float32)
        self.pos          = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done, adjacency=None, node_id=None):
        """New transitions get max priority — sampled at least once."""
        idx = self.pos % self.capacity
        transition = (state, action, reward, next_state, done, adjacency, node_id)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[idx] = transition
        self.priorities[idx] = self.max_priority
        self.pos += 1
    
    def push(self, state, action, reward, next_state, done, adjacency=None, node_id=None, log_prob=None, value=None):
        """Alias for add() to maintain compatibility with ReplayBuffer interface."""
        self.add(state, action, reward, next_state, done, adjacency, node_id)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch_size transitions weighted by priority.
        Returns: (samples, indices, importance_weights)
        """
        n = len(self.buffer)
        if n == 0:
            return [], [], np.array([])
        
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        batch_size = min(batch_size, n)
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities after each training step.
        Call with absolute TD errors from loss computation.
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(float(error)) + self.eps)
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    """Buffer for on-policy algorithms."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.adjacencies = []
        self.node_ids = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        adjacency: np.ndarray | None = None,
        node_id: int | None = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.adjacencies.append(adjacency)
        self.node_ids.append(node_id)
    
    def get(self):
        return (
            self.states, self.actions, self.rewards, self.log_probs,
            self.values, self.dones, self.adjacencies, self.node_ids,
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.adjacencies.clear()
        self.node_ids.clear()
    
    def __len__(self):
        return len(self.states)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        identity = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_with_loops = adjacency + identity
        degree = adjacency_with_loops.sum(dim=2, keepdim=True)
        degree_inv = torch.pow(degree, -1.0)
        degree_inv[degree_inv == float('inf')] = 0.0
        normalized_adj = degree_inv * adjacency_with_loops
        message = torch.bmm(normalized_adj, node_features)
        output = self.linear(message)
        return output

class VehicleClassAttention(nn.Module):
    """
    Novel attention mechanism for Indian mixed traffic vehicle classes.
    
    Learns to weight different vehicle classes (two-wheelers, autos, cars, pedestrians)
    based on their importance for traffic signal control decisions.
    """
    
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.ns_attention = nn.Linear(n_classes, n_classes)
        self.ew_attention = nn.Linear(n_classes, n_classes)
        self.context_proj = nn.Linear(n_classes * 2, 2)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.ns_attention.weight)
        nn.init.xavier_uniform_(self.ew_attention.weight)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.constant_(self.ns_attention.bias, 0.0)
        nn.init.constant_(self.ew_attention.bias, 0.0)
        nn.init.constant_(self.context_proj.bias, 0.0)
    
    def forward(self, vehicle_class_features: torch.Tensor) -> torch.Tensor:
        ns_classes = vehicle_class_features[:, :self.n_classes]
        ew_classes = vehicle_class_features[:, self.n_classes:]
        ns_attn = F.softmax(self.ns_attention(ns_classes), dim=-1)
        ew_attn = F.softmax(self.ew_attention(ew_classes), dim=-1)
        ns_context = (ns_attn * ns_classes).sum(dim=-1, keepdim=True)
        ew_context = (ew_attn * ew_classes).sum(dim=-1, keepdim=True)
        context = torch.cat([ns_context, ew_context], dim=-1)
        return context


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for spatial attention in traffic networks."""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=out_features, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        identity = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_with_loops = adjacency + identity
        attn_mask = (adjacency_with_loops == 0)
        queries = self.W_q(node_features)
        keys = self.W_k(node_features)
        values = self.W_v(node_features)
        attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_mask_expanded = attn_mask_expanded.reshape(batch_size * self.n_heads, num_nodes, num_nodes)
        queries = queries.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        queries = queries.contiguous().view(batch_size, num_nodes, -1)
        keys = keys.contiguous().view(batch_size, num_nodes, -1)
        values = values.contiguous().view(batch_size, num_nodes, -1)
        attended_features, _ = self.attention(queries, keys, values, attn_mask=attn_mask_expanded)
        output = self.out_proj(attended_features)
        output = self.dropout(output)
        return output


class DQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GNN_DQNet(nn.Module):
    """Graph Neural Network DQN for shared-policy multi-agent traffic control."""
    def __init__(self, node_features: int, n_actions: int, hidden_gcn: int = 64,
                 hidden_dqn: int = 128, num_gcn_layers: int = 2):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gcn = hidden_gcn
        self.hidden_dqn = hidden_dqn
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(node_features, hidden_gcn))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_gcn, hidden_gcn))
        self.dqn_head = nn.Sequential(
            nn.Linear(hidden_gcn, hidden_dqn), nn.ReLU(),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(),
            nn.Linear(hidden_dqn, n_actions),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adjacency)
            x = F.relu(x)
        x = x.view(batch_size * num_nodes, self.hidden_gcn)
        q_values = self.dqn_head(x)
        q_values = q_values.view(batch_size, num_nodes, self.n_actions)
        if squeeze_output:
            q_values = q_values.squeeze(0)
        return q_values


class GAT_DQNet(nn.Module):
    """Graph Attention Network DQN with VehicleClassAttention for Indian mixed traffic."""
    
    def __init__(self, node_features: int, n_actions: int, hidden_gat: int = 64,
                 hidden_dqn: int = 128, num_gat_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gat = hidden_gat
        self.hidden_dqn = hidden_dqn
        self.vehicle_attention = VehicleClassAttention(n_classes=4)
        gat_input_dim = node_features - 8 + 2
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttentionLayer(gat_input_dim, hidden_gat, n_heads, dropout))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GraphAttentionLayer(hidden_gat, hidden_gat, n_heads, dropout))
        self.dqn_head = nn.Sequential(
            nn.Linear(hidden_gat, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, n_actions),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier uniform initialization with gain=1.4 for attention layers.
        Default PyTorch init produces near-zero attention weights.
        With random early actions all neighbors look similar, producing
        zero gradient signal to differentiate neighbors.
        Higher gain ensures attention weights start differentiated enough
        to receive meaningful gradients from early exploration."""
        for module in self.modules():
            if hasattr(module, 'attention'):
                # MultiheadAttention parameters
                for name, param in module.named_parameters():
                    if 'in_proj' in name or 'out_proj' in name:
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param, gain=1.4)
                        elif 'bias' in name and param is not None:
                            nn.init.zeros_(param)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        vehicle_class_features = node_features[:, :, 6:14].reshape(batch_size * num_nodes, 8)
        vehicle_context = self.vehicle_attention(vehicle_class_features)
        vehicle_context = vehicle_context.view(batch_size, num_nodes, 2)
        # Concatenate: [0-5: basic, vehicle_context(2), 14: scenario, 15-21: neighbor+mask]
        x = torch.cat([node_features[:, :, :6], vehicle_context, node_features[:, :, 14:]], dim=-1)
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adjacency)
            x = F.relu(x)
        x = x.view(batch_size * num_nodes, self.hidden_gat)
        q_values = self.dqn_head(x)
        q_values = q_values.view(batch_size, num_nodes, self.n_actions)
        if squeeze_output:
            q_values = q_values.squeeze(0)
        return q_values


class GATDQNBase(nn.Module):
    """
    Ablation model — GAT without vehicle class attention.
    Critical ablation to prove VehicleClassAttention specifically contributes.
    """
    
    def __init__(self, node_features: int, n_actions: int, hidden_gat: int = 64,
                 hidden_dqn: int = 128, num_gat_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gat = hidden_gat
        self.hidden_dqn = hidden_dqn
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttentionLayer(node_features, hidden_gat, n_heads, dropout))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GraphAttentionLayer(hidden_gat, hidden_gat, n_heads, dropout))
        self.dqn_head = nn.Sequential(
            nn.Linear(hidden_gat, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, n_actions),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        x = node_features
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adjacency)
            x = F.relu(x)
        x = x.view(batch_size * num_nodes, self.hidden_gat)
        q_values = self.dqn_head(x)
        q_values = q_values.view(batch_size, num_nodes, self.n_actions)
        if squeeze_output:
            q_values = q_values.squeeze(0)
        return q_values


class PositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for grid-based intersections."""
    
    def __init__(self, d_model: int, max_grid_size: int = 10):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_grid_size, max_grid_size, d_model)
        for row in range(max_grid_size):
            for col in range(max_grid_size):
                for i in range(0, d_model, 4):
                    pe[row, col, i] = math.sin(row / (10000 ** (i / d_model)))
                    if i + 1 < d_model:
                        pe[row, col, i + 1] = math.cos(row / (10000 ** (i / d_model)))
                    if i + 2 < d_model:
                        pe[row, col, i + 2] = math.sin(col / (10000 ** ((i + 2) / d_model)))
                    if i + 3 < d_model:
                        pe[row, col, i + 3] = math.cos(col / (10000 ** ((i + 2) / d_model)))
        self.register_buffer('pe', pe)
    
    def forward(self, num_nodes: int, grid_size: int = 3) -> torch.Tensor:
        encodings = []
        for i in range(num_nodes):
            row = i // grid_size
            col = i % grid_size
            encodings.append(self.pe[row, col])
        return torch.stack(encodings)


class STGATTransformerDQN(nn.Module):
    """
    Spatial-Temporal Graph Transformer DQN with VehicleClassAttention.
    
    CONTRIBUTION 1: Algorithmic innovation combining:
    - Spatial: Graph Transformer with full self-attention across all intersections
    - Temporal: GRU processing last T=5 observation steps to capture queue dynamics
    - VehicleClassAttention: Explicit modeling of Indian mixed traffic
    """
    
    def __init__(self, node_features: int, n_actions: int, hidden_spatial: int = 64,
                 hidden_temporal: int = 32, hidden_dqn: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, pos_enc_dim: int = 16, history_length: int = 5):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_spatial = hidden_spatial
        self.hidden_temporal = hidden_temporal
        self.history_length = history_length
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for STGATTransformerDQN")
        
        self.vehicle_attention = VehicleClassAttention(n_classes=4)
        self.pos_encoding = PositionalEncoding(pos_enc_dim)
        spatial_input_dim = node_features - 8 + 2 + pos_enc_dim
        
        self.spatial_transformer = TransformerConv(
            in_channels=spatial_input_dim, out_channels=hidden_spatial,
            heads=n_heads, dropout=dropout, concat=False,
        )
        
        self.temporal_gru = nn.GRU(
            input_size=node_features, hidden_size=hidden_temporal,
            num_layers=1, batch_first=True,
        )
        
        combined_dim = hidden_spatial + hidden_temporal
        self.dqn_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dqn, n_actions),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier uniform initialization with gain=1.4 for attention layers."""
        for module in self.modules():
            if isinstance(module, TransformerConv):
                # TransformerConv attention parameters
                for name, param in module.named_parameters():
                    if 'att' in name or 'lin' in name:
                        if param.dim() >= 2:
                            nn.init.xavier_uniform_(param, gain=1.4)
                        elif param.dim() == 1:
                            nn.init.zeros_(param)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor,
                obs_history: torch.Tensor = None) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            if obs_history is not None:
                obs_history = obs_history.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_nodes, _ = node_features.shape
        
        # Spatial Module
        vehicle_class_features = node_features[:, :, 6:14].reshape(batch_size * num_nodes, 8)
        vehicle_context = self.vehicle_attention(vehicle_class_features)
        vehicle_context = vehicle_context.view(batch_size, num_nodes, 2)
        pos_enc = self.pos_encoding(num_nodes).unsqueeze(0).expand(batch_size, -1, -1).to(node_features.device)
        spatial_input = torch.cat([
            node_features[:, :, :6], vehicle_context, node_features[:, :, 14:15], pos_enc
        ], dim=-1)
        
        edge_indices = []
        for b in range(batch_size):
            edge_index = adjacency[b].nonzero(as_tuple=False).t()
            edge_indices.append(edge_index)
        
        spatial_outputs = []
        for b in range(batch_size):
            spatial_out = self.spatial_transformer(spatial_input[b], edge_indices[b])
            spatial_outputs.append(spatial_out)
        spatial_output = torch.stack(spatial_outputs)
        
        # Temporal Module
        if obs_history is not None and obs_history.shape[2] > 0:
            obs_history_flat = obs_history.view(batch_size * num_nodes, self.history_length, self.node_features)
            _, temporal_hidden = self.temporal_gru(obs_history_flat)
            temporal_output = temporal_hidden.squeeze(0).view(batch_size, num_nodes, self.hidden_temporal)
        else:
            temporal_output = torch.zeros(batch_size, num_nodes, self.hidden_temporal, device=node_features.device)
        
        # Combine
        combined = torch.cat([spatial_output, temporal_output], dim=-1)
        combined_flat = combined.view(batch_size * num_nodes, -1)
        q_values = self.dqn_head(combined_flat)
        q_values = q_values.view(batch_size, num_nodes, self.n_actions)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
        return q_values


class FederatedCoordinator:
    """
    Federated learning coordinator implementing FedAvg aggregation.
    
    CONTRIBUTION 2: Systems/Deployment innovation for edge-based traffic control.
    Implements distributed training where each intersection trains locally and shares
    only model gradients — not raw vehicle data. This satisfies edge privacy requirements.
    
    No raw sensor data is transmitted, only model parameter updates.
    """
    
    def __init__(self, global_model: nn.Module, fed_round_interval: int = 50, track_communication: bool = True):
        self.global_model = global_model
        self.fed_round_interval = fed_round_interval
        self.track_communication = track_communication
        self.communication_cost_bytes = 0.0
        self.fed_rounds_completed = 0
    
    def aggregate(self, local_models: list[nn.Module]) -> None:
        """FedAvg aggregation: average all local model parameters."""
        if not local_models:
            return
        
        aggregated = {}
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            local_params = torch.stack([
                model.state_dict()[key].float().to(global_state[key].device)
                for model in local_models
            ])
            aggregated[key] = local_params.mean(dim=0)
            
            if self.track_communication:
                param_bytes = aggregated[key].numel() * aggregated[key].element_size()
                self.communication_cost_bytes += param_bytes * len(local_models)
        
        self.global_model.load_state_dict(aggregated)
        self.fed_rounds_completed += 1
    
    def broadcast(self, local_models: list[nn.Module]) -> None:
        """Broadcast global model parameters to all local models."""
        global_state = self.global_model.state_dict()
        for model in local_models:
            model.load_state_dict(global_state)
    
    def get_metrics(self) -> dict:
        return {
            "fed_rounds_completed": self.fed_rounds_completed,
            "communication_cost_bytes": self.communication_cost_bytes,
        }

class FedSTGATDQN(nn.Module):
    """
    Federated Spatial-Temporal Graph Transformer DQN.
    
    CONTRIBUTION 2: Wraps STGATTransformerDQN with federated gradient aggregation.
    Each intersection trains locally, coordinator aggregates every fed_round_interval steps.
    """
    
    def __init__(self, node_features: int, n_actions: int, hidden_spatial: int = 64,
                 hidden_temporal: int = 32, hidden_dqn: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, pos_enc_dim: int = 16, history_length: int = 5,
                 fed_round_interval: int = 50):
        super().__init__()
        self.base_model = STGATTransformerDQN(
            node_features=node_features, n_actions=n_actions,
            hidden_spatial=hidden_spatial, hidden_temporal=hidden_temporal,
            hidden_dqn=hidden_dqn, n_heads=n_heads, dropout=dropout,
            pos_enc_dim=pos_enc_dim, history_length=history_length,
        )
        self.fed_round_interval = fed_round_interval
        self.node_features = node_features
        self.n_actions = n_actions
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor,
                obs_history: torch.Tensor = None) -> torch.Tensor:
        return self.base_model(node_features, adjacency, obs_history)
    
    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.base_model.load_state_dict(*args, **kwargs)
