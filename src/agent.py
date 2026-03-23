from __future__ import annotations

from collections import namedtuple, deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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


class HistoryBuffer:
   
    
    def __init__(self, n_agents: int, window: int, obs_dim: int):
        self.n_agents = n_agents
        self.window   = window
        self.obs_dim  = obs_dim
        self.reset()
    
    def reset(self):
        """Clear history at episode start."""
        self.buffer = np.zeros(
            (self.n_agents, self.window, self.obs_dim),
            dtype=np.float32
        )
    
    def update(self, obs: np.ndarray):
        """
        Add new observation to history.
        obs shape: (n_agents, obs_dim)
        Shifts buffer left, adds new obs at end.
        """
        # Roll along time axis: oldest dropped, newest added
        self.buffer = np.roll(self.buffer, shift=-1, axis=1)
        self.buffer[:, -1, :] = obs   # most recent timestep at index -1
    
    def get(self) -> np.ndarray:
        """
        Returns history of shape (n_agents, T, obs_dim).
        Ready to pass directly to STGATAgent.act() or STGATAgent.learn().
        """
        return self.buffer.copy()


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
        # Cap at 1.0 so complex models can use the full training budget
        self.decay_steps = min(
            int(total_steps * cfg["decay_fraction"] * complexity),
            total_steps
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
    
    def sample_uniform(self, batch_size: int):
        """
        Sample batch_size transitions uniformly (no prioritization).
        Returns: (samples, indices, uniform_weights)
        """
        n = len(self.buffer)
        if n == 0:
            return [], [], np.array([])
        
        batch_size = min(batch_size, n)
        indices = np.random.choice(n, batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities after each training step.
        Call with absolute TD errors from loss computation.
        Priorities are capped to prevent unbounded growth.
        """
        for idx, error in zip(indices, td_errors):
            # Cap priorities between 0.01 and 5.0 to prevent outlier dominance
            priority = np.clip(abs(float(error)) + self.eps, 0.01, 5.0)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return len(self.buffer)


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


class VehicleClassAttentionSTGAT(nn.Module):

    def __init__(self, n_classes: int = 8, embed_dim: int = 16):
        super().__init__()
        # PCU prior weights — initialize attention toward car-class (highest PCU)
        # two_wheeler(NS,EW)=0.5, auto(NS,EW)=0.75, car(NS,EW)=1.0, ped=0.0
        self.pcu_prior = nn.Parameter(
            torch.tensor([0.5, 0.75, 1.0, 3.0, 0.5, 0.75, 1.0, 3.0],
                         dtype=torch.float32),
            requires_grad=True   # learned on top of PCU prior
        )
        self.attention = nn.Sequential(
            nn.Linear(n_classes, n_classes),
            nn.Tanh(),
            nn.Linear(n_classes, n_classes),
            nn.Softmax(dim=-1),
        )
        self.embed = nn.Linear(n_classes, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)  # Normalize output for stability
    
    def forward(self, class_features: torch.Tensor) -> torch.Tensor:
        """
        class_features: (..., 8) — vehicle class counts for NS and EW (pre-normalized)
        Returns: (..., 16) — PCU-weighted class embedding
        """
        # Input is pre-normalized in STGATNetwork.forward() — no division needed here
        weighted = class_features * torch.sigmoid(self.pcu_prior)
        # Compute attention weights
        attn_weights = self.attention(weighted)
        # Apply attention and embed
        attended = weighted * attn_weights
        embedded = self.embed(attended)
        
        return self.layer_norm(embedded)  # Normalize output


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
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        # Project to Q/K/V — shape stays (B, N, out_features)
        queries = self.W_q(node_features)
        keys    = self.W_k(node_features)
        values  = self.W_v(node_features)
        # Build attn_mask: (B*heads, N, N) bool — True = block
        # Expand adjacency with self-loops so each node attends to itself
        identity = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        adj_with_loops = adjacency + identity.unsqueeze(0)  # (B, N, N)
        block = (adj_with_loops == 0)                        # True where NOT connected
        attn_mask = block.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * self.n_heads, num_nodes, num_nodes)
        # MultiheadAttention with batch_first=True expects (B, N, embed_dim)
        attended_features, _ = self.attention(queries, keys, values, attn_mask=attn_mask)
        output = self.out_proj(attended_features)
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
        self.vca = VehicleClassAttentionSTGAT(n_classes=8, embed_dim=16)
        gcn_input_dim = node_features - 8 + 16  # replace 8 class features with 16-dim VCA
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(gcn_input_dim, hidden_gcn))
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        final_layer = self.dqn_head[-1]
        nn.init.uniform_(final_layer.weight, -0.001, 0.001)
        nn.init.constant_(final_layer.bias, 0.0)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        x = node_features.clone()
        queue_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        x[:, :, queue_idx] = x[:, :, queue_idx] / 50.0
        x[:, :, 5] = x[:, :, 5] / 300.0
        # VCA: replace indices 6-13 with 16-dim embedding
        vca_out = self.vca(x[:, :, 6:14].reshape(batch_size * num_nodes, 8))
        vca_out = vca_out.view(batch_size, num_nodes, 16)
        x = torch.cat([x[:, :, :6], vca_out, x[:, :, 14:]], dim=-1)  # (B, N, 32)
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
        self.vca = VehicleClassAttentionSTGAT(n_classes=8, embed_dim=16)
        gat_input_dim = node_features - 8 + 16
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttentionLayer(gat_input_dim, hidden_gat, n_heads, dropout))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GraphAttentionLayer(hidden_gat, hidden_gat, n_heads, dropout))
        self.dqn_head = nn.Sequential(
            nn.Linear(hidden_gat, hidden_dqn), nn.ReLU(),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(),
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
        
        # Initialize Q-head final layer with small weights to prevent initial Q-value explosion
        final_layer = self.dqn_head[-1]
        nn.init.uniform_(final_layer.weight, -0.001, 0.001)
        nn.init.constant_(final_layer.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        # Normalize inputs before VCA and GAT layers
        x = node_features.clone()
        queue_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        x[:, :, queue_idx] = x[:, :, queue_idx] / 50.0
        x[:, :, 5] = x[:, :, 5] / 300.0
        vehicle_class_features = x[:, :, 6:14].reshape(batch_size * num_nodes, 8)
        vehicle_context = self.vca(vehicle_class_features)
        vehicle_context = vehicle_context.view(batch_size, num_nodes, 16)
        # Concatenate: [0-5: basic, vehicle_context(16), 14+: scenario, neighbor+mask]
        x = torch.cat([x[:, :, :6], vehicle_context, x[:, :, 14:]], dim=-1)
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
    Ablation model — GAT without VCA. Raw node features fed directly to GAT.
    Proves VehicleClassAttention contributes by comparison with GAT_DQNet.
    """
    
    def __init__(self, node_features: int, n_actions: int, hidden_gat: int = 64,
                 hidden_dqn: int = 128, num_gat_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gat = hidden_gat
        self.hidden_dqn = hidden_dqn
        # No VCA — raw node_features fed directly to GAT
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttentionLayer(node_features, hidden_gat, n_heads, dropout))
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GraphAttentionLayer(hidden_gat, hidden_gat, n_heads, dropout))
        self.dqn_head = nn.Sequential(
            nn.Linear(hidden_gat, hidden_dqn), nn.ReLU(),
            nn.Linear(hidden_dqn, hidden_dqn), nn.ReLU(),
            nn.Linear(hidden_dqn, n_actions),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        final_layer = self.dqn_head[-1]
        nn.init.uniform_(final_layer.weight, -0.001, 0.001)
        nn.init.constant_(final_layer.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, num_nodes, _ = node_features.shape
        # Normalize inputs (same indices as other models, no VCA substitution)
        x = node_features.clone()
        queue_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        x[:, :, queue_idx] = x[:, :, queue_idx] / 50.0
        x[:, :, 5] = x[:, :, 5] / 300.0
        # NO VCA — pass all 24 features directly to GAT layers
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
        
        # Initialize Q-head final layer with small weights to prevent initial Q-value explosion
        final_layer = self.dqn_head[-1]
        nn.init.uniform_(final_layer.weight, -0.001, 0.001)
        nn.init.constant_(final_layer.bias, 0.0)
    
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


class STGATAgent:

    
    def __init__(
        self,
        obs_dim:          int,
        action_dim:       int,
        n_agents:         int,
        adjacency_matrix: np.ndarray,
        config:           dict = None,
    ):
        cfg = config or {}
        self.obs_dim     = obs_dim      # 24
        self.action_dim  = action_dim   # 2
        self.n_agents    = n_agents     # 9
        self.gamma       = cfg.get("gamma",    0.99)
        self.tau         = cfg.get("tau",      0.005)
        self.lr          = cfg.get("lr",       0.001)
        self.window      = cfg.get("window",   5)
        self.hidden_dim  = cfg.get("hidden_dim", 64)
        self.gat_heads   = cfg.get("gat_heads",  4)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build adjacency for GAT with symmetric normalization
        adj_with_self_loops = adjacency_matrix + np.eye(n_agents)
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degree = adj_with_self_loops.sum(axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        degree_mat_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_normalized = degree_mat_inv_sqrt @ adj_with_self_loops @ degree_mat_inv_sqrt
        
        self.adj = torch.tensor(
            adj_normalized,
            dtype=torch.float32,
            device=self.device
        )
        
        # Networks
        self.online_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        
        self.memory  = PrioritizedReplayBuffer(
            capacity  = 100000,
        )
        self.epsilon = 1.0
        self.steps   = 0
    
    def _build_network(self) -> nn.Module:
        """
        Builds the ST-GAT network as a single nn.Module.
        Easier to serialize for FedAvg weight averaging.
        """

        class STGATNetwork(nn.Module):
            def __init__(self, obs_dim, action_dim, n_agents,
                         window, hidden_dim, gat_heads, adj: torch.Tensor):
                super().__init__()
                self.n_agents = n_agents
                self.window   = window
                self.hidden   = hidden_dim

                self.register_buffer('adj_float', adj.float())

                self.vca  = VehicleClassAttentionSTGAT(n_classes=8, embed_dim=16)
                self.proj = nn.Linear(obs_dim + 16, hidden_dim)

                self.temporal_lstm = nn.LSTM(
                    input_size=hidden_dim, hidden_size=hidden_dim,
                    num_layers=1, batch_first=True,
                )

                self.gat1      = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=gat_heads, batch_first=True)
                self.gat_norm1 = nn.LayerNorm(hidden_dim)

                self.residual_scale = 0.01

                self.q_head = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_dim),
                )

                self._initialize_weights()

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                    elif isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight' in name:
                                nn.init.orthogonal_(param)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0.0)
                    elif isinstance(m, nn.MultiheadAttention):
                        nn.init.xavier_uniform_(m.in_proj_weight)
                        nn.init.xavier_uniform_(m.out_proj.weight)
                        if m.in_proj_bias is not None:
                            nn.init.constant_(m.in_proj_bias, 0.0)
                        if m.out_proj.bias is not None:
                            nn.init.constant_(m.out_proj.bias, 0.0)
                final_layer = self.q_head[-1]
                nn.init.uniform_(final_layer.weight, -0.001, 0.001)
                nn.init.constant_(final_layer.bias, 0.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                x shape: (batch, n_agents, T, obs_dim)
                Returns: (batch, n_agents, action_dim)
                """
                B, N, T, D = x.shape

                # Normalize inputs
                x_norm = x.clone()
                queue_idx = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
                x_norm[:, :, :, queue_idx] = x_norm[:, :, :, queue_idx] / 50.0
                x_norm[:, :, :, 5] = x_norm[:, :, :, 5] / 300.0
                x = x_norm

                # VehicleClassAttention
                class_feats = x[:, :, :, 6:14].reshape(B * N * T, 8)
                vca_out = self.vca(class_feats).reshape(B, N, T, 16)

                # Feature projection
                x_cat  = torch.cat([x, vca_out], dim=-1)          # (B, N, T, 40)
                x_proj = torch.relu(self.proj(x_cat.reshape(B * N, T, D + 16)))  # (B*N, T, hidden)

                # Temporal LSTM
                lstm_out, _ = self.temporal_lstm(x_proj)           # (B*N, T, hidden)
                temporal   = lstm_out[:, -1, :].reshape(B, N, -1)  # (B, N, hidden)

                # Spatial attention with adjacency mask (Pre-LN)
                # Build attn_mask: True = block attention (not connected)
                # adj_float is (N, N); expand to (B, N, N) then to (B*heads, N, N)
                adj = self.adj_float.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
                identity = torch.eye(N, device=adj.device).unsqueeze(0)
                adj_with_loops = (adj + identity).clamp(max=1.0)
                block_mask = (adj_with_loops == 0)  # True where NOT connected
                n_heads = self.gat1.num_heads
                attn_mask = block_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
                attn_mask = attn_mask.reshape(B * n_heads, N, N)

                temporal_ln = self.gat_norm1(temporal)
                sp1_out, _ = self.gat1(
                    temporal_ln, temporal_ln, temporal_ln,
                    attn_mask=attn_mask,
                )
                sp1 = temporal + self.residual_scale * sp1_out      # (B, N, hidden)

                return self.q_head(sp1)                             # (B, N, action_dim)

        return STGATNetwork(
            self.obs_dim, self.action_dim, self.n_agents,
            self.window, self.hidden_dim, self.gat_heads,
            adj=self.adj,
        )
    
    def act(self, obs_history: np.ndarray, evaluate: bool = False) -> list:
        """
        obs_history shape: (n_agents, T, obs_dim)
        Returns list of actions for all 9 agents.
        """
        if not evaluate and np.random.random() < self.epsilon:
            return [np.random.randint(self.action_dim)
                    for _ in range(self.n_agents)]
        
        obs_t = torch.tensor(
            obs_history[np.newaxis],   # add batch dim: (1, N, T, D)
            dtype=torch.float32,
            device=self.device
        )
        with torch.no_grad():
            q_vals = self.online_net(obs_t)   # (1, N, action_dim)
        return q_vals[0].argmax(dim=-1).cpu().numpy().tolist()
    
    def remember(self, obs_history, actions, rewards,
                 next_obs_history, dones):
        """Store transition in replay buffer."""
        self.memory.add(
            obs_history, actions, rewards, next_obs_history, dones
        )
    
    def learn(self, batch_size: int = 256) -> float:
        """
        Double DQN update with soft target network sync.
        Returns loss value for logging.
        """
        # Start training once we have enough samples (at least batch_size)
        if len(self.memory) < batch_size:
            return 0.0
        
        # Use smaller effective batch if buffer is still small
        effective_batch_size = min(batch_size, len(self.memory))
        
        # Use uniform sampling instead of prioritized sampling to reduce bias
        samples, _, _ = self.memory.sample_uniform(effective_batch_size)
        
        # Unpack samples - PER stores (state, action, reward, next_state, done, adjacency, node_id)
        obs_h_list = []
        acts_list = []
        rews_list = []
        next_obs_h_list = []
        dones_list = []
        
        for sample in samples:
            obs_h_list.append(sample[0])
            acts_list.append(sample[1])
            rews_list.append(sample[2])
            next_obs_h_list.append(sample[3])
            dones_list.append(sample[4])
        
        obs_h = np.stack(obs_h_list)  # (B, N, T, D)
        acts = np.stack(acts_list)    # (B, N)
        rews = np.stack(rews_list)    # (B, N)
        next_obs_h = np.stack(next_obs_h_list)  # (B, N, T, D)
        dones = np.stack(dones_list)  # (B, N) - but dones is scalar, need to broadcast
        
        # Handle scalar dones
        if dones.ndim == 1:
            dones = np.tile(dones[:, np.newaxis], (1, self.n_agents))
        
        obs_t      = torch.tensor(obs_h,      dtype=torch.float32,  device=self.device)
        next_obs_t = torch.tensor(next_obs_h, dtype=torch.float32,  device=self.device)
        acts_t     = torch.tensor(acts,       dtype=torch.long,     device=self.device)
        rews_t     = torch.tensor(rews,       dtype=torch.float32,  device=self.device)
        dones_t    = torch.tensor(dones,      dtype=torch.float32,  device=self.device)
        
        # Clip rewards to prevent unbounded targets
        rews_t = torch.clamp(rews_t, -1.0, 1.0)
        
        # Current Q-values
        q_curr = self.online_net(obs_t)                     # (B, N, 3)
        q_curr = torch.clamp(q_curr, -10.0, 10.0)           # Clip to prevent explosion
        q_curr = q_curr.gather(
            2, acts_t.unsqueeze(-1)
        ).squeeze(-1)                                        # (B, N)
        
        # Double DQN targets
        with torch.no_grad():
            # Target net evaluates - clip outputs to prevent explosion
            q_next_target = self.target_net(next_obs_t)
            q_next_target = torch.clamp(q_next_target, -10.0, 10.0)
            
            # Online net selects action - clip outputs for action selection
            q_next_online = self.online_net(next_obs_t)
            q_next_online = torch.clamp(q_next_online, -10.0, 10.0)
            next_actions = q_next_online.argmax(dim=-1)  # (B, N)
            
            # Use target net's Q-values with selected actions
            q_next = q_next_target.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1)                                    # (B, N)
        
        # Compute Bellman targets and clip to prevent explosion
        targets = rews_t + self.gamma * q_next * (1 - dones_t)
        targets = torch.clamp(targets, -10.0, 10.0)
        
        # Loss
        loss_per = nn.functional.smooth_l1_loss(
            q_curr, targets.detach(), reduction='none'
        )
        loss = loss_per.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=1.0
        )
        self.optimizer.step()
        
        # Priority updates removed - using uniform sampling, no need to update priorities
        # td_errors = loss_per.mean(dim=-1).detach().cpu().numpy()
        # self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Soft target update using configured tau
        for t_p, o_p in zip(self.target_net.parameters(),
                             self.online_net.parameters()):
            t_p.data.copy_(self.tau * o_p.data + (1 - self.tau) * t_p.data)
        
        self.steps += 1
        return loss.item()
    
    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon
    
    def get_weights(self) -> dict:
        return {k: v.cpu().clone() for k, v in
                self.online_net.state_dict().items()}
    
    def set_weights(self, weights: dict):
        self.online_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
