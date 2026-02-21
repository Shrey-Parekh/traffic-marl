from __future__ import annotations

from collections import namedtuple, deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "adjacency", "node_id", "log_prob", "value"),
)

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

class RolloutBuffer:
    """Buffer for on-policy algorithms like PPO and A2C."""
    
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
            self.states,
            self.actions,
            self.rewards,
            self.log_probs,
            self.values,
            self.dones,
            self.adjacencies,
            self.node_ids,
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
            embed_dim=out_features,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
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
        

        attended_features, attention_weights = self.attention(
            queries, keys, values, attn_mask=attn_mask_expanded
        )
        

        output = self.out_proj(attended_features)
        output = self.dropout(output)
        
        return output

class DQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
    """Graph Neural Network DQN for shared-policy multi-agent traffic control.
    
    This model uses PARAMETER SHARING - one GNN policy controls all intersections.
    Each intersection is a node in the graph, and the GNN learns spatial coordination
    patterns through message passing over the adjacency matrix.
    
    This is NOT independent-agent MARL - all agents share the same policy parameters.
    """
    def __init__(
        self,
        node_features: int,
        n_actions: int,
        hidden_gcn: int = 64,
        hidden_dqn: int = 128,
        num_gcn_layers: int = 2,
    ):
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
            nn.Linear(hidden_gcn, hidden_dqn),
            nn.ReLU(),
            nn.Linear(hidden_dqn, hidden_dqn),
            nn.ReLU(),
            nn.Linear(hidden_dqn, n_actions),
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass with support for variable graph sizes.
        
        Args:
            node_features: [batch_size, num_nodes, node_features] or [num_nodes, node_features]
            adjacency: [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
        
        Returns:
            Q-values: [batch_size, num_nodes, n_actions] or [num_nodes, n_actions]
        """

        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_nodes, _ = node_features.shape
        

        if adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError(
                f"Adjacency matrix shape {adjacency.shape} does not match "
                f"node features shape {node_features.shape}. Expected adjacency "
                f"shape: ({batch_size}, {num_nodes}, {num_nodes})"
            )
        

        if batch_size > 1:

            for i in range(1, batch_size):
                if node_features[i].shape[0] != num_nodes:
                    raise ValueError(
                        f"Mixed graph sizes in batch not supported. "
                        f"Graph 0 has {num_nodes} nodes, graph {i} has {node_features[i].shape[0]} nodes. "
                        f"All graphs in a batch must have the same number of nodes."
                    )
        

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
    """Graph Attention Network DQN for traffic control with attention-based spatial reasoning."""
    
    def __init__(
        self,
        node_features: int,
        n_actions: int,
        hidden_gat: int = 64,
        hidden_dqn: int = 128,
        num_gat_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
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
            nn.Linear(hidden_gat, hidden_dqn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dqn, hidden_dqn),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        """Forward pass with attention-based message passing."""

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

class PPO_GNN(nn.Module):
    """PPO with Graph Neural Network for traffic control."""
    
    def __init__(
        self,
        node_features: int,
        n_actions: int,
        hidden_gcn: int = 64,
        hidden_policy: int = 128,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gcn = hidden_gcn
        

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(node_features, hidden_gcn))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_gcn, hidden_gcn))
        

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_gcn, hidden_policy),
            nn.ReLU(),
            nn.Linear(hidden_policy, hidden_policy),
            nn.ReLU(),
            nn.Linear(hidden_policy, n_actions),
        )
        

        self.value_head = nn.Sequential(
            nn.Linear(hidden_gcn, hidden_policy),
            nn.ReLU(),
            nn.Linear(hidden_policy, hidden_policy),
            nn.ReLU(),
            nn.Linear(hidden_policy, 1),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both policy logits and values."""

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
        

        policy_logits = self.policy_head(x)
        values = self.value_head(x)
        

        policy_logits = policy_logits.view(batch_size, num_nodes, self.n_actions)
        values = values.view(batch_size, num_nodes, 1)
        
        if squeeze_output:
            policy_logits = policy_logits.squeeze(0)
            values = values.squeeze(0)
        
        return policy_logits, values
    
    def get_action_and_value(self, node_features: torch.Tensor, adjacency: torch.Tensor, node_idx: int = 0):
        """Get action and value for a specific node."""
        policy_logits, values = self.forward(node_features, adjacency)
        

        if policy_logits.dim() == 2:
            node_logits = policy_logits[node_idx]
            node_value = values[node_idx]
        else:
            node_logits = policy_logits[:, node_idx]
            node_value = values[:, node_idx]
        

        probs = F.softmax(node_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), node_value.item()

class GNN_A2C(nn.Module):
    """Actor-Critic with Graph Neural Network for traffic control."""
    
    def __init__(
        self,
        node_features: int,
        n_actions: int,
        hidden_gcn: int = 64,
        hidden_ac: int = 128,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        self.node_features = node_features
        self.n_actions = n_actions
        self.hidden_gcn = hidden_gcn
        

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(node_features, hidden_gcn))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_gcn, hidden_gcn))
        

        self.actor = nn.Sequential(
            nn.Linear(hidden_gcn, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, n_actions),
        )
        

        self.critic = nn.Sequential(
            nn.Linear(hidden_gcn, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, hidden_ac),
            nn.ReLU(),
            nn.Linear(hidden_ac, 1),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both action logits and values."""

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
        

        action_logits = self.actor(x)
        values = self.critic(x)
        

        action_logits = action_logits.view(batch_size, num_nodes, self.n_actions)
        values = values.view(batch_size, num_nodes, 1)
        
        if squeeze_output:
            action_logits = action_logits.squeeze(0)
            values = values.squeeze(0)
        
        return action_logits, values
    
    def get_action_and_value(self, node_features: torch.Tensor, adjacency: torch.Tensor, node_idx: int = 0):
        """Get action and value for a specific node."""
        action_logits, values = self.forward(node_features, adjacency)
        

        if action_logits.dim() == 2:
            node_logits = action_logits[node_idx]
            node_value = values[node_idx]
        else:
            node_logits = action_logits[:, node_idx]
            node_value = values[:, node_idx]
        

        probs = F.softmax(node_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), node_value.item()
