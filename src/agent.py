from __future__ import annotations

from collections import namedtuple, deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MetaController(nn.Module):
    """Meta-controller for adaptive exploration and learning rate scaling.
    
    Takes recent performance metrics and outputs adaptive hyperparameters
    within user-defined bounds.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # [recent_reward, recent_queue, episode_progress]
        hidden_dim: int = 32,
        epsilon_min: float = 0.05,
        epsilon_max: float = 0.3,
        lr_scale_min: float = 0.5,
        lr_scale_max: float = 1.5,
    ):
        super().__init__()
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.lr_scale_min = lr_scale_min
        self.lr_scale_max = lr_scale_max
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [epsilon_logit, lr_scale_logit]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, metrics: torch.Tensor) -> tuple[float, float]:
        """
        Args:
            metrics: [recent_reward, recent_queue, episode_progress]
        
        Returns:
            epsilon: Adaptive exploration rate
            lr_scale: Learning rate scale factor
        """
        logits = self.net(metrics)
        
        # Use sigmoid to map to [0, 1], then scale to desired ranges
        epsilon_raw = torch.sigmoid(logits[0])
        lr_scale_raw = torch.sigmoid(logits[1])
        
        # Scale to user-defined bounds
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * epsilon_raw
        lr_scale = self.lr_scale_min + (self.lr_scale_max - self.lr_scale_min) * lr_scale_raw
        
        return float(epsilon.item()), float(lr_scale.item())


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "adjacency", "node_id"),
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
            )
        )

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        
        # Add self-loops
        identity = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_with_loops = adjacency + identity
        
        # Normalize: D^-1 * A
        degree = adjacency_with_loops.sum(dim=2, keepdim=True)
        degree_inv = torch.pow(degree, -1.0)
        degree_inv[degree_inv == float('inf')] = 0.0
        normalized_adj = degree_inv * adjacency_with_loops
        
        # Aggregate: normalized_adj @ node_features
        message = torch.bmm(normalized_adj, node_features)
        
        # Transform
        output = self.linear(message)
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
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvLayer(node_features, hidden_gcn))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GraphConvLayer(hidden_gcn, hidden_gcn))
        
        # DQN head
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
        # Handle both 2D (single sample) and 3D (batch) inputs
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adjacency = adjacency.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_nodes, _ = node_features.shape
        
        # Validate input consistency
        if adjacency.shape != (batch_size, num_nodes, num_nodes):
            raise ValueError(
                f"Adjacency matrix shape {adjacency.shape} does not match "
                f"node features shape {node_features.shape}. Expected adjacency "
                f"shape: ({batch_size}, {num_nodes}, {num_nodes})"
            )
        
        # Check for consistent graph sizes within batch
        if batch_size > 1:
            # For now, we only support batches with identical graph sizes
            # Future enhancement could support variable sizes with padding
            for i in range(1, batch_size):
                if node_features[i].shape[0] != num_nodes:
                    raise ValueError(
                        f"Mixed graph sizes in batch not supported. "
                        f"Graph 0 has {num_nodes} nodes, graph {i} has {node_features[i].shape[0]} nodes. "
                        f"All graphs in a batch must have the same number of nodes."
                    )
        
        # GCN message passing
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adjacency)
            x = F.relu(x)
        
        # Reshape for DQN head: [batch, num_nodes, hidden_gcn] -> [batch*num_nodes, hidden_gcn]
        x = x.view(batch_size * num_nodes, self.hidden_gcn)
        
        # DQN head
        q_values = self.dqn_head(x)
        
        # Reshape back: [batch*num_nodes, n_actions] -> [batch, num_nodes, n_actions]
        q_values = q_values.view(batch_size, num_nodes, self.n_actions)
        
        if squeeze_output:
            q_values = q_values.squeeze(0)
        
        return q_values
