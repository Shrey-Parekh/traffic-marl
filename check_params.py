"""Check parameter counts for different models."""

import torch
import numpy as np
from src.agent import DQNet, GNN_DQNet, GAT_DQNet, STGATAgent

obs_dim = 24
n_actions = 3
n_agents = 9

# DQN
dqn = DQNet(obs_dim, n_actions)
dqn_params = sum(p.numel() for p in dqn.parameters())

# GNN-DQN
gnn = GNN_DQNet(obs_dim, n_actions)
gnn_params = sum(p.numel() for p in gnn.parameters())

# GAT-DQN
gat = GAT_DQNet(obs_dim, n_actions)
gat_params = sum(p.numel() for p in gat.parameters())

# ST-GAT
stgat = STGATAgent(
    obs_dim=obs_dim,
    action_dim=n_actions,
    n_agents=n_agents,
    adjacency_matrix=np.eye(n_agents),
    config={"lr": 0.0001, "gamma": 0.95, "window": 5}
)
stgat_params = sum(p.numel() for p in stgat.online_net.parameters())

print("Parameter Counts:")
print(f"DQN:     {dqn_params:>8,} parameters")
print(f"GNN-DQN: {gnn_params:>8,} parameters")
print(f"GAT-DQN: {gat_params:>8,} parameters")
print(f"ST-GAT:  {stgat_params:>8,} parameters")
print(f"\nST-GAT / DQN ratio: {stgat_params / dqn_params:.2f}x")
