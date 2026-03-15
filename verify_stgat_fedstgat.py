"""
Verification script for ST-GAT and Fed-ST-GAT implementation.
Tests all 9 required checks before training.
"""

import numpy as np
import torch
from src.agent import STGATAgent, FedSTGATAgent, HistoryBuffer
from src.config import TEMPORAL_CONFIG, FEDERATED_CONFIG, OBS_FEATURES_PER_AGENT

print('=== ST-GAT / Fed-ST-GAT VERIFICATION ===')
print()

# Adjacency matrix for 3x3 grid
adj = np.array([
    [0,1,0,1,0,0,0,0,0],
    [1,0,1,0,1,0,0,0,0],
    [0,1,0,0,0,1,0,0,0],
    [1,0,0,0,1,0,1,0,0],
    [0,1,0,1,0,1,0,1,0],
    [0,0,1,0,1,0,0,0,1],
    [0,0,0,1,0,0,0,1,0],
    [0,0,0,0,1,0,1,0,1],
    [0,0,0,0,0,1,0,1,0],
], dtype=np.float32)

# Test 1: HistoryBuffer
print('Test 1: HistoryBuffer...')
hb = HistoryBuffer(n_agents=9, window=5, obs_dim=24)
obs = np.random.randn(9, 24).astype(np.float32)
for _ in range(7):  # update more than window to test circular
    hb.update(obs)
h = hb.get()
assert h.shape == (9, 5, 24), f'FAIL: expected (9,5,24), got {h.shape}'
hb.reset()
assert np.all(hb.get() == 0), 'FAIL: reset should zero buffer'
print(f'  PASS: shape {h.shape}, reset works')

# Test 2: STGATAgent instantiation
print('Test 2: STGATAgent instantiation...')
agent = STGATAgent(
    obs_dim=24, action_dim=3, n_agents=9,
    adjacency_matrix=adj
)
print(f'  PASS: STGATAgent created')

# Test 3: STGATAgent forward pass
print('Test 3: STGATAgent forward pass...')
obs_history = np.random.randn(9, 5, 24).astype(np.float32)
actions = agent.act(obs_history)
assert len(actions) == 9, f'FAIL: expected 9 actions, got {len(actions)}'
assert all(0 <= a <= 2 for a in actions), f'FAIL: invalid actions {actions}'
print(f'  PASS: actions {actions}')

# Test 4: STGATAgent remember and learn
print('Test 4: STGATAgent learn...')
for _ in range(300):
    obs_h      = np.random.randn(9, 5, 24).astype(np.float32)
    acts       = [np.random.randint(3) for _ in range(9)]
    rews       = np.random.randn(9).astype(np.float32)
    next_obs_h = np.random.randn(9, 5, 24).astype(np.float32)
    done       = False
    agent.remember(obs_h, acts, rews, next_obs_h, done)
loss = agent.learn(batch_size=32)
assert loss > 0, f'FAIL: loss should be > 0, got {loss}'
print(f'  PASS: loss = {loss:.4f}')

# Test 5: FedSTGATAgent instantiation
print('Test 5: FedSTGATAgent instantiation...')
fed_agent = FedSTGATAgent(
    obs_dim=24, action_dim=3, n_agents=9,
    adjacency_matrix=adj
)
assert len(fed_agent.local_agents) == 9, 'FAIL: should have 9 local agents'
print(f'  PASS: FedSTGATAgent with 9 local agents')

# Test 6: FedSTGATAgent act
print('Test 6: FedSTGATAgent act...')
obs_history = np.random.randn(9, 5, 24).astype(np.float32)
actions = fed_agent.act(obs_history)
assert len(actions) == 9, f'FAIL: expected 9 actions, got {len(actions)}'
print(f'  PASS: actions {actions}')

# Test 7: FedAvg aggregation
print('Test 7: FedAvg aggregation...')
# Get weights before aggregation
weights_before = [
    {k: v.clone() for k, v in fed_agent.local_agents[i].online_net.state_dict().items()}
    for i in range(9)
]
# Manually set different weights for agent 0
for p in fed_agent.local_agents[0].online_net.parameters():
    p.data.fill_(1.0)
for p in fed_agent.local_agents[1].online_net.parameters():
    p.data.fill_(0.0)

fed_agent.federated_aggregate()

# After aggregation all agents should have same weights
w0 = list(fed_agent.local_agents[0].online_net.parameters())[0].data
w1 = list(fed_agent.local_agents[1].online_net.parameters())[0].data
assert torch.allclose(w0, w1, atol=1e-5), 'FAIL: agents have different weights after FedAvg'
print(f'  PASS: FedAvg aggregation equalizes weights')

# Test 8: on_episode_end triggers aggregation at correct interval
print('Test 8: Federation interval...')
fed_agent2 = FedSTGATAgent(
    obs_dim=24, action_dim=3, n_agents=9,
    adjacency_matrix=adj,
    config={'fed_interval': 3}
)
for p in fed_agent2.local_agents[0].online_net.parameters():
    p.data.fill_(99.0)
# Episode 1 and 2: no aggregation
fed_agent2.on_episode_end()
fed_agent2.on_episode_end()
w_before = list(fed_agent2.local_agents[0].online_net.parameters())[0].data.mean().item()
# Episode 3: aggregation should fire
fed_agent2.on_episode_end()
w_after = list(fed_agent2.local_agents[0].online_net.parameters())[0].data.mean().item()
assert abs(w_before - w_after) > 0.1, 'FAIL: aggregation did not change weights at interval'
print(f'  PASS: aggregation fires at episode 3 (interval=3)')

# Test 9: existing models unaffected
print('Test 9: Existing models unaffected...')
from src.agent import DQNet, GAT_DQNet
# These are networks, not agents - they need to be wrapped differently
# Just verify they can still be instantiated
dqn_net = DQNet(obs_dim=24, n_actions=3)
gat_net = GAT_DQNet(node_features=24, n_actions=3)
print(f'  PASS: DQN and GAT-DQN networks unaffected')

print()
print('=== ALL 9 TESTS PASSED ===')
print('Ready to train ST-GAT and Fed-ST-GAT.')
