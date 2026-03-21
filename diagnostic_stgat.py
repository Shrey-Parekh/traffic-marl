"""
ST-GAT specific bug diagnosis.
Isolates issues in temporal module, history buffer, and STGATAgent.
"""

import sys
import numpy as np
import torch
from src.agent import STGATAgent, HistoryBuffer
from src.config import TEMPORAL_CONFIG
import inspect

print("=== ST-GAT SPECIFIC BUG DIAGNOSIS ===\n")

# Test 1: History buffer size
obs_dim = 24
n_agents = 9
window = TEMPORAL_CONFIG['window']

print(f"1. History buffer config: window={window}, obs_dim={obs_dim}")

# Simulate history buffer
test_obs = np.random.randn(n_agents, obs_dim)
history = np.zeros((n_agents, window, obs_dim))

# Roll and update (same as HistoryBuffer.update)
history = np.roll(history, shift=-1, axis=1)
history[:, -1, :] = test_obs

print(f"   History shape: {history.shape}")
print(f"   Expected: (9, 5, 24)")
print(f"   Status: {'✓ OK' if history.shape == (9, 5, 24) else '✗ WRONG'}\n")

# Test 2: STGATAgent initialization
agent = STGATAgent(
    obs_dim=24,
    action_dim=3,
    n_agents=9,
    adjacency_matrix=np.ones((9, 9)),
    config=TEMPORAL_CONFIG
)

print(f"2. STGATAgent created")
print(f"   Online net: {type(agent.online_net).__name__}")
print(f"   Target net: {type(agent.target_net).__name__}")
print(f"   Memory size: {len(agent.memory)}")
print(f"   Tau: {agent.tau if hasattr(agent, 'tau') else 'NOT SET'}")
print(f"   Learning rate: {agent.lr if hasattr(agent, 'lr') else 'NOT SET'}\n")

# Test 3: Check tau value in learn() method
print("3. Checking tau in STGATAgent.learn():")
source = inspect.getsource(agent.learn)
if '0.005' in source or '0.995' in source:
    print(f"   ✗ CRITICAL BUG: tau still 0.005 in STGATAgent.learn()")
    tau_lines = [line.strip() for line in source.split('\n') if '0.005' in line or '0.995' in line]
    for line in tau_lines:
        print(f"   Line: {line}")
elif '0.01' in source and '0.99' in source:
    print(f"   ✓ Tau correctly set to 0.01 in STGATAgent.learn()")
else:
    print(f"   ⚠ Cannot find tau value in learn()")

print("\n" + "="*60)

# Test 4: Forward pass with history
test_history = np.random.randn(1, n_agents, window, obs_dim)
test_history_t = torch.FloatTensor(test_history).to(agent.device)

print(f"4. Forward pass test:")
print(f"   Input shape: {test_history_t.shape}")
print(f"   Expected: (1, 9, 5, 24)")

with torch.no_grad():
    try:
        q_values = agent.online_net(test_history_t)
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {q_values.shape}")
        print(f"   Expected output: (1, 9, 3)")
        print(f"   Status: {'✓ OK' if q_values.shape == (1, 9, 3) else '✗ WRONG'}")
        
        # Check for NaN or extreme values
        if torch.isnan(q_values).any():
            print(f"   ✗ CRITICAL: Q-values contain NaN!")
        elif torch.isinf(q_values).any():
            print(f"   ✗ CRITICAL: Q-values contain Inf!")
        elif q_values.abs().max() > 100:
            print(f"   ⚠ WARNING: Q-values very large (max={q_values.abs().max():.2f})")
        else:
            print(f"   ✓ Q-values in normal range (max={q_values.abs().max():.4f})")
    except Exception as e:
        print(f"   ✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()

print("="*60)

# Test 5: Training step with realistic data
print(f"\n5. Training step test:")
print(f"   Adding 300 transitions and training...")

for step in range(300):
    obs = np.random.randn(9, 5, 24)
    actions = np.random.randint(0, 3, 9)
    rewards = -np.abs(np.random.randn(9)) * 0.1
    next_obs = obs + np.random.randn(9, 5, 24) * 0.01
    agent.remember(obs, actions, rewards, next_obs, False)

loss = agent.learn(batch_size=64)
print(f"   Buffer size: {len(agent.memory)}")
print(f"   Loss: {loss:.6f}")
print(f"   Status: {'✓ Training started' if loss > 0 else '✗ No training (loss=0)'}")

if loss > 0:
    # Check target-online distance
    online_param = list(agent.online_net.parameters())[0].data
    target_param = list(agent.target_net.parameters())[0].data
    distance = torch.mean((online_param - target_param)**2).item()
    print(f"   Target-Online distance: {distance:.8f}")
    print(f"   Status: {'✓ OK' if distance > 1e-6 else '✗ Target=Online (distance too small)'}")

print("\n" + "="*60)

# Test 6: Check gradient flow
print(f"\n6. Gradient flow test:")
agent.online_net.train()
obs_batch = torch.randn(64, 9, 5, 24).to(agent.device)
q_out = agent.online_net(obs_batch)
loss_test = q_out.mean()
loss_test.backward()

grad_norms = []
for name, param in agent.online_net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        if grad_norm == 0:
            print(f"   ✗ Zero gradient in: {name}")
        elif grad_norm > 10:
            print(f"   ⚠ Large gradient in: {name} (norm={grad_norm:.4f})")

if grad_norms:
    avg_grad = np.mean(grad_norms)
    max_grad = np.max(grad_norms)
    print(f"   Average gradient norm: {avg_grad:.6f}")
    print(f"   Max gradient norm: {max_grad:.6f}")
    print(f"   Status: {'✓ OK' if 1e-6 < avg_grad < 10 else '⚠ Unusual gradient scale'}")
else:
    print(f"   ✗ CRITICAL: No gradients computed!")

print("\n" + "="*60)
print("\nDiagnostic complete. Check for ✗ and ⚠ markers above.")
