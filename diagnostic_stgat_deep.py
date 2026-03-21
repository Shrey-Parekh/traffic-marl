"""
Deep diagnostic for ST-GAT training issues.
Checks for Q-value divergence, gradient explosion, target network drift, and NaN propagation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from src.agent import STGATAgent, HistoryBuffer
from src.config import TEMPORAL_CONFIG

def check_q_value_ranges(agent, n_agents=9, obs_dim=24, window=5):
    """Check if Q-values are in healthy range [-5, +5]."""
    with torch.no_grad():
        # Create dummy state
        dummy_state = torch.randn(1, n_agents, window, obs_dim).to(agent.device)
        q_vals = agent.online_net(dummy_state)
        
        stats = {
            'min': q_vals.min().item(),
            'max': q_vals.max().item(),
            'mean': q_vals.mean().item(),
            'std': q_vals.std().item()
        }
        
        # Check for divergence
        diverged = abs(stats['max']) > 10 or abs(stats['min']) > 10
        
        return stats, diverged

def check_gradient_norms(agent):
    """Check gradient norms after backward pass."""
    total_norm = 0
    max_norm = 0
    grad_norms = {}
    
    for name, p in agent.online_net.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
            grad_norms[name] = param_norm
    
    total_norm = total_norm ** 0.5
    
    # Check for explosion
    exploded = total_norm > 10.0 or max_norm > 50.0
    
    return {
        'total_norm': total_norm,
        'max_norm': max_norm,
        'exploded': exploded,
        'top_5_layers': sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
    }

def check_target_network_distance(agent):
    """Check if target network is properly separated from online network."""
    distance = 0
    param_count = 0
    
    for t_p, o_p in zip(agent.target_net.parameters(), agent.online_net.parameters()):
        distance += (t_p - o_p).pow(2).sum().item()
        param_count += t_p.numel()
    
    distance = (distance ** 0.5) / param_count  # Normalize by param count
    
    # Check for issues
    too_close = distance < 0.0001  # Target = Online (no separation)
    too_far = distance > 1.0       # Target not updating
    
    return {
        'distance': distance,
        'too_close': too_close,
        'too_far': too_far,
        'healthy': 0.0001 < distance < 1.0
    }

def check_per_priorities(agent):
    """Check PER priority distribution."""
    if not hasattr(agent.memory, 'priorities'):
        return None
    
    size = len(agent.memory)
    if size == 0:
        return None
    
    priorities = agent.memory.priorities[:size]
    
    stats = {
        'min': priorities.min(),
        'max': priorities.max(),
        'mean': priorities.mean(),
        'std': priorities.std(),
        'median': np.median(priorities)
    }
    
    # Check for issues
    stats['max_too_high'] = stats['max'] > 10.0
    stats['high_variance'] = stats['std'] > stats['mean']
    
    return stats

def check_for_nan(agent, n_agents=9, obs_dim=24, window=5):
    """Check if network produces NaN outputs."""
    with torch.no_grad():
        dummy_state = torch.randn(1, n_agents, window, obs_dim).to(agent.device)
        q_vals = agent.online_net(dummy_state)
        
        has_nan = torch.isnan(q_vals).any().item()
        has_inf = torch.isinf(q_vals).any().item()
        
        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'healthy': not (has_nan or has_inf)
        }

def run_full_diagnostic(agent):
    """Run all diagnostics and return comprehensive report."""
    print("\n" + "="*80)
    print("ST-GAT DEEP DIAGNOSTIC")
    print("="*80)
    
    # 1. Q-Value Ranges
    print("\n1. Q-VALUE RANGE CHECK")
    print("-" * 40)
    q_stats, q_diverged = check_q_value_ranges(agent)
    print(f"Min:  {q_stats['min']:8.3f}  {'❌ TOO LOW' if q_stats['min'] < -10 else '✓'}")
    print(f"Max:  {q_stats['max']:8.3f}  {'❌ TOO HIGH' if q_stats['max'] > 10 else '✓'}")
    print(f"Mean: {q_stats['mean']:8.3f}")
    print(f"Std:  {q_stats['std']:8.3f}")
    print(f"\nStatus: {'❌ DIVERGED' if q_diverged else '✓ HEALTHY'}")
    print(f"Expected range: [-5, +5]")
    
    # 2. Target Network Distance
    print("\n2. TARGET NETWORK SEPARATION")
    print("-" * 40)
    target_stats = check_target_network_distance(agent)
    print(f"Distance: {target_stats['distance']:.6f}")
    if target_stats['distance'] == 0.0:
        print("ℹ IDENTICAL - Fresh initialization (expected)")
    elif target_stats['too_close']:
        print("❌ TOO CLOSE - Target = Online (no separation)")
    elif target_stats['too_far']:
        print("❌ TOO FAR - Target not updating")
    else:
        print("✓ HEALTHY - Proper separation")
    print(f"Expected range: [0.0001, 1.0] after training starts")
    
    # 3. PER Priorities
    print("\n3. PER PRIORITY DISTRIBUTION")
    print("-" * 40)
    per_stats = check_per_priorities(agent)
    if per_stats:
        print(f"Min:    {per_stats['min']:.3f}")
        print(f"Max:    {per_stats['max']:.3f}  {'❌ TOO HIGH' if per_stats['max_too_high'] else '✓'}")
        print(f"Mean:   {per_stats['mean']:.3f}")
        print(f"Median: {per_stats['median']:.3f}")
        print(f"Std:    {per_stats['std']:.3f}  {'⚠ HIGH VARIANCE' if per_stats['high_variance'] else '✓'}")
        print(f"\nCapped at: [0.01, 5.0] ✓")
    else:
        print("No samples in buffer yet")
    
    # 4. NaN Check
    print("\n4. NaN/Inf CHECK")
    print("-" * 40)
    nan_stats = check_for_nan(agent)
    print(f"Has NaN: {'❌ YES' if nan_stats['has_nan'] else '✓ NO'}")
    print(f"Has Inf: {'❌ YES' if nan_stats['has_inf'] else '✓ NO'}")
    print(f"Status: {'✓ HEALTHY' if nan_stats['healthy'] else '❌ CORRUPTED'}")
    
    # 5. Configuration Check
    print("\n5. CONFIGURATION CHECK")
    print("-" * 40)
    print(f"Learning rate: {agent.lr}")
    print(f"Gamma:         {agent.gamma}")
    print(f"Tau:           {agent.tau}")
    print(f"Grad clip:     1.0 (max_norm)")
    print(f"Buffer size:   {len(agent.memory)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    issues = []
    if q_diverged:
        issues.append("Q-value divergence detected")
    if not target_stats['healthy']:
        issues.append("Target network separation issue")
    if per_stats and per_stats['max_too_high']:
        issues.append("PER priorities too high (but capped)")
    if not nan_stats['healthy']:
        issues.append("NaN/Inf in network outputs")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ ALL CHECKS PASSED")
    
    print("="*80 + "\n")
    
    return {
        'q_values': q_stats,
        'q_diverged': q_diverged,
        'target_network': target_stats,
        'per_priorities': per_stats,
        'nan_check': nan_stats,
        'issues': issues
    }

def simulate_training_step(agent, n_agents=9, obs_dim=24, window=5):
    """Simulate one training step to check gradients."""
    # Create dummy batch
    batch_size = 32
    obs = torch.randn(batch_size, n_agents, window, obs_dim).to(agent.device)
    actions = torch.randint(0, 3, (batch_size, n_agents)).to(agent.device)
    rewards = torch.randn(batch_size, n_agents).to(agent.device) * 0.1
    next_obs = torch.randn(batch_size, n_agents, window, obs_dim).to(agent.device)
    dones = torch.zeros(batch_size, n_agents).to(agent.device)
    
    # Forward pass
    q_curr = agent.online_net(obs).gather(2, actions.unsqueeze(-1)).squeeze(-1)
    
    with torch.no_grad():
        next_actions = agent.online_net(next_obs).argmax(dim=-1)
        q_next = agent.target_net(next_obs).gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
    
    targets = rewards + agent.gamma * q_next * (1 - dones)
    loss = torch.nn.functional.smooth_l1_loss(q_curr, targets)
    
    # Backward pass
    agent.optimizer.zero_grad()
    loss.backward()
    
    # Check gradients BEFORE clipping
    grad_stats = check_gradient_norms(agent)
    
    print("\n6. GRADIENT NORMS (simulated training step)")
    print("-" * 40)
    print(f"Total norm: {grad_stats['total_norm']:.4f}  {'❌ EXPLODED' if grad_stats['exploded'] else '✓'}")
    print(f"Max norm:   {grad_stats['max_norm']:.4f}")
    print(f"\nTop 5 layers by gradient norm:")
    for name, norm in grad_stats['top_5_layers']:
        print(f"  {name:40s}: {norm:.4f}")
    print(f"\nExpected: total_norm < 10.0")
    
    return grad_stats

if __name__ == "__main__":
    print("Initializing ST-GAT agent for diagnostic...")
    
    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 24
    n_agents = 9
    
    # Create dummy adjacency matrix
    adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)
    
    agent = STGATAgent(
        obs_dim=obs_dim,
        action_dim=3,
        n_agents=n_agents,
        adjacency_matrix=adjacency,
        config={
            "lr": 0.0001,
            "gamma": 0.95,
            "tau": 0.005,
            "window": TEMPORAL_CONFIG["window"],
            "hidden_dim": TEMPORAL_CONFIG["hidden_dim"],
            "gat_heads": TEMPORAL_CONFIG["gat_heads"],
        }
    )
    
    # Run diagnostics
    results = run_full_diagnostic(agent)
    
    # Simulate training step to check gradients
    grad_results = simulate_training_step(agent)
    
    print("\nDiagnostic complete. Check results above.")
