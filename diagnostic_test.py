"""
Diagnostic tests to identify training loop bugs.
Run these tests to isolate the root cause of loss divergence.
"""

import torch
import numpy as np
from src.agent import STGATAgent

def test_network_learning():
    """Test 1: Is the network actually learning?"""
    print("\n=== Test 1: Network Learning ===")
    
    # Create minimal agent
    config = {
        "gamma": 0.95,
        "lr": 0.0005,  # Updated learning rate
        "window": 5,
        "hidden_dim": 64,
        "gat_heads": 4,
    }
    
    adjacency = np.eye(9)  # Simple adjacency
    agent = STGATAgent(
        obs_dim=24,
        action_dim=3,
        n_agents=9,
        adjacency_matrix=adjacency,
        config=config
    )
    
    # Fixed test state
    test_obs = torch.zeros(1, 9, 5, 24).to(agent.device)
    
    print("Episode | Q[0,0] | Q[0,1] | Q[0,2] | Buffer | Updates")
    print("-" * 60)
    
    total_updates = 0
    # Simulate realistic training: 300 steps per episode
    for ep in range(1, 51):
        episode_updates = 0
        # Add 300 transitions per episode (realistic)
        for step in range(300):
            obs = np.random.randn(9, 5, 24)
            actions = np.random.randint(0, 3, 9)
            # Rewards with some structure (not pure noise)
            rewards = -np.abs(np.random.randn(9)) * 0.1  # Negative rewards (minimize queue)
            next_obs = obs + np.random.randn(9, 5, 24) * 0.01  # Small state change
            agent.remember(obs, actions, rewards, next_obs, False)
            
            # Train every step once buffer is ready
            loss = agent.learn(batch_size=64)
            if loss > 0:
                episode_updates += 1
                total_updates += 1
        
        # Check Q-values
        if ep % 10 == 0 or ep == 1:
            with torch.no_grad():
                q_values = agent.online_net(test_obs)
                print(f"Ep {ep:3d} | {q_values[0,0,0]:6.3f} | {q_values[0,0,1]:6.3f} | "
                      f"{q_values[0,0,2]:6.3f} | {len(agent.memory):6d} | {episode_updates:3d}")

def test_target_network_distance():
    """Test 4: Target network distance from online network"""
    print("\n=== Test 4: Target-Online Distance ===")
    
    config = {"gamma": 0.95, "lr": 0.0005, "window": 5}
    adjacency = np.eye(9)
    agent = STGATAgent(24, 3, 9, adjacency, config)
    
    print("Updates | Target-Online MSE | Buffer Size")
    print("-" * 45)
    
    # Realistic: add transitions in batches
    for update in range(1, 501):
        # Add 10 transitions per update cycle
        for _ in range(10):
            obs = np.random.randn(9, 5, 24)
            actions = np.random.randint(0, 3, 9)
            rewards = -np.abs(np.random.randn(9)) * 0.1
            next_obs = obs + np.random.randn(9, 5, 24) * 0.01
            agent.remember(obs, actions, rewards, next_obs, False)
        
        # Train
        loss = agent.learn(batch_size=64)
        
        # Check distance every 100 updates
        if update % 100 == 0:
            online_param = list(agent.online_net.parameters())[0].data
            target_param = list(agent.target_net.parameters())[0].data
            distance = torch.mean((online_param - target_param)**2).item()
            print(f"{update:7d} | {distance:17.6f} | {len(agent.memory):11d}")

if __name__ == "__main__":
    print("Running diagnostic tests...")
    print("These tests isolate training loop bugs.")
    
    test_network_learning()
    test_target_network_distance()
    
    print("\n=== Expected Results ===")
    print("Test 1: Q-values should grow from ~0.0 to ~2.0+ by episode 50")
    print("        Updates should be ~290-300 per episode after buffer fills")
    print("Test 4: Distance should reach 0.001-0.05 by update 500")
