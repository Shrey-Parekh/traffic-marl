# Requirements Document: Traffic Light Reward System Fix

## Introduction

This specification addresses the reward system for a GNN-DQN traffic light control system managing 10 intersections in a grid layout. The current reward systems have failed to enable the agent to learn effective traffic control policies. The system requires a reward structure that provides clear learning signals for queue minimization while incentivizing appropriate switching behavior.

## Glossary

- **Agent**: The GNN-DQN reinforcement learning model that controls traffic lights across all intersections
- **Intersection**: A single traffic light location with North-South (NS) and East-West (EW) phases
- **Queue**: The number of vehicles waiting at an intersection approach
- **Total_Queue**: Sum of all waiting vehicles across all approaches at an intersection
- **Imbalance**: The absolute difference between NS queue length and EW queue length at an intersection
- **Switching**: Changing the traffic light phase from NS to EW or vice versa
- **Phase**: The current active direction (NS or EW) at an intersection
- **Episode**: A complete training run of fixed duration
- **Reward_System**: The function that maps environment state and agent action to a scalar reward value
- **Baseline_Controller**: A fixed-time traffic light controller used for performance comparison

## Requirements

### Requirement 1: Queue Minimization

**User Story:** As a traffic control system, I want to minimize the total number of waiting vehicles, so that overall traffic congestion is reduced.

#### Acceptance Criteria

1. THE Reward_System SHALL penalize high queue lengths at each intersection
2. WHEN Total_Queue increases, THE Reward_System SHALL provide a negative reward proportional to the queue length
3. THE Reward_System SHALL apply queue penalties consistently across all intersections
4. THE Reward_System SHALL use Total_Queue as the primary optimization target

### Requirement 2: Balanced Queue Service

**User Story:** As a traffic control system, I want to serve both directions fairly, so that no direction experiences excessive waiting times.

#### Acceptance Criteria

1. THE Reward_System SHALL penalize queue imbalance between NS and EW directions
2. WHEN Imbalance exceeds a threshold, THE Reward_System SHALL provide stronger negative rewards
3. THE Reward_System SHALL calculate Imbalance as the absolute difference between NS queue and EW queue
4. THE Reward_System SHALL weight imbalance penalties to encourage balanced service

### Requirement 3: Conditional Switching Incentives

**User Story:** As a reinforcement learning agent, I want clear signals about when switching is beneficial, so that I can learn effective switching policies.

#### Acceptance Criteria

1. WHEN the Agent switches phases AND Imbalance is high, THE Reward_System SHALL provide a positive reward
2. WHEN the Agent switches phases AND queues are balanced, THE Reward_System SHALL provide a negative reward
3. WHEN the Agent maintains the current phase, THE Reward_System SHALL provide zero switching reward
4. THE Reward_System SHALL define "high imbalance" as Imbalance exceeding a configurable threshold
5. THE Reward_System SHALL define "balanced queues" as Imbalance below a configurable threshold

### Requirement 4: Controllable Reward Components

**User Story:** As a reinforcement learning agent, I want rewards based only on factors I can control, so that I can learn effective policies without noise from random events.

#### Acceptance Criteria

1. THE Reward_System SHALL NOT include components based on random vehicle arrivals
2. THE Reward_System SHALL NOT include components based on cars_served (which depends on arrival timing)
3. THE Reward_System SHALL only use state variables directly observable by the Agent
4. THE Reward_System SHALL only reward actions that the Agent can directly control

### Requirement 5: Reward Component Weights

**User Story:** As a system designer, I want configurable reward weights, so that I can tune the balance between competing objectives.

#### Acceptance Criteria

1. THE Reward_System SHALL use a configurable weight for queue penalties
2. THE Reward_System SHALL use a configurable weight for imbalance penalties
3. THE Reward_System SHALL use configurable rewards for beneficial switches
4. THE Reward_System SHALL use configurable penalties for unnecessary switches
5. THE Reward_System SHALL use a configurable threshold for determining high imbalance
6. WHERE weights are not explicitly configured, THE Reward_System SHALL use default values: queue_weight=-0.5, imbalance_weight=-1.5, good_switch_reward=+3.0, bad_switch_penalty=-2.0, imbalance_threshold=3.0

### Requirement 6: Edge Case Handling

**User Story:** As a traffic control system, I want robust reward calculation, so that the system handles edge cases without errors.

#### Acceptance Criteria

1. WHEN Total_Queue is zero, THE Reward_System SHALL return zero queue penalty
2. WHEN Imbalance is zero, THE Reward_System SHALL return zero imbalance penalty
3. WHEN both NS and EW queues are zero, THE Reward_System SHALL classify this as balanced (no switching incentive)
4. IF queue state data is missing or invalid, THEN THE Reward_System SHALL raise an error with a descriptive message
5. THE Reward_System SHALL handle floating-point queue values by rounding or truncating appropriately

### Requirement 7: Performance Validation

**User Story:** As a system operator, I want to validate that the reward system enables learning, so that I can confirm the agent improves over time.

#### Acceptance Criteria

1. WHEN training for 100 episodes, THE Agent SHALL demonstrate decreasing loss values
2. WHEN training for 100 episodes, THE Agent SHALL demonstrate decreasing average queue lengths
3. WHEN training for 100 episodes, THE Agent SHALL demonstrate increasing throughput compared to early episodes
4. THE Agent SHALL achieve performance exceeding the Baseline_Controller by at least 10% after 100 episodes
5. THE Reward_System SHALL enable the Agent to learn basic switching behavior within the first 30 episodes

### Requirement 8: Reward Calculation Interface

**User Story:** As a developer, I want a clear interface for reward calculation, so that I can integrate it with the training loop.

#### Acceptance Criteria

1. THE Reward_System SHALL accept current state information as input
2. THE Reward_System SHALL accept the action taken as input
3. THE Reward_System SHALL accept previous state information for differential calculations
4. THE Reward_System SHALL return a single scalar reward value
5. THE Reward_System SHALL compute rewards independently for each intersection
6. THE Reward_System SHALL provide a method to aggregate individual intersection rewards into a total reward

### Requirement 9: Logging and Debugging

**User Story:** As a developer, I want detailed reward component logging, so that I can debug and tune the reward system.

#### Acceptance Criteria

1. THE Reward_System SHALL log individual reward components (queue penalty, imbalance penalty, switching reward) for each intersection
2. THE Reward_System SHALL log the total reward for each step
3. THE Reward_System SHALL log aggregate statistics (mean queue, mean imbalance, switch frequency) for each episode
4. WHEN debug mode is enabled, THE Reward_System SHALL log detailed state information for each reward calculation
5. THE Reward_System SHALL provide a summary report at the end of each episode showing reward component distributions

### Requirement 10: Configuration Management

**User Story:** As a system operator, I want centralized reward configuration, so that I can easily adjust parameters without modifying code.

#### Acceptance Criteria

1. THE Reward_System SHALL read configuration parameters from a configuration file or object
2. THE Reward_System SHALL validate configuration parameters at initialization
3. IF configuration parameters are invalid, THEN THE Reward_System SHALL raise an error with specific validation failures
4. THE Reward_System SHALL support runtime parameter updates for experimentation
5. THE Reward_System SHALL log the active configuration at the start of each training run
