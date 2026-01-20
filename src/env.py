from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class EnvConfig:
    num_intersections: int = 2
    step_length: float = 2.0  # seconds per step
    max_steps: int = 300
    min_green: int = 5  # minimum steps before switching
    arrival_rate_ns: float = 0.3  # default vehicles/step (Poisson)
    arrival_rate_ew: float = 0.3
    # Optional per-intersection overrides for heterogeneous demand
    arrival_rate_ns_per_int: Optional[List[float]] = None
    arrival_rate_ew_per_int: Optional[List[float]] = None
    depart_capacity: int = 2  # vehicles that can depart per green per step
    seed: int = 42
    neighbor_obs: bool = False
    # Grid topology parameters
    grid_rows: Optional[int] = None  # Number of rows for grid topology (auto-computed if not specified)
    grid_cols: Optional[int] = None  # Number of columns for grid topology (auto-computed if not specified)


class MiniTrafficEnv:
    """Queue-based multi-intersection traffic environment with shared-policy multi-agent learning.

    This environment uses PARAMETER SHARING across all intersections - a single GNN/DQN policy
    controls all traffic lights. This is NOT independent-agent MARL.

    Grid topology: intersections arranged in a grid (rows × cols) with bidirectional routing
    
    Two approaches per intersection:
    - NS (North-South): Routes vertically (up/down) to neighbor intersections
    - EW (East-West): Routes horizontally (left/right) to neighbor intersections
    
    Actions per intersection: 0 = keep current phase, 1 = switch (if min_green satisfied)
    Phase: 0 = NS green, 1 = EW green
    Observations per agent: [ns_len, ew_len, phase, time_since_switch] (+ neighbor info if enabled)
    Reward per agent: negative queue length at its intersection: -(ns_len + ew_len)

    Tracking per-vehicle entry/exit times for average travel time.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.rng: Generator = default_rng(self.config.seed)

        # Grid topology setup: auto-compute dimensions if not specified
        if self.config.grid_rows is None or self.config.grid_cols is None:
            # Auto-compute from num_intersections (try to make square-ish grid)
            import math
            n = self.config.num_intersections
            self.grid_cols = int(math.ceil(math.sqrt(n)))
            self.grid_rows = int(math.ceil(n / self.grid_cols))
        else:
            self.grid_rows = self.config.grid_rows
            self.grid_cols = self.config.grid_cols
        # Override num_intersections to match grid
        self.num_intersections = self.grid_rows * self.grid_cols
        
        self.step_length = self.config.step_length
        self.max_steps = self.config.max_steps
        self.min_green = self.config.min_green
        self.arrival_rate_ns = self.config.arrival_rate_ns
        self.arrival_rate_ew = self.config.arrival_rate_ew
        self.arrival_rate_ns_per_int = (
            self.config.arrival_rate_ns_per_int
            if self.config.arrival_rate_ns_per_int is not None
            else [self.arrival_rate_ns] * self.num_intersections
        )
        self.arrival_rate_ew_per_int = (
            self.config.arrival_rate_ew_per_int
            if self.config.arrival_rate_ew_per_int is not None
            else [self.arrival_rate_ew] * self.num_intersections
        )
        self.depart_capacity = self.config.depart_capacity
        self.neighbor_obs = self.config.neighbor_obs

        # State
        self.current_step: int = 0
        self.phase: List[int] = [0 for _ in range(self.num_intersections)]
        self.time_since_switch: List[int] = [0 for _ in range(self.num_intersections)]

        # Queues store enter_step for each vehicle
        self.ns_queues: List[List[int]] = [[] for _ in range(self.num_intersections)]
        self.ew_queues: List[List[int]] = [[] for _ in range(self.num_intersections)]

        # Context features for traffic simulation
        self.episode_start_time: float = 0.0  # Will be set in reset()

        # Metrics (global and per-intersection)
        self.exited_vehicle_times: List[float] = []
        self.episode_throughput: int = 0
        self.episode_queue_sum: float = 0.0
        self.episode_queue_steps: int = 0
        self.per_int_throughput: List[int] = [0 for _ in range(self.num_intersections)]
        self.per_int_avg_queue_accum: List[float] = [0.0 for _ in range(self.num_intersections)]

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = default_rng(seed)
    
    def _grid_to_index(self, row: int, col: int) -> int:
        """Convert grid coordinates (row, col) to linear index."""
        return row * self.grid_cols + col
    
    def _index_to_grid(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to grid coordinates (row, col)."""
        return (idx // self.grid_cols, idx % self.grid_cols)
    
    def _get_grid_neighbors(self, idx: int) -> List[int]:
        """Get neighbor indices for an intersection in grid topology."""
        row, col = self._index_to_grid(idx)
        neighbors = []
        
        # Horizontal neighbors (left, right)
        if col > 0:
            neighbors.append(self._grid_to_index(row, col - 1))  # Left
        if col < self.grid_cols - 1:
            neighbors.append(self._grid_to_index(row, col + 1))  # Right
        
        # Vertical neighbors (up, down)
        if row > 0:
            neighbors.append(self._grid_to_index(row - 1, col))  # Up
        if row < self.grid_rows - 1:
            neighbors.append(self._grid_to_index(row + 1, col))  # Down
        
        return neighbors

    def _arrival(self, rate: float) -> int:
        return int(self.rng.poisson(rate))

    def _add_arrivals(self) -> None:
        for i in range(self.num_intersections):
            num_ns = self._arrival(self.arrival_rate_ns_per_int[i])
            num_ew = self._arrival(self.arrival_rate_ew_per_int[i])
            if num_ns > 0:
                self.ns_queues[i].extend([self.current_step] * num_ns)
            if num_ew > 0:
                self.ew_queues[i].extend([self.current_step] * num_ew)

    def _serve(self) -> None:
        """Serve vehicles based on current phase and topology."""
        for i in range(self.num_intersections):
            row, col = self._index_to_grid(i)
            
            # Check if this is a boundary intersection
            is_top_boundary = (row == 0)
            is_bottom_boundary = (row == self.grid_rows - 1)
            is_left_boundary = (col == 0)
            is_right_boundary = (col == self.grid_cols - 1)
            
            if self.phase[i] == 0:  # NS green
                can_depart = min(self.depart_capacity, len(self.ns_queues[i]))
                if can_depart > 0:
                    departed = []
                    for _ in range(can_depart):
                        if self.ns_queues[i]:  # Safety check
                            departed.append(self.ns_queues[i].pop(0))
                    
                    # Check for valid exits first (deterministic exit logic)
                    should_exit = False
                    if is_top_boundary or is_bottom_boundary:
                        # NS vehicles can exit from top/bottom boundaries during NS green
                        should_exit = True
                    
                    if should_exit:
                        # Vehicles exit the system
                        for enter_step in departed:
                            travel_steps = max(1, self.current_step - enter_step + 1)
                            self.exited_vehicle_times.append(travel_steps * self.step_length)
                        self.episode_throughput += len(departed)
                        self.per_int_throughput[i] += len(departed)
                    else:
                        # Interior intersection: route to vertical neighbors
                        vertical_neighbors = []
                        if row > 0:
                            vertical_neighbors.append(self._grid_to_index(row - 1, col))  # Up
                        if row < self.grid_rows - 1:
                            vertical_neighbors.append(self._grid_to_index(row + 1, col))  # Down
                        
                        if vertical_neighbors:
                            # Route to random vertical neighbor's EW queue
                            target_idx = self.rng.choice(vertical_neighbors)
                            self.ew_queues[target_idx].extend(departed)
                        # Note: Interior intersections should always have neighbors
                        
            else:  # EW green
                can_depart = min(self.depart_capacity, len(self.ew_queues[i]))
                if can_depart > 0:
                    departed = []
                    for _ in range(can_depart):
                        if self.ew_queues[i]:  # Safety check
                            departed.append(self.ew_queues[i].pop(0))
                    
                    # Check for valid exits first (deterministic exit logic)
                    should_exit = False
                    if is_left_boundary or is_right_boundary:
                        # EW vehicles can exit from left/right boundaries during EW green
                        should_exit = True
                    
                    if should_exit:
                        # Vehicles exit the system
                        for enter_step in departed:
                            travel_steps = max(1, self.current_step - enter_step + 1)
                            self.exited_vehicle_times.append(travel_steps * self.step_length)
                        self.episode_throughput += len(departed)
                        self.per_int_throughput[i] += len(departed)
                    else:
                        # Interior intersection: route to horizontal neighbors
                        horizontal_neighbors = []
                        if col > 0:
                            horizontal_neighbors.append(self._grid_to_index(row, col - 1))  # Left
                        if col < self.grid_cols - 1:
                            horizontal_neighbors.append(self._grid_to_index(row, col + 1))  # Right
                        
                        if horizontal_neighbors:
                            # Route to random horizontal neighbor's NS queue
                            target_idx = self.rng.choice(horizontal_neighbors)
                            self.ns_queues[target_idx].extend(departed)
                        # Note: Interior intersections should always have neighbors

    def _apply_actions(self, actions: Dict[str, int]) -> None:
        for i in range(self.num_intersections):
            act = actions.get(self._agent_id(i), 0)
            # 0 keep, 1 switch if min_green satisfied
            if act == 1 and self.time_since_switch[i] >= self.min_green:
                self.phase[i] = 1 - self.phase[i]
                self.time_since_switch[i] = 0
            else:
                self.time_since_switch[i] += 1

    def _observe(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        # Normalization constants for better neural network training
        # Queue lengths normalized by a reasonable max (50), time_since_switch by max_steps
        max_queue_norm = 50.0
        max_time_norm = float(self.max_steps)
        
        # Get global context features
        context = self._get_context_features()
        
        for i in range(self.num_intersections):
            ns_len = float(len(self.ns_queues[i]))
            ew_len = float(len(self.ew_queues[i]))
            phase = float(self.phase[i])
            tss = float(self.time_since_switch[i])
            
            # Normalize observations for better neural network training
            ns_norm = min(ns_len / max_queue_norm, 1.0)  # Clip at 1.0
            ew_norm = min(ew_len / max_queue_norm, 1.0)
            tss_norm = min(tss / max_time_norm, 1.0)
            
            # Base features: [ns_len, ew_len, phase, time_since_switch]
            base_features = [ns_norm, ew_norm, phase, tss_norm]
            
            # Add context features: [time_of_day, global_congestion]
            context_features = [
                context["time_of_day"],
                min(context["global_congestion"] / max_queue_norm, 1.0)  # Normalize congestion
            ]
            
            # Only include neighbor features if neighbor_obs is True (not when using GNN)
            if self.neighbor_obs:
                # Get a neighbor for observation (use right neighbor if exists, else left)
                row, col = self._index_to_grid(i)
                neighbor_idx = None
                if col < self.grid_cols - 1:
                    neighbor_idx = self._grid_to_index(row, col + 1)  # Right neighbor
                elif col > 0:
                    neighbor_idx = self._grid_to_index(row, col - 1)  # Left neighbor
                
                if neighbor_idx is not None:
                    next_ew = float(len(self.ew_queues[neighbor_idx]))
                    next_ew_norm = min(next_ew / max_queue_norm, 1.0)
                    obs_vec = np.array(base_features + context_features + [next_ew_norm], dtype=np.float32)
                else:
                    obs_vec = np.array(base_features + context_features, dtype=np.float32)
            else:
                obs_vec = np.array(base_features + context_features, dtype=np.float32)
            
            obs[self._agent_id(i)] = obs_vec
        return obs

    def _get_context_features(self) -> Dict[str, float]:
        """Get global context features for traffic simulation."""
        # Time of day (normalized 0-1, simulating 24-hour cycle)
        # Use episode progress as proxy for time of day
        time_of_day = (self.current_step / self.max_steps) % 1.0
        
        # Global congestion level (average queue length across all intersections)
        total_queue = sum(len(self.ns_queues[i]) + len(self.ew_queues[i]) 
                         for i in range(self.num_intersections))
        global_congestion = total_queue / max(1, self.num_intersections)
        
        return {
            "time_of_day": time_of_day,
            "global_congestion": global_congestion,
        }

    def _agent_id(self, idx: int) -> str:
        return f"int{idx}"
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.phase = [0 for _ in range(self.num_intersections)]
        self.time_since_switch = [0 for _ in range(self.num_intersections)]
        self.ns_queues = [[] for _ in range(self.num_intersections)]
        self.ew_queues = [[] for _ in range(self.num_intersections)]
        self.exited_vehicle_times = []
        self.episode_throughput = 0
        self.episode_queue_sum = 0.0
        self.episode_queue_steps = 0
        self.per_int_throughput = [0 for _ in range(self.num_intersections)]
        self.per_int_avg_queue_accum = [0.0 for _ in range(self.num_intersections)]
        
        # Initialize episode start time for context
        import time
        self.episode_start_time = time.time()
        
        # Initial arrivals for a warm start
        self._add_arrivals()
        return self._observe()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, float]]:
        # Apply actions and advance environment
        self._apply_actions(actions)
        self._serve()
        self._add_arrivals()

        # Compute rewards and accumulate queue stats
        rewards: Dict[str, float] = {}
        total_queue_len = 0
        
        # Calculate neighbor-aware rewards
        for i in range(self.num_intersections):
            q_len = len(self.ns_queues[i]) + len(self.ew_queues[i])
            total_queue_len += q_len
            
            # Get neighbor queue lengths for spatial awareness
            neighbor_indices = self._get_grid_neighbors(i)
            neighbor_queue_sum = 0.0
            if neighbor_indices:
                for neighbor_idx in neighbor_indices:
                    neighbor_q_len = len(self.ns_queues[neighbor_idx]) + len(self.ew_queues[neighbor_idx])
                    neighbor_queue_sum += neighbor_q_len
                avg_neighbor_queue = neighbor_queue_sum / len(neighbor_indices)
            else:
                avg_neighbor_queue = 0.0
            
            # Neighbor-aware reward: own queue + weighted neighbor congestion
            alpha = 0.3  # Neighbor influence weight
            raw_reward = -(float(q_len) + alpha * avg_neighbor_queue)
            
            # Clip reward to prevent extreme values that cause training instability
            clipped_reward = max(-15.0, min(0.0, raw_reward))  # Expanded range for neighbor component
            rewards[self._agent_id(i)] = clipped_reward
            self.per_int_avg_queue_accum[i] += q_len
        self.episode_queue_sum += total_queue_len / max(1, self.num_intersections)
        self.episode_queue_steps += 1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        obs = self._observe()
        # Per-intersection average queue up to now
        per_int_avg_queue = [
            (self.per_int_avg_queue_accum[i] / self.episode_queue_steps)
            if self.episode_queue_steps else 0.0
            for i in range(self.num_intersections)
        ]

        info: Dict[str, float] = {
            "throughput": float(self.episode_throughput),
            "avg_travel_time": float(np.mean(self.exited_vehicle_times)) if self.exited_vehicle_times else 0.0,
            "avg_queue": float(self.episode_queue_sum / max(1, self.episode_queue_steps)) if self.episode_queue_steps else 0.0,
            # Extra detail for richer dashboards
            "per_int_throughput": self.per_int_throughput.copy(),
            "per_int_avg_queue": per_int_avg_queue,
        }
        return obs, rewards, done, info

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return NxN adjacency matrix for intersections.
        
        Grid topology: 4-neighbor connections (up, down, left, right)
        Note: Self-loops are NOT included here - they are added in GraphConvLayer
        """
        n = self.num_intersections
        adj = np.zeros((n, n), dtype=np.float32)
        
        # Grid topology: connect to 4 neighbors (up, down, left, right)
        # No self-loops here - GraphConvLayer will add them
        for i in range(n):
            # Get grid neighbors
            neighbors = self._get_grid_neighbors(i)
            for neighbor_idx in neighbors:
                adj[i, neighbor_idx] = 1.0
                adj[neighbor_idx, i] = 1.0  # Symmetric
        
        return adj

    def get_node_features(self, use_gnn: bool = False) -> np.ndarray:
        """Return structured node features for GNN: [num_intersections, features_per_node]."""
        max_queue_norm = 50.0
        max_time_norm = float(self.max_steps)
        
        # Get global context features
        context = self._get_context_features()
        
        node_features = []
        for i in range(self.num_intersections):
            ns_len = float(len(self.ns_queues[i]))
            ew_len = float(len(self.ew_queues[i]))
            phase = float(self.phase[i])
            tss = float(self.time_since_switch[i])
            
            ns_norm = min(ns_len / max_queue_norm, 1.0)
            ew_norm = min(ew_len / max_queue_norm, 1.0)
            tss_norm = min(tss / max_time_norm, 1.0)
            
            # Base features: [ns_len, ew_len, phase, time_since_switch]
            base_features = [ns_norm, ew_norm, phase, tss_norm]
            
            # Add context features: [time_of_day, global_congestion]
            context_features = [
                context["time_of_day"],
                min(context["global_congestion"] / max_queue_norm, 1.0)  # Normalize congestion
            ]
            
            # When using GNN, don't include neighbor features - let GNN learn spatial relationships
            if use_gnn or not self.neighbor_obs:
                features = np.array(base_features + context_features, dtype=np.float32)
            else:
                # Only include neighbor features for non-GNN case
                row, col = self._index_to_grid(i)
                neighbor_idx = None
                if col < self.grid_cols - 1:
                    neighbor_idx = self._grid_to_index(row, col + 1)  # Right neighbor
                elif col > 0:
                    neighbor_idx = self._grid_to_index(row, col - 1)  # Left neighbor
                
                if neighbor_idx is not None:
                    next_ew = float(len(self.ew_queues[neighbor_idx]))
                    next_ew_norm = min(next_ew / max_queue_norm, 1.0)
                    features = np.array(base_features + context_features + [next_ew_norm], dtype=np.float32)
                else:
                    features = np.array(base_features + context_features, dtype=np.float32)
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)

    def get_obs_dim(self, use_gnn: bool = False) -> int:
        # Base features: [ns_len, ew_len, phase, time_since_switch] = 4
        # Context features: [time_of_day, global_congestion] = 2
        # Total base + context = 6
        
        if use_gnn:
            return 6  # Base + context, no neighbor features
        
        # For non-GNN case, add neighbor feature if enabled
        return 7 if self.neighbor_obs else 6

    def get_n_actions(self) -> int:
        return 2


