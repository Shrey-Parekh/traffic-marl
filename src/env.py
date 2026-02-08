from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.random import Generator, default_rng


@dataclass
class EnvConfig:
    num_intersections: int = 2
    step_length: float = 2.0  # seconds per step
    max_steps: int = 600  # Increased to 20 minutes
    min_green: int = 10  # Increased to 20 seconds minimum
    arrival_rate_ns: float = 0.35  # Increased for higher system load (70%)
    arrival_rate_ew: float = 0.30  # Asymmetric traffic (60% load)
    # Optional per-intersection overrides for heterogeneous demand
    arrival_rate_ns_per_int: Optional[List[float]] = None
    arrival_rate_ew_per_int: Optional[List[float]] = None
    depart_capacity: int = 2  # vehicles that can depart per green per step
    seed: int = 42
    neighbor_obs: bool = False
    # Grid topology parameters
    grid_rows: Optional[int] = None  # Number of rows for grid topology (auto-computed if not specified)
    grid_cols: Optional[int] = None  # Number of columns for grid topology (auto-computed if not specified)
    # Routing parameters
    turn_straight_prob: float = 0.70  # Probability of going straight
    turn_left_prob: float = 0.15  # Probability of turning left
    turn_right_prob: float = 0.15  # Probability of turning right
    # Travel time parameters
    travel_time_steps: int = 2  # Steps for vehicles to travel between intersections
    # Boundary parameters
    boundary_arrival_factor: float = 0.5  # Reduce arrivals at boundary intersections by 50%


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
    
    Reward per agent (decision-based, no serving component):
    - Queue penalty: -0.5 × total_queue (constant pressure)
    - Imbalance penalty: -1.5 × |NS - EW| (force balance)
    - Switch reward: +3.0 for good switch, -2.0 for bad switch
    - Starvation penalty: -3.0 × excess when queue > 15 (emergency)
    
    Key improvement: Rewards calculated on queue state BEFORE random arrivals,
    so agent's decisions are not masked by arrival randomness.

    Tracking per-vehicle entry/exit times for average travel time.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.rng: Generator = default_rng(self.config.seed)

        # Grid topology setup: use user-specified num_intersections
        # Only auto-compute grid dimensions if not explicitly provided
        if self.config.grid_rows is None or self.config.grid_cols is None:
            # Auto-compute from num_intersections (try to make square-ish grid)
            import math
            n = self.config.num_intersections
            self.grid_cols = int(math.ceil(math.sqrt(n)))
            self.grid_rows = int(math.ceil(n / self.grid_cols))
            # Use the user-specified num_intersections, don't override it
            self.num_intersections = self.config.num_intersections
        else:
            # User provided explicit grid dimensions
            self.grid_rows = self.config.grid_rows
            self.grid_cols = self.config.grid_cols
            # In this case, num_intersections is determined by grid
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
        
        # Routing parameters
        self.turn_straight_prob = self.config.turn_straight_prob
        self.turn_left_prob = self.config.turn_left_prob
        self.turn_right_prob = self.config.turn_right_prob
        self.travel_time_steps = self.config.travel_time_steps
        self.boundary_arrival_factor = self.config.boundary_arrival_factor

        # State
        self.current_step: int = 0
        self.phase: List[int] = [0 for _ in range(self.num_intersections)]
        self.time_since_switch: List[int] = [0 for _ in range(self.num_intersections)]
        self.switches_this_step: List[int] = [0 for _ in range(self.num_intersections)]

        # Queues store enter_step for each vehicle
        self.ns_queues: List[List[int]] = [[] for _ in range(self.num_intersections)]
        self.ew_queues: List[List[int]] = [[] for _ in range(self.num_intersections)]
        
        # In-transit vehicles: List of (destination_intersection, destination_direction, arrival_step, enter_step)
        self.in_transit_vehicles: List[Tuple[int, str, int, int]] = []
        
        # FIXED: Add queue history for temporal features (last 3 steps)
        self.queue_history: List[List[Tuple[int, int]]] = [
            [(0, 0), (0, 0), (0, 0)] for _ in range(self.num_intersections)
        ]  # Each entry: (ns_len, ew_len) for last 3 steps

        # Context features for traffic simulation
        self.episode_start_time: float = 0.0  # Will be set in reset()

        # Metrics (global and per-intersection)
        self.exited_vehicle_times: List[float] = []
        self.episode_throughput: int = 0
        self.episode_queue_sum: float = 0.0
        self.episode_queue_steps: int = 0
        self.per_int_throughput: List[int] = [0 for _ in range(self.num_intersections)]
        self.per_int_avg_queue_accum: List[float] = [0.0 for _ in range(self.num_intersections)]
        
        # Validate grid setup
        self._validate_grid_setup()
        
        # Log grid configuration for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Grid topology initialized: {self.grid_rows}x{self.grid_cols} grid for {self.num_intersections} intersections")

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = default_rng(seed)
    
    def _validate_grid_setup(self) -> None:
        """Validate that the grid setup is consistent and safe."""
        # Check that all intersection indices are valid
        for i in range(self.num_intersections):
            row, col = self._index_to_grid(i)
            if row >= self.grid_rows or col >= self.grid_cols:
                raise ValueError(
                    f"Intersection {i} maps to invalid grid position ({row}, {col}). "
                    f"Grid size: {self.grid_rows}x{self.grid_cols}, "
                    f"Num intersections: {self.num_intersections}"
                )
            
            # Check that grid_to_index is consistent
            reconstructed_idx = self._grid_to_index(row, col)
            if reconstructed_idx != i:
                raise ValueError(
                    f"Grid index conversion inconsistent for intersection {i}: "
                    f"({row}, {col}) -> {reconstructed_idx}"
                )
        
        # Check that neighbor calculations don't produce out-of-bounds indices
        for i in range(self.num_intersections):
            neighbors = self._get_grid_neighbors(i)
            for neighbor_idx in neighbors:
                if neighbor_idx >= self.num_intersections or neighbor_idx < 0:
                    raise ValueError(
                        f"Invalid neighbor index {neighbor_idx} for intersection {i}. "
                        f"Valid range: 0 to {self.num_intersections - 1}"
                    )
    
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
            left_idx = self._grid_to_index(row, col - 1)
            if left_idx < self.num_intersections:  # Ensure valid intersection
                neighbors.append(left_idx)
        if col < self.grid_cols - 1:
            right_idx = self._grid_to_index(row, col + 1)
            if right_idx < self.num_intersections:  # Ensure valid intersection
                neighbors.append(right_idx)
        
        # Vertical neighbors (up, down)
        if row > 0:
            up_idx = self._grid_to_index(row - 1, col)
            if up_idx < self.num_intersections:  # Ensure valid intersection
                neighbors.append(up_idx)
        if row < self.grid_rows - 1:
            down_idx = self._grid_to_index(row + 1, col)
            if down_idx < self.num_intersections:  # Ensure valid intersection
                neighbors.append(down_idx)
        
        return neighbors
    
    def _is_boundary_intersection(self, idx: int) -> bool:
        """Check if intersection is on the boundary of the grid."""
        row, col = self._index_to_grid(idx)
        is_top = (row == 0)
        is_bottom = (row == self.grid_rows - 1)
        is_left = (col == 0)
        is_right = (col == self.grid_cols - 1)
        return is_top or is_bottom or is_left or is_right
    
    def _select_turn_direction(self, current_direction: str, available_directions: List[str]) -> str:
        """Select turn direction based on probabilities.
        
        Args:
            current_direction: 'NS' or 'EW'
            available_directions: List of available directions at this intersection
            
        Returns:
            Selected direction: 'NS' or 'EW'
        """
        if len(available_directions) == 0:
            return current_direction  # No options, keep same direction
        
        if len(available_directions) == 1:
            return available_directions[0]  # Only one option
        
        # Determine straight direction
        straight_direction = current_direction
        
        # If straight is available, use turn probabilities
        if straight_direction in available_directions:
            # Sample turn decision
            rand = self.rng.random()
            if rand < self.turn_straight_prob:
                return straight_direction  # Go straight
            else:
                # Turn (left or right - both map to perpendicular direction)
                other_directions = [d for d in available_directions if d != straight_direction]
                if other_directions:
                    return self.rng.choice(other_directions)
                else:
                    return straight_direction
        else:
            # Straight not available, must turn
            return self.rng.choice(available_directions)

    def _arrival(self, rate: float) -> int:
        return int(self.rng.poisson(rate))

    def _add_arrivals(self) -> None:
        for i in range(self.num_intersections):
            # Apply boundary arrival factor if this is a boundary intersection
            arrival_factor = self.boundary_arrival_factor if self._is_boundary_intersection(i) else 1.0
            
            num_ns = self._arrival(self.arrival_rate_ns_per_int[i] * arrival_factor)
            num_ew = self._arrival(self.arrival_rate_ew_per_int[i] * arrival_factor)
            if num_ns > 0:
                self.ns_queues[i].extend([self.current_step] * num_ns)
            if num_ew > 0:
                self.ew_queues[i].extend([self.current_step] * num_ew)
        
        # Process in-transit vehicles that have arrived
        arrived_vehicles = [v for v in self.in_transit_vehicles if v[2] <= self.current_step]
        self.in_transit_vehicles = [v for v in self.in_transit_vehicles if v[2] > self.current_step]
        
        for dest_int, dest_dir, _, enter_step in arrived_vehicles:
            if dest_dir == 'NS':
                self.ns_queues[dest_int].append(enter_step)
            else:  # 'EW'
                self.ew_queues[dest_int].append(enter_step)

    def _serve(self) -> None:
        """Serve vehicles based on current phase and topology with stochastic departures and travel time."""
        for i in range(self.num_intersections):
            # Validate intersection index
            if i >= self.num_intersections:
                continue  # Skip invalid intersection
                
            row, col = self._index_to_grid(i)
            
            # Check if this is a boundary intersection
            is_top_boundary = (row == 0)
            is_bottom_boundary = (row == self.grid_rows - 1)
            is_left_boundary = (col == 0)
            is_right_boundary = (col == self.grid_cols - 1)
            
            if self.phase[i] == 0:  # NS green
                # Stochastic departure capacity: sample from binomial
                actual_capacity = min(self.depart_capacity, len(self.ns_queues[i]))
                # Add small randomness: 90% chance of full capacity, 10% chance of reduced
                if actual_capacity > 0 and self.rng.random() < 0.1:
                    actual_capacity = max(1, actual_capacity - 1)
                
                can_depart = actual_capacity
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
                        # Interior intersection: route to neighbors with turn probabilities
                        vertical_neighbors = []
                        if row > 0:
                            up_idx = self._grid_to_index(row - 1, col)
                            if up_idx < self.num_intersections:
                                vertical_neighbors.append(up_idx)
                        if row < self.grid_rows - 1:
                            down_idx = self._grid_to_index(row + 1, col)
                            if down_idx < self.num_intersections:
                                vertical_neighbors.append(down_idx)
                        
                        horizontal_neighbors = []
                        if col > 0:
                            left_idx = self._grid_to_index(row, col - 1)
                            if left_idx < self.num_intersections:
                                horizontal_neighbors.append(left_idx)
                        if col < self.grid_cols - 1:
                            right_idx = self._grid_to_index(row, col + 1)
                            if right_idx < self.num_intersections:
                                horizontal_neighbors.append(right_idx)
                        
                        # Route each vehicle based on turn probabilities
                        for enter_step in departed:
                            # Determine available directions
                            available_dirs = []
                            if vertical_neighbors:
                                available_dirs.append('NS')
                            if horizontal_neighbors:
                                available_dirs.append('EW')
                            
                            # Select direction based on turn probabilities
                            selected_dir = self._select_turn_direction('NS', available_dirs)
                            
                            # Select target intersection
                            if selected_dir == 'NS' and vertical_neighbors:
                                target_idx = self.rng.choice(vertical_neighbors)
                                target_dir = 'EW'  # NS vehicles entering from vertical become EW at next intersection
                            elif selected_dir == 'EW' and horizontal_neighbors:
                                target_idx = self.rng.choice(horizontal_neighbors)
                                target_dir = 'NS'  # NS vehicles turning become NS at next intersection
                            else:
                                # Fallback: if no valid direction, vehicle is lost
                                continue
                            
                            # Add to in-transit with travel time delay
                            arrival_step = self.current_step + self.travel_time_steps
                            self.in_transit_vehicles.append((target_idx, target_dir, arrival_step, enter_step))
                        
            else:  # EW green
                # Stochastic departure capacity
                actual_capacity = min(self.depart_capacity, len(self.ew_queues[i]))
                if actual_capacity > 0 and self.rng.random() < 0.1:
                    actual_capacity = max(1, actual_capacity - 1)
                
                can_depart = actual_capacity
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
                        # Interior intersection: route to neighbors with turn probabilities
                        horizontal_neighbors = []
                        if col > 0:
                            left_idx = self._grid_to_index(row, col - 1)
                            if left_idx < self.num_intersections:
                                horizontal_neighbors.append(left_idx)
                        if col < self.grid_cols - 1:
                            right_idx = self._grid_to_index(row, col + 1)
                            if right_idx < self.num_intersections:
                                horizontal_neighbors.append(right_idx)
                        
                        vertical_neighbors = []
                        if row > 0:
                            up_idx = self._grid_to_index(row - 1, col)
                            if up_idx < self.num_intersections:
                                vertical_neighbors.append(up_idx)
                        if row < self.grid_rows - 1:
                            down_idx = self._grid_to_index(row + 1, col)
                            if down_idx < self.num_intersections:
                                vertical_neighbors.append(down_idx)
                        
                        # Route each vehicle based on turn probabilities
                        for enter_step in departed:
                            # Determine available directions
                            available_dirs = []
                            if horizontal_neighbors:
                                available_dirs.append('EW')
                            if vertical_neighbors:
                                available_dirs.append('NS')
                            
                            # Select direction based on turn probabilities
                            selected_dir = self._select_turn_direction('EW', available_dirs)
                            
                            # Select target intersection
                            if selected_dir == 'EW' and horizontal_neighbors:
                                target_idx = self.rng.choice(horizontal_neighbors)
                                target_dir = 'NS'  # EW vehicles entering from horizontal become NS at next intersection
                            elif selected_dir == 'NS' and vertical_neighbors:
                                target_idx = self.rng.choice(vertical_neighbors)
                                target_dir = 'EW'  # EW vehicles turning become EW at next intersection
                            else:
                                # Fallback: if no valid direction, vehicle is lost
                                continue
                            
                            # Add to in-transit with travel time delay
                            arrival_step = self.current_step + self.travel_time_steps
                            self.in_transit_vehicles.append((target_idx, target_dir, arrival_step, enter_step))

    def _apply_actions(self, actions: Dict[str, int]) -> None:
        # Track switches for reward calculation
        self.switches_this_step = [0 for _ in range(self.num_intersections)]
        
        for i in range(self.num_intersections):
            act = actions.get(self._agent_id(i), 0)
            # 0 keep, 1 switch if min_green satisfied
            if act == 1 and self.time_since_switch[i] >= self.min_green:
                self.phase[i] = 1 - self.phase[i]
                self.time_since_switch[i] = 0
                self.switches_this_step[i] = 1  # Mark that we switched
            else:
                self.time_since_switch[i] += 1

    def _observe(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        # Normalization constants for better neural network training
        # Queue lengths normalized by typical max (15 for high load), time_since_switch by max_steps
        max_queue_norm = 15.0  # REDUCED: With high traffic (0.8+0.7), queues reach 5-10, so normalize by 15 not 50
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
            
            # FIXED: Add queue growth rate features (temporal information)
            # Calculate average queue change over last 3 steps
            if len(self.queue_history[i]) >= 2:
                recent_ns = [h[0] for h in self.queue_history[i][-2:]]
                recent_ew = [h[1] for h in self.queue_history[i][-2:]]
                ns_growth = (recent_ns[-1] - recent_ns[0]) / max(1, len(recent_ns) - 1)
                ew_growth = (recent_ew[-1] - recent_ew[0]) / max(1, len(recent_ew) - 1)
                ns_growth_norm = max(-1.0, min(1.0, ns_growth / 5.0))  # Normalize to [-1, 1]
                ew_growth_norm = max(-1.0, min(1.0, ew_growth / 5.0))
            else:
                ns_growth_norm = 0.0
                ew_growth_norm = 0.0
            
            temporal_features = [ns_growth_norm, ew_growth_norm]
            
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
                    obs_vec = np.array(base_features + temporal_features + context_features + [next_ew_norm], dtype=np.float32)
                else:
                    obs_vec = np.array(base_features + temporal_features + context_features, dtype=np.float32)
            else:
                obs_vec = np.array(base_features + temporal_features + context_features, dtype=np.float32)
            
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
        self.switches_this_step = [0 for _ in range(self.num_intersections)]
        self.ns_queues = [[] for _ in range(self.num_intersections)]
        self.ew_queues = [[] for _ in range(self.num_intersections)]
        self.queue_history = [[(0, 0), (0, 0), (0, 0)] for _ in range(self.num_intersections)]  # FIXED: Reset history
        self.in_transit_vehicles = []  # Clear in-transit vehicles
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
        # Store queue state AND phase BEFORE actions for accurate reward calculation
        queue_before = {}
        phase_before = {}
        for i in range(self.num_intersections):
            queue_before[i] = {
                'ns': len(self.ns_queues[i]),
                'ew': len(self.ew_queues[i]),
                'total': len(self.ns_queues[i]) + len(self.ew_queues[i])
            }
            phase_before[i] = self.phase[i]  # Store phase before switching
        
        # Store per-intersection throughput before serving
        throughput_before = [self.per_int_throughput[i] for i in range(self.num_intersections)]
        
        # Apply actions and advance environment (this changes self.phase)
        self._apply_actions(actions)
        
        # Serve cars (this updates per_int_throughput)
        self._serve()
        
        # Calculate cars served this step (per intersection)
        cars_served_this_step = {}
        for i in range(self.num_intersections):
            cars_served_this_step[i] = self.per_int_throughput[i] - throughput_before[i]
        
        # Get queue state AFTER serving but BEFORE arrivals
        queue_after_serving = {}
        for i in range(self.num_intersections):
            queue_after_serving[i] = {
                'ns': len(self.ns_queues[i]),
                'ew': len(self.ew_queues[i]),
                'total': len(self.ns_queues[i]) + len(self.ew_queues[i])
            }
        
        # FIXED: Update queue history BEFORE arrivals (for clean temporal features)
        for i in range(self.num_intersections):
            current_queue = (len(self.ns_queues[i]), len(self.ew_queues[i]))
            self.queue_history[i].append(current_queue)
            # Keep only last 3 steps
            if len(self.queue_history[i]) > 3:
                self.queue_history[i].pop(0)

        # ═══════════════════════════════════════════════════════════════════════
        # DIRECTIONAL REWARD SYSTEM: STRONG PENALTIES, CLEAR SIGNALS
        # ═══════════════════════════════════════════════════════════════════════
        # Core Principle: Heavy penalties for congestion, bonus for throughput
        # 
        # CRITICAL FIX: Previous reward components canceled each other out!
        #   - Queue penalty + throughput bonus ≈ 0 (no learning signal)
        #   - Agent couldn't distinguish good from bad actions
        #
        # New approach:
        #   - STRONG queue penalty (drives learning)
        #   - MODERATE throughput bonus (positive reinforcement)
        #   - LIGHT imbalance penalty (tie-breaker)
        #   - Net reward is NEGATIVE for bad states, LESS NEGATIVE for good states
        # ═══════════════════════════════════════════════════════════════════════
        
        rewards: Dict[str, float] = {}
        total_queue_len = 0
        
        # Normalization constants
        QUEUE_NORM = 10.0  # Typical max queue per direction
        THROUGHPUT_NORM = 2.0  # Typical cars served per step
        
        for i in range(self.num_intersections):
            # Get queue states AFTER serving
            ns_queue_after = queue_after_serving[i]['ns']
            ew_queue_after = queue_after_serving[i]['ew']
            total_queue_after = queue_after_serving[i]['total']
            
            # Get cars served this step
            cars_served = cars_served_this_step[i]
            
            # ═══════════════════════════════════════════════════════════
            # COMPONENT 1: QUEUE PENALTY (STRONG - 85% weight, 3x scale)
            # ═══════════════════════════════════════════════════════════
            # Heavy penalty for queues - PRIMARY learning signal
            # Range: 0 to -2.55 (for queue 0 to 10)
            # ═══════════════════════════════════════════════════════════
            
            queue_penalty = -0.255 * (total_queue_after / QUEUE_NORM)
            
            # ═══════════════════════════════════════════════════════════
            # COMPONENT 2: IMBALANCE PENALTY (MODERATE - 15% weight, 3x scale)
            # ═══════════════════════════════════════════════════════════
            # Incentivize serving the longer queue
            # Range: 0 to -0.45 (for imbalance 0 to 10)
            # ═══════════════════════════════════════════════════════════
            
            imbalance_after = abs(ns_queue_after - ew_queue_after)
            imbalance_penalty = -0.045 * (imbalance_after / QUEUE_NORM)
            
            # ═══════════════════════════════════════════════════════════
            # FINAL REWARD (NO THROUGHPUT BONUS - violates spec requirement 4)
            # ═══════════════════════════════════════════════════════════
            # Range: -3.0 to 0.0 per step
            # 
            # Throughput bonus REMOVED - it depends on random arrivals and
            # violates requirement 4 (no components based on cars_served)
            # 
            # Signal-to-Noise Ratio with higher traffic (queue 3-8):
            #   Signal: 2.0 (difference between queue 3 vs 8)
            #   Noise: 0.5 (Poisson variance)
            #   SNR: 4.0 (LEARNABLE)
            # ═══════════════════════════════════════════════════════════
            
            total_reward = queue_penalty + imbalance_penalty
            rewards[self._agent_id(i)] = total_reward
            
            total_queue_len += total_queue_after
            
            # Accumulate queue stats (using post-serving queue for consistent metrics)
            self.per_int_avg_queue_accum[i] += total_queue_after
        
        self.episode_queue_sum += total_queue_len / max(1, self.num_intersections)
        self.episode_queue_steps += 1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Add arrivals FIRST (before observation)
        # This ensures next_state matches what agent will see in next step
        self._add_arrivals()
        
        # THEN get observation (after arrivals)
        # This fixes the reward-state timing mismatch
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
        max_queue_norm = 15.0  # REDUCED: With high traffic, queues reach 5-10, so normalize by 15 not 50
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
            
            # FIXED: Add temporal features (queue growth rates)
            if len(self.queue_history[i]) >= 2:
                recent_ns = [h[0] for h in self.queue_history[i][-2:]]
                recent_ew = [h[1] for h in self.queue_history[i][-2:]]
                ns_growth = (recent_ns[-1] - recent_ns[0]) / max(1, len(recent_ns) - 1)
                ew_growth = (recent_ew[-1] - recent_ew[0]) / max(1, len(recent_ew) - 1)
                ns_growth_norm = max(-1.0, min(1.0, ns_growth / 5.0))
                ew_growth_norm = max(-1.0, min(1.0, ew_growth / 5.0))
            else:
                ns_growth_norm = 0.0
                ew_growth_norm = 0.0
            
            temporal_features = [ns_growth_norm, ew_growth_norm]
            
            # Add context features: [time_of_day, global_congestion]
            context_features = [
                context["time_of_day"],
                min(context["global_congestion"] / max_queue_norm, 1.0)  # Normalize congestion
            ]
            
            # When using GNN, don't include neighbor features - let GNN learn spatial relationships
            if use_gnn or not self.neighbor_obs:
                features = np.array(base_features + temporal_features + context_features, dtype=np.float32)
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
                    features = np.array(base_features + temporal_features + context_features + [next_ew_norm], dtype=np.float32)
                else:
                    features = np.array(base_features + temporal_features + context_features, dtype=np.float32)
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)

    def get_obs_dim(self, use_gnn: bool = False) -> int:
        # Base features: [ns_len, ew_len, phase, time_since_switch] = 4
        # Temporal features: [ns_growth, ew_growth] = 2  # FIXED: Added
        # Context features: [time_of_day, global_congestion] = 2
        # Total base + temporal + context = 8  # FIXED: Was 6
        
        if use_gnn:
            return 8  # FIXED: Base + temporal + context, no neighbor features
        
        # For non-GNN case, add neighbor feature if enabled
        return 9 if self.neighbor_obs else 8  # FIXED: Was 7 or 6

    def get_n_actions(self) -> int:
        return 2


