from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


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


class MiniTrafficEnv:
    """Queue-based multi-intersection traffic environment.

    - N intersections in a simple line topology with wrap-around for EW routing
    - Two approaches per intersection: NS (exits network when served) and EW (routes to next NS)
    - Actions per intersection: 0 = keep current phase, 1 = switch (if min_green satisfied)
    - Phase: 0 = NS green, 1 = EW green
    - Observations per agent: [ns_len, ew_len, phase, time_since_switch] (+ neighbor ew_len if enabled)
    - Reward per agent: negative queue length at its intersection: -(ns_len + ew_len)

    Tracking per-vehicle entry/exit times for average travel time.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.rng = np.random.RandomState(self.config.seed)

        self.num_intersections = self.config.num_intersections
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

        # Metrics (global and per-intersection)
        self.exited_vehicle_times: List[float] = []
        self.episode_throughput: int = 0
        self.episode_queue_sum: float = 0.0
        self.episode_queue_steps: int = 0
        self.per_int_throughput: List[int] = [0 for _ in range(self.num_intersections)]
        self.per_int_avg_queue_accum: List[float] = [0.0 for _ in range(self.num_intersections)]

    def seed(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed)

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
        # For each intersection, serve the green approach
        # NS: vehicles exit the system
        # EW: vehicles route to next intersection's NS queue (wrap-around)
        for i in range(self.num_intersections):
            if self.phase[i] == 0:  # NS green
                can_depart = min(self.depart_capacity, len(self.ns_queues[i]))
                if can_depart > 0:
                    departed = []
                    for _ in range(can_depart):
                        if self.ns_queues[i]:  # Safety check
                            departed.append(self.ns_queues[i].pop(0))
                    # These vehicles exit; record travel times
                    for enter_step in departed:
                        travel_steps = max(1, self.current_step - enter_step + 1)  # Ensure positive
                        self.exited_vehicle_times.append(travel_steps * self.step_length)
                    self.episode_throughput += len(departed)
                    self.per_int_throughput[i] += len(departed)
            else:  # EW green
                can_depart = min(self.depart_capacity, len(self.ew_queues[i]))
                if can_depart > 0:
                    departed = []
                    for _ in range(can_depart):
                        if self.ew_queues[i]:  # Safety check
                            departed.append(self.ew_queues[i].pop(0))
                    next_i = (i + 1) % self.num_intersections
                    # Route to next NS queue with same enter_step
                    self.ns_queues[next_i].extend(departed)

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
        
        for i in range(self.num_intersections):
            ns_len = float(len(self.ns_queues[i]))
            ew_len = float(len(self.ew_queues[i]))
            phase = float(self.phase[i])
            tss = float(self.time_since_switch[i])
            
            # Normalize observations for better neural network training
            ns_norm = min(ns_len / max_queue_norm, 1.0)  # Clip at 1.0
            ew_norm = min(ew_len / max_queue_norm, 1.0)
            tss_norm = min(tss / max_time_norm, 1.0)
            
            if self.neighbor_obs:
                next_i = (i + 1) % self.num_intersections
                next_ew = float(len(self.ew_queues[next_i]))
                next_ew_norm = min(next_ew / max_queue_norm, 1.0)
                obs_vec = np.array([ns_norm, ew_norm, phase, tss_norm, next_ew_norm], dtype=np.float32)
            else:
                obs_vec = np.array([ns_norm, ew_norm, phase, tss_norm], dtype=np.float32)
            obs[self._agent_id(i)] = obs_vec
        return obs

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
        for i in range(self.num_intersections):
            q_len = len(self.ns_queues[i]) + len(self.ew_queues[i])
            total_queue_len += q_len
            rewards[self._agent_id(i)] = -float(q_len)
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

    def get_obs_dim(self) -> int:
        return 5 if self.neighbor_obs else 4

    def get_n_actions(self) -> int:
        return 2


