"""
SUMO-based Indian Mixed Traffic Environment for Multi-Agent RL Traffic Signal Control.

Features:
- 4 Indian vehicle classes with PCU weighting
- Peak hour asymmetry (morning/evening/uniform scenarios)
- Non-lane-based flow (two-wheeler lane-splitting)
- 3-phase signal logic (NS_GREEN, ALL_RED_CLEARANCE, EW_GREEN)
- 15-feature observation space per intersection
- 3-action space (keep_phase, switch_phase, force_clearance)
- PCU-based reward function
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import sys

import numpy as np

# Add SUMO tools to path for libsumo
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

# Use libsumo for in-process simulation (3-6x faster than TraCI)
# Falls back to TraCI if libsumo not available
try:
    import libsumo as traci
    USING_LIBSUMO = True
except ImportError:
    import traci
    USING_LIBSUMO = False

from .config import (
    VEHICLE_CLASSES,
    PEAK_HOUR_CONFIG,
    SUMO_CONFIG,
    SCENARIOS,
    OBS_FEATURES_PER_AGENT,
    REWARD_CONFIG,
    INJECTION_CONFIG,
)


@dataclass
class VehicleClass:
    """Vehicle class definition with PCU and arrival weight."""
    name: str
    pcu: float
    arrival_weight: float
    vtype: str
    service_rate: int


class MixedTrafficQueue:
    """Per-lane queue tracking with PCU calculation."""
    
    def __init__(self, lane_id: str):
        self.lane_id = lane_id
        self.vehicles: List[str] = []
        self.vehicle_types: Dict[str, str] = {}
    
    def update(self, vehicle_ids: List[str], traci_conn):
        """Update queue with current vehicles on lane."""
        self.vehicles = vehicle_ids
        self.vehicle_types = {}
        for veh_id in vehicle_ids:
            try:
                vtype = traci_conn.vehicle.getTypeID(veh_id)
                self.vehicle_types[veh_id] = vtype
            except traci.exceptions.TraCIException:
                pass
    
    def get_pcu(self) -> float:
        """Calculate total PCU for this lane."""
        total_pcu = 0.0
        for veh_id, vtype in self.vehicle_types.items():
            for vclass_name, vclass_data in VEHICLE_CLASSES.items():
                if vclass_data["vtype"] == vtype:
                    total_pcu += vclass_data["pcu"]
                    break
        return total_pcu
    
    def get_count(self) -> int:
        """Get raw vehicle count."""
        return len(self.vehicles)
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count per vehicle class."""
        counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        for veh_id, vtype in self.vehicle_types.items():
            if vtype in counts:
                counts[vtype] += 1
        return counts


class PuneSUMOEnv:
    """
    SUMO-based environment for Pune 3x3 grid with Indian mixed traffic.
    
    Observation space: (n_intersections, 15) float array
    Action space: 3 discrete actions per intersection
    """
    
    def __init__(self, config: Dict):
        """Initialize SUMO environment."""
        self.n_intersections = config.get("n_intersections", 9)
        self.n_agents = self.n_intersections  # Alias for consistency
        self.scenario = config.get("scenario", "uniform")
        self.render = config.get("render", False)
        self.seed = config.get("seed", 42)
        self.max_steps = config.get("max_steps", 3600)
        self.use_global_reward = config.get("use_global_reward", True)
        
        self.sumo_config_file = SUMO_CONFIG["config_file"]
        self.step_length = SUMO_CONFIG["step_length"]
        self.min_green_steps = SUMO_CONFIG["min_green_steps"]
        self.clearance_steps = SUMO_CONFIG["clearance_steps"]
        self.lane_split_probability = SUMO_CONFIG["lane_split_probability"]
        self.lane_split_min_queue = SUMO_CONFIG["lane_split_min_queue"]
        
        self.tl_ids = [f"n{row}{col}" for row in range(3) for col in range(3)][:self.n_intersections]
        self.current_phases = [0] * self.n_intersections
        self.steps_since_switch = [0] * self.n_intersections
        self.queues: Dict[str, Dict[str, MixedTrafficQueue]] = {}
        self.current_step = 0
        self.traci_conn = None
        self.peak_config = PEAK_HOUR_CONFIG[self.scenario]
        
        # Initialize prev_pcu tracking for reward improvement signal
        self.prev_pcu = {}
        self._prev_network_pcu = None
        self._before_serving_pcu = {}
        
        # Track turning movements for paper documentation
        self.turning_counts = {
            "straight": 0,
            "right_turn": 0,
            "left_turn": 0,
            "u_turn": 0
        }
        
        self.vehicle_classes = [
            VehicleClass(
                name=name,
                pcu=data["pcu"],
                arrival_weight=data["arrival_weight"],
                vtype=data["vtype"],
                service_rate=data["service_rate"]
            )
            for name, data in VEHICLE_CLASSES.items()
        ]
        
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix for 3x3 grid topology."""
        adj = np.zeros((self.n_intersections, self.n_intersections), dtype=np.float32)
        for row in range(3):
            for col in range(2):
                idx = row * 3 + col
                adj[idx, idx + 1] = 1.0
                adj[idx + 1, idx] = 1.0
        for row in range(2):
            for col in range(3):
                idx = row * 3 + col
                adj[idx, idx + 3] = 1.0
                adj[idx + 3, idx] = 1.0
        return adj
    
    def _get_neighbor_indices(self, intersection_idx: int) -> list:
        """Returns list of neighboring intersection indices for a given intersection.
        Based on 3x3 grid topology — each node connects to adjacent nodes only.
        Center node (4) has 4 neighbors, corner nodes have 2, edge nodes have 3."""
        # 3x3 grid adjacency
        # Node indices: 0=n00, 1=n01, 2=n02, 3=n10, 4=n11, 5=n12,
        #               6=n20, 7=n21, 8=n22
        adjacency = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        }
        return adjacency.get(intersection_idx, [])
    
    def _start_sumo(self):
        """Start SUMO simulation with retry logic."""
        # libsumo doesn't support GUI mode
        if USING_LIBSUMO and self.render:
            print("Warning: libsumo doesn't support GUI. Disabling render mode.")
            self.render = False
        
        sumo_binary = "sumo-gui" if self.render else "sumo"
        sumo_cmd = [
            sumo_binary, "-c", self.sumo_config_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true", "--no-step-log", "true",
            "--time-to-teleport", "-1", "--seed", str(self.seed),
        ]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.traci_conn is not None:
                    try:
                        self.traci_conn.close()
                    except:
                        pass
                traci.start(sumo_cmd)
                self.traci_conn = traci
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Failed to start SUMO: {e}") from e
    
    def _initialize_queues(self):
        """Initialize queue tracking for all intersections."""
        self.queues = {}
        for tl_id in self.tl_ids:
            self.queues[tl_id] = {}
            try:
                controlled_lanes = self.traci_conn.trafficlight.getControlledLanes(tl_id)
                for lane_id in set(controlled_lanes):
                    self.queues[tl_id][lane_id] = MixedTrafficQueue(lane_id)
            except traci.exceptions.TraCIException:
                pass
    
    def _set_sumo_phase(self, tl_id: str, phase_idx: int):
        """Set SUMO traffic light phase."""
        try:
            # Get the number of controlled lanes for this traffic light
            controlled_lanes = self.traci_conn.trafficlight.getControlledLanes(tl_id)
            num_signals = len(controlled_lanes)
            
            # Create phase strings based on number of signals
            # Assuming equal split between NS and EW directions
            half = num_signals // 2
            
            if phase_idx == 0:  # NS_GREEN
                phase_str = "G" * half + "r" * (num_signals - half)
            elif phase_idx == 1:  # ALL_RED_CLEARANCE
                phase_str = "r" * num_signals
            elif phase_idx == 2:  # EW_GREEN
                phase_str = "r" * half + "G" * (num_signals - half)
            else:
                phase_str = "G" * num_signals  # Default to all green
            
            self.traci_conn.trafficlight.setRedYellowGreenState(tl_id, phase_str)
        except traci.exceptions.TraCIException:
            pass
    
    def _update_queues(self):
        """Update all queue states from SUMO."""
        for tl_id in self.tl_ids:
            if tl_id not in self.queues:
                continue
            for lane_id, queue in self.queues[tl_id].items():
                try:
                    vehicle_ids = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
                    queue.update(vehicle_ids, self.traci_conn)
                except traci.exceptions.TraCIException:
                    pass
    
    def _get_intersection_queues(self, tl_id: str) -> Tuple[float, float, int, int, Dict[str, int], Dict[str, int]]:
        """Extract NS and EW queue information for an intersection."""
        ns_pcu = 0.0
        ew_pcu = 0.0
        ns_count = 0
        ew_count = 0
        ns_class_counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        ew_class_counts = {vclass_data["vtype"]: 0 for vclass_data in VEHICLE_CLASSES.values()}
        
        if tl_id not in self.queues:
            return ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts
        
        for lane_id, queue in self.queues[tl_id].items():
            is_ns = any(d in lane_id for d in ["_0", "_2"])
            pcu = queue.get_pcu()
            count = queue.get_count()
            class_counts = queue.get_class_counts()
            
            if is_ns:
                ns_pcu += pcu
                ns_count += count
                for vtype, cnt in class_counts.items():
                    ns_class_counts[vtype] += cnt
            else:
                ew_pcu += pcu
                ew_count += count
                for vtype, cnt in class_counts.items():
                    ew_class_counts[vtype] += cnt
        
        return ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts
    
    def _apply_lane_splitting(self):
        """Simulate two-wheeler lane-splitting behavior."""
        for tl_id in self.tl_ids:
            if tl_id not in self.queues:
                continue
            for lane_id, queue in self.queues[tl_id].items():
                if queue.get_count() < self.lane_split_min_queue:
                    continue
                two_wheeler_vtype = VEHICLE_CLASSES["TWO_WHEELER"]["vtype"]
                for veh_id, vtype in queue.vehicle_types.items():
                    if vtype == two_wheeler_vtype:
                        if random.random() < self.lane_split_probability:
                            try:
                                current_speed = self.traci_conn.vehicle.getSpeed(veh_id)
                                self.traci_conn.vehicle.setSpeed(veh_id, min(current_speed + 2.0, 15.0))
                            except traci.exceptions.TraCIException:
                                pass
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self._start_sumo()
        self.current_step = 0
        self.current_phases = [0] * self.n_intersections
        self.steps_since_switch = [0] * self.n_intersections
        self.turning_counts = {"straight": 0, "right_turn": 0, "left_turn": 0, "u_turn": 0}
        
        # Initialize metrics tracking
        self.departed_vehicles = set()
        self.arrived_vehicles = set()
        self.vehicle_travel_times = {}
        
        # Initialize prev_pcu tracking for reward improvement signal
        self.prev_pcu = {}
        self._prev_network_pcu = None
        self._before_serving_pcu = {}
        
        self._initialize_queues()
        for tl_id in self.tl_ids:
            self._set_sumo_phase(tl_id, 0)
        for _ in range(10):
            try:
                self.traci_conn.simulationStep()
            except traci.exceptions.TraCIException:
                pass
        self._update_queues()
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for all intersections.
        Returns (n_intersections, 22) array:
        0-14: self observation (unchanged)
        15-20: neighbor aggregate observations
        21: action mask (can_switch)
        """
        obs = np.zeros((self.n_intersections, OBS_FEATURES_PER_AGENT), dtype=np.float32)
        scenario_flag = 0.0
        if self.scenario == "morning_peak":
            scenario_flag = 1.0
        elif self.scenario == "evening_peak":
            scenario_flag = 2.0
        
        MAX_PCU = 30.0  # Use reward_queue_norm as max PCU reference
        
        for i, tl_id in enumerate(self.tl_ids):
            # Self features (0-14)
            ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts = self._get_intersection_queues(tl_id)
            obs[i, 0] = ns_count
            obs[i, 1] = ew_count
            obs[i, 2] = ns_pcu
            obs[i, 3] = ew_pcu
            obs[i, 4] = self.current_phases[i]
            obs[i, 5] = self.steps_since_switch[i]
            vtype_list = [vc["vtype"] for vc in VEHICLE_CLASSES.values()]
            for j, vtype in enumerate(vtype_list):
                obs[i, 6 + j] = ns_class_counts.get(vtype, 0)
                obs[i, 10 + j] = ew_class_counts.get(vtype, 0)
            obs[i, 14] = scenario_flag
            
            # Neighbor features (15-20)
            neighbors = self._get_neighbor_indices(i)
            if len(neighbors) == 0:
                obs[i, 15:21] = 0.0
            else:
                neighbor_ns_pcu = []
                neighbor_ew_pcu = []
                neighbor_phases = []
                for n_idx in neighbors:
                    n_tl_id = self.tl_ids[n_idx]
                    n_ns_pcu, n_ew_pcu, _, _, _, _ = self._get_intersection_queues(n_tl_id)
                    neighbor_ns_pcu.append(n_ns_pcu)
                    neighbor_ew_pcu.append(n_ew_pcu)
                    neighbor_phases.append(self.current_phases[n_idx])
                
                obs[i, 15] = np.mean(neighbor_ns_pcu) / MAX_PCU  # mean NS PCU neighbors
                obs[i, 16] = np.mean(neighbor_ew_pcu) / MAX_PCU  # mean EW PCU neighbors
                obs[i, 17] = np.max(neighbor_ns_pcu) / MAX_PCU   # max NS PCU neighbors
                obs[i, 18] = np.max(neighbor_ew_pcu) / MAX_PCU   # max EW PCU neighbors
                obs[i, 19] = sum(1 for p in neighbor_phases if p == 0) / max(len(neighbors), 1)  # NS_GREEN count
                obs[i, 20] = sum(1 for p in neighbor_phases if p == 2) / max(len(neighbors), 1)  # EW_GREEN count
            
            # Action mask (21)
            can_switch = 1.0 if self.steps_since_switch[i] >= self.min_green_steps else 0.0
            obs[i, 21] = can_switch
        
        return obs
    
    def _inject_vehicles(self):
        """
        Dynamically injects vehicles using edge-based routes.
        Applies peak hour multipliers from PEAK_HOUR_CONFIG.
        Uses traci.vehicle.addFull() with edge lists instead of route IDs.
        """
        # Define routes as edge lists (not route IDs)
        straight_routes = [
            # NS routes
            (["e_n00_n10", "e_n10_n20"], "NS"),
            (["e_n20_n10", "e_n10_n00"], "NS"),
            (["e_n01_n11", "e_n11_n21"], "NS"),
            (["e_n21_n11", "e_n11_n01"], "NS"),
            (["e_n02_n12", "e_n12_n22"], "NS"),
            (["e_n22_n12", "e_n12_n02"], "NS"),
            # EW routes
            (["e_n00_n01", "e_n01_n02"], "EW"),
            (["e_n02_n01", "e_n01_n00"], "EW"),
            (["e_n10_n11", "e_n11_n12"], "EW"),
            (["e_n12_n11", "e_n11_n10"], "EW"),
            (["e_n20_n21", "e_n21_n22"], "EW"),
            (["e_n22_n21", "e_n21_n20"], "EW"),
        ]
        
        # Get multipliers
        ns_mult = self.peak_config["NS_multiplier"]
        ew_mult = self.peak_config["EW_multiplier"]
        
        # Inject vehicles
        for edges, direction in straight_routes:
            mult = ns_mult if direction == "NS" else ew_mult
            
            for vclass_name, vclass_data in VEHICLE_CLASSES.items():
                vtype = vclass_data["vtype"]
                weight = vclass_data["arrival_weight"]
                
                # Use calibrated base rate from INJECTION_CONFIG
                base_rate = weight * INJECTION_CONFIG["base_rate"]
                rate = base_rate * mult
                num = np.random.poisson(rate)
                
                for i in range(num):
                    vid = f"{vtype}_s{self.current_step}_{i}_{edges[0]}"
                    try:
                        if vtype == "pedestrian_group":
                            self.traci_conn.vehicle.addFull(
                                vehID=vid,
                                routeID="",
                                typeID=vtype,
                                depart="now",
                                departLane="0",
                                departPos="0",
                                departSpeed="0",
                                arrivalLane="current",
                                arrivalPos="max",
                                arrivalSpeed="current",
                                fromTaz="",
                                toTaz="",
                                line="",
                                personCapacity=0,
                                personNumber=0
                            )
                            self.traci_conn.vehicle.setRoute(vid, edges)
                        else:
                            self.traci_conn.vehicle.addFull(
                                vehID=vid,
                                routeID="",
                                typeID=vtype,
                                depart="now",
                                departLane="random",
                                departPos="base",
                                departSpeed="random",
                                arrivalLane="current",
                                arrivalPos="max",
                                arrivalSpeed="current",
                                fromTaz="",
                                toTaz="",
                                line="",
                                personCapacity=0,
                                personNumber=0
                            )
                            self.traci_conn.vehicle.setRoute(vid, edges)
                        self.turning_counts["straight"] += 1
                    except traci.exceptions.TraCIException as e:
                        # Log first error only to avoid spam
                        if self.current_step == 0 and i == 0:
                            print(f"Warning: Failed to add vehicle {vid}: {e}")
                        pass

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """Execute one step in the environment."""
        # Capture before-serving state for reward calculation
        self._before_serving_pcu = {}
        self._before_phases = []
        for i, tl_id in enumerate(self.tl_ids):
            ns_pcu, ew_pcu, _, _, _, _ = self._get_intersection_queues(tl_id)
            self._before_serving_pcu[i] = {
                "NS": ns_pcu,
                "EW": ew_pcu,
            }
            self._before_phases.append(self.current_phases[i])
        
        # Inject vehicles before applying actions
        self._inject_vehicles()
        
        # Track departed and arrived vehicles
        self._track_vehicle_metrics()
        
        self._apply_action(actions)
        self._apply_lane_splitting()
        try:
            self.traci_conn.simulationStep()
        except traci.exceptions.TraCIException:
            obs = self._get_observation()
            rewards = [0.0] * self.n_intersections
            info = self._get_info()
            return obs, rewards, True, info
        
        self.current_step += 1
        for i in range(self.n_intersections):
            self.steps_since_switch[i] += 1
        self._update_queues()
        obs = self._get_observation()
        rewards = self._calculate_rewards(actions)
        done = self.current_step >= self.max_steps
        info = self._get_info()
        return obs, rewards, done, info
    
    def _apply_action(self, actions: List[int]):
        """Apply actions to traffic lights."""
        for i, (tl_id, action) in enumerate(zip(self.tl_ids, actions)):
            current_phase = self.current_phases[i]
            if action == 0:
                pass
            elif action == 1:
                if self.steps_since_switch[i] >= self.min_green_steps:
                    if current_phase == 0:
                        self.current_phases[i] = 1
                        self.steps_since_switch[i] = 0
                    elif current_phase == 1:
                        if self.steps_since_switch[i] >= self.clearance_steps:
                            self.current_phases[i] = 2
                            self.steps_since_switch[i] = 0
                    elif current_phase == 2:
                        self.current_phases[i] = 1
                        self.steps_since_switch[i] = 0
                    self._set_sumo_phase(tl_id, self.current_phases[i])
            elif action == 2:
                self.current_phases[i] = 1
                self.steps_since_switch[i] = 0
                self._set_sumo_phase(tl_id, 1)
    
    def _compute_reward(self, intersection_id: int, action: int) -> float:
        """
        Three-component reward adapted from original working system.
        
        Component 1 — Queue penalty:
            Penalizes total PCU waiting after serving. Always negative.
        Component 2 — Imbalance penalty:
            Penalizes uneven NS/EW PCU queues after serving. Always negative.
        Component 3 — Switch reward:
            +3.0 if switched AND imbalance_before >= threshold (good switch)
            -2.0 if switched AND imbalance_before <  threshold (bad switch)
             0.0 if kept phase (neutral)
        
        Uses PCU-weighted queues throughout for Indian mixed traffic.
        """
        cfg = REWARD_CONFIG
        
        # After-serving state (result of agent decision)
        tl_id = self.tl_ids[intersection_id]
        ns_after, ew_after, _, _, _, _ = self._get_intersection_queues(tl_id)
        total_after = ns_after + ew_after
        imbalance_after = abs(ns_after - ew_after)
        
        # Before-serving state (what agent saw when deciding)
        before = self._before_serving_pcu[intersection_id]
        imbalance_before = abs(before["NS"] - before["EW"])
        
        # Detect if phase actually changed (action may be blocked by min_green)
        phase_before = self._before_phases[intersection_id]
        phase_after = self.current_phases[intersection_id]
        actually_switched = (phase_before != phase_after)
        
        norm = cfg["reward_queue_norm"]
        
        # Component 1: queue penalty
        queue_penalty = cfg["reward_queue_weight"] * (total_after / norm)
        
        # Component 2: imbalance penalty
        imbalance_penalty = cfg["reward_imbalance_weight"] * (imbalance_after / norm)
        
        # Component 3: switch reward (only if actually switched)
        if actually_switched:
            if imbalance_before >= cfg["reward_imbalance_threshold"]:
                switch_reward = cfg["reward_good_switch"]   # +3.0
            else:
                switch_reward = cfg["reward_bad_switch"]    # -2.0
        else:
            switch_reward = 0.0
        
        return float(queue_penalty + imbalance_penalty + switch_reward)
    
    def _calculate_rewards(self, actions: List[int]) -> List[float]:
        """Calculate rewards for all intersections."""
        return [self._compute_reward(i, actions[i]) for i in range(self.n_agents)]
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state."""
        total_pcu = 0.0
        total_vehicles = 0
        
        # Aggregate vehicle class counts across all intersections
        vehicle_class_counts = {
            "two_wheeler": 0,
            "auto_rickshaw": 0,
            "car": 0,
            "pedestrian_group": 0
        }
        
        for tl_id in self.tl_ids:
            ns_pcu, ew_pcu, ns_count, ew_count, ns_class_counts, ew_class_counts = self._get_intersection_queues(tl_id)
            total_pcu += ns_pcu + ew_pcu
            total_vehicles += ns_count + ew_count
            
            # Aggregate class counts from both NS and EW directions
            for vtype in vehicle_class_counts.keys():
                vehicle_class_counts[vtype] += ns_class_counts.get(vtype, 0) + ew_class_counts.get(vtype, 0)
        
        # Calculate average travel time
        avg_travel_time = 0.0
        if len(self.vehicle_travel_times) > 0:
            avg_travel_time = sum(self.vehicle_travel_times.values()) / len(self.vehicle_travel_times)
        
        # Calculate average queue (raw vehicle count)
        avg_queue_raw = total_vehicles / self.n_intersections if self.n_intersections > 0 else 0.0
        
        return {
            "step": self.current_step,
            "total_pcu": total_pcu,
            "total_vehicles": total_vehicles,
            "avg_pcu_per_intersection": total_pcu / self.n_intersections,
            "avg_queue_pcu": total_pcu / self.n_intersections,  # Average PCU per intersection
            "avg_queue_raw": avg_queue_raw,  # Average raw vehicle count per intersection
            "avg_queue": avg_queue_raw,  # For compatibility with train.py
            "throughput": len(self.arrived_vehicles),  # Total vehicles that completed their trip
            "avg_travel_time": avg_travel_time,  # Average travel time in seconds
            "scenario": self.scenario,
            "vehicle_class_counts": vehicle_class_counts,
            "turning_counts": self.turning_counts.copy(),
        }
    
    def _track_vehicle_metrics(self) -> None:
        """Track vehicle departure and arrival for throughput and travel time metrics."""
        try:
            # Track newly departed vehicles
            current_departed = set(self.traci_conn.simulation.getDepartedIDList())
            for veh_id in current_departed:
                if veh_id not in self.departed_vehicles:
                    self.departed_vehicles.add(veh_id)
                    self.vehicle_travel_times[veh_id] = self.current_step
            
            # Track newly arrived vehicles and calculate travel time
            current_arrived = set(self.traci_conn.simulation.getArrivedIDList())
            for veh_id in current_arrived:
                if veh_id not in self.arrived_vehicles:
                    self.arrived_vehicles.add(veh_id)
                    if veh_id in self.vehicle_travel_times:
                        departure_time = self.vehicle_travel_times[veh_id]
                        travel_time = self.current_step - departure_time
                        self.vehicle_travel_times[veh_id] = travel_time
        except Exception:
            # If TraCI calls fail, skip metrics tracking for this step
            pass
    
    def close(self):
        """Close SUMO connection."""
        if self.traci_conn is not None:
            try:
                self.traci_conn.close()
            except:
                pass
            self.traci_conn = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
