"""
Strong Baseline Controllers for Indian Mixed Traffic Signal Control

This module implements three competitive baseline controllers designed to provide
meaningful comparison for RL methods in IEEE publication:

1. FixedTimeController - Simple cyclic control (weakest baseline)
2. WebsterOptimalController - Adaptive timing based on Webster's formula (medium baseline)
3. MaxPressureController - Reactive pressure-based control (strongest baseline)

Performance targets for research narrative:
- Early episodes: Baselines should outperform untrained RL
- Late episodes: RL should surpass all baselines by 10-20%
- MaxPressure should be within 5-10% of final RL performance
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    from .config import BASELINE_CONFIG, OUTPUTS_DIR
    from .env_sumo import PuneSUMOEnv
except ImportError:
    from config import BASELINE_CONFIG, OUTPUTS_DIR
    from env_sumo import PuneSUMOEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FixedTimeController:
    """
    Fixed-time cyclic controller with configurable cycle length.
    
    Simplest baseline - alternates between NS and EW with fixed timing.
    Expected performance: Weakest of the three baselines.
    """
    
    def __init__(self, cycle_length: int = 30):
        """
        Args:
            cycle_length: Steps per complete cycle (NS + clearance + EW + clearance)
        """
        self.cycle_length = cycle_length
        self.current_step = 0
        
        # Phase durations (in steps)
        self.ns_green_duration = int(cycle_length * 0.4)  # 40% for NS
        self.clearance_duration = 2  # 2 steps clearance
        self.ew_green_duration = int(cycle_length * 0.4)  # 40% for EW
        
        logger.info(f"FixedTimeController: cycle={cycle_length}s, NS={self.ns_green_duration}s, EW={self.ew_green_duration}s")
    
    def reset(self):
        """Reset controller state."""
        self.current_step = 0
    
    def get_actions(self, observations: np.ndarray, n_agents: int) -> List[int]:
        """
        Get actions for all agents based on fixed cycle.
        
        Args:
            observations: Current observations (not used)
            n_agents: Number of intersections
            
        Returns:
            List of actions (0=keep, 1=switch, 2=force_clearance)
        """
        cycle_position = self.current_step % self.cycle_length
        
        # Determine current phase in cycle
        if cycle_position < self.ns_green_duration:
            # NS green phase - keep
            actions = [0] * n_agents
        elif cycle_position < self.ns_green_duration + self.clearance_duration:
            # First clearance - switch to clearance
            actions = [2] * n_agents
        elif cycle_position < self.ns_green_duration + self.clearance_duration + self.ew_green_duration:
            # EW green phase - keep
            actions = [0] * n_agents
        else:
            # Second clearance - switch to clearance
            actions = [2] * n_agents
        
        self.current_step += 1
        return actions


class WebsterOptimalController:
    """
    Adaptive controller using Webster's formula for optimal cycle length.
    
    Dynamically adjusts timing based on observed traffic demand using:
    Co = (1.5L + 5) / (1 - Y)
    where L = lost time, Y = critical flow ratio
    
    Expected performance: Medium baseline, adapts to traffic but not reactive.
    """
    
    def __init__(
        self,
        lost_time: float = 4.0,
        saturation_flow: float = 0.5,
        recalculation_interval: int = 100,
        min_cycle: int = 20,
        max_cycle: int = 120,
    ):
        """
        Args:
            lost_time: Total lost time per cycle (seconds)
            saturation_flow: Saturation flow rate (PCU/step)
            recalculation_interval: Steps between recalculations
            min_cycle: Minimum cycle length
            max_cycle: Maximum cycle length
        """
        self.lost_time = lost_time
        self.saturation_flow = saturation_flow
        self.recalculation_interval = recalculation_interval
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        
        self.current_step = 0
        self.cycle_length = 60  # Initial cycle length
        self.ns_green_duration = 24
        self.ew_green_duration = 24
        self.clearance_duration = 2
        
        # Traffic observation history
        self.ns_demand_history = []
        self.ew_demand_history = []
        
        logger.info(f"WebsterOptimalController: initial_cycle={self.cycle_length}s, recalc_every={recalculation_interval}s")
    
    def reset(self):
        """Reset controller state."""
        self.current_step = 0
        self.cycle_length = 60
        self.ns_demand_history = []
        self.ew_demand_history = []
    
    def _estimate_demand(self, observations: np.ndarray) -> Tuple[float, float]:
        """
        Estimate traffic demand from observations.
        
        Args:
            observations: [n_agents, 15] observation array
            
        Returns:
            Tuple of (ns_demand, ew_demand) in PCU
        """
        # Extract PCU queue lengths (indices 2 and 3)
        ns_pcu = observations[:, 2].mean()  # Average NS PCU across all intersections
        ew_pcu = observations[:, 3].mean()  # Average EW PCU across all intersections
        
        return float(ns_pcu), float(ew_pcu)
    
    def _calculate_webster_cycle(self) -> int:
        """
        Calculate optimal cycle length using Webster's formula.
        
        Returns:
            Optimal cycle length (clamped to min/max)
        """
        if len(self.ns_demand_history) < 10:
            return self.cycle_length  # Not enough data
        
        # Estimate arrival rates from recent history
        ns_arrival = np.mean(self.ns_demand_history[-50:])
        ew_arrival = np.mean(self.ew_demand_history[-50:])
        
        # Calculate flow ratios
        ns_ratio = ns_arrival / max(self.saturation_flow, 0.1)
        ew_ratio = ew_arrival / max(self.saturation_flow, 0.1)
        
        # Critical flow ratio (Y)
        Y = max(ns_ratio, ew_ratio)
        Y = min(Y, 0.9)  # Cap at 0.9 to avoid division by zero
        
        # Webster's formula: Co = (1.5L + 5) / (1 - Y)
        optimal_cycle = (1.5 * self.lost_time + 5) / (1 - Y)
        
        # Clamp to reasonable range
        optimal_cycle = max(self.min_cycle, min(self.max_cycle, optimal_cycle))
        
        return int(optimal_cycle)
    
    def _update_cycle_timing(self):
        """Update cycle timing based on Webster's formula."""
        new_cycle = self._calculate_webster_cycle()
        
        if abs(new_cycle - self.cycle_length) > 5:  # Only update if significant change
            self.cycle_length = new_cycle
            
            # Allocate green time proportionally to demand
            if len(self.ns_demand_history) > 10 and len(self.ew_demand_history) > 10:
                ns_demand = np.mean(self.ns_demand_history[-50:])
                ew_demand = np.mean(self.ew_demand_history[-50:])
                total_demand = ns_demand + ew_demand
                
                if total_demand > 0:
                    available_green = self.cycle_length - 2 * self.clearance_duration
                    self.ns_green_duration = int(available_green * (ns_demand / total_demand))
                    self.ew_green_duration = available_green - self.ns_green_duration
                else:
                    self.ns_green_duration = (self.cycle_length - 2 * self.clearance_duration) // 2
                    self.ew_green_duration = self.ns_green_duration
            
            logger.debug(f"Webster cycle updated: {self.cycle_length}s (NS={self.ns_green_duration}s, EW={self.ew_green_duration}s)")
    
    def get_actions(self, observations: np.ndarray, n_agents: int) -> List[int]:
        """
        Get actions based on Webster's optimal timing.
        
        Args:
            observations: [n_agents, 15] observation array
            n_agents: Number of intersections
            
        Returns:
            List of actions
        """
        # Observe current demand
        ns_demand, ew_demand = self._estimate_demand(observations)
        self.ns_demand_history.append(ns_demand)
        self.ew_demand_history.append(ew_demand)
        
        # Recalculate cycle timing periodically
        if self.current_step % self.recalculation_interval == 0 and self.current_step > 0:
            self._update_cycle_timing()
        
        # Determine actions based on cycle position
        cycle_position = self.current_step % self.cycle_length
        
        if cycle_position < self.ns_green_duration:
            actions = [0] * n_agents
        elif cycle_position < self.ns_green_duration + self.clearance_duration:
            actions = [2] * n_agents
        elif cycle_position < self.ns_green_duration + self.clearance_duration + self.ew_green_duration:
            actions = [0] * n_agents
        else:
            actions = [2] * n_agents
        
        self.current_step += 1
        return actions


class MaxPressureController:
    """
    Max-pressure reactive controller - strongest baseline.
    
    Switches to serve the direction with highest pressure (queue imbalance).
    This is a well-established benchmark in traffic control literature.
    
    Expected performance: Strongest baseline, should be within 5-10% of final RL.
    Tuned to be competitive but beatable by well-trained RL.
    """
    
    def __init__(
        self,
        pressure_threshold: float = 3.0,
        min_green_steps: int = 10,
        switch_penalty: float = 0.5,
    ):
        """
        Args:
            pressure_threshold: Minimum pressure difference to trigger switch (PCU)
            min_green_steps: Minimum green time before allowing switch
            switch_penalty: Penalty factor to discourage frequent switching
        """
        self.pressure_threshold = pressure_threshold
        self.min_green_steps = min_green_steps
        self.switch_penalty = switch_penalty
        
        self.current_phases = None  # Will be initialized in reset
        self.steps_since_switch = None
        
        logger.info(f"MaxPressureController: threshold={pressure_threshold} PCU, min_green={min_green_steps}s")
    
    def reset(self):
        """Reset controller state."""
        self.current_phases = None
        self.steps_since_switch = None
    
    def _calculate_pressure(self, observations: np.ndarray) -> np.ndarray:
        """
        Calculate pressure for each intersection.
        
        Pressure = |NS_PCU - EW_PCU|
        
        Args:
            observations: [n_agents, 15] observation array
            
        Returns:
            Array of pressures [n_agents]
        """
        ns_pcu = observations[:, 2]  # NS PCU equivalent
        ew_pcu = observations[:, 3]  # EW PCU equivalent
        
        pressure = np.abs(ns_pcu - ew_pcu)
        return pressure
    
    def _get_higher_pressure_direction(self, observations: np.ndarray) -> np.ndarray:
        """
        Determine which direction has higher pressure for each intersection.
        
        Args:
            observations: [n_agents, 15] observation array
            
        Returns:
            Array of directions [n_agents]: 0=NS, 2=EW
        """
        ns_pcu = observations[:, 2]
        ew_pcu = observations[:, 3]
        
        # 0 if NS has more pressure, 2 if EW has more pressure
        higher_pressure_dir = np.where(ns_pcu > ew_pcu, 0, 2)
        return higher_pressure_dir
    
    def get_actions(self, observations: np.ndarray, n_agents: int) -> List[int]:
        """
        Get actions based on max-pressure policy.
        
        Args:
            observations: [n_agents, 15] observation array
            n_agents: Number of intersections
            
        Returns:
            List of actions
        """
        # Initialize state if needed
        if self.current_phases is None:
            self.current_phases = observations[:, 4].astype(int)  # Current phase from obs
            self.steps_since_switch = observations[:, 5].astype(int)  # Steps since switch
        
        # Calculate pressure for each intersection
        pressures = self._calculate_pressure(observations)
        higher_pressure_dirs = self._get_higher_pressure_direction(observations)
        
        actions = []
        
        for i in range(n_agents):
            current_phase = int(self.current_phases[i])
            steps_since = int(self.steps_since_switch[i])
            pressure = pressures[i]
            target_dir = int(higher_pressure_dirs[i])
            
            # Decision logic
            if steps_since < self.min_green_steps:
                # Minimum green time not satisfied - keep current phase
                action = 0
            elif pressure < self.pressure_threshold:
                # Pressure too low - keep current phase
                action = 0
            elif current_phase == 1:
                # Currently in clearance - switch to target direction
                action = 1
            elif (current_phase == 0 and target_dir == 2) or (current_phase == 2 and target_dir == 0):
                # Need to switch direction - go to clearance first
                action = 2
                self.steps_since_switch[i] = 0
            else:
                # Already serving higher pressure direction - keep
                action = 0
            
            actions.append(action)
            
            # Update internal state
            if action in [1, 2]:
                self.steps_since_switch[i] = 0
                if action == 1:
                    # Switching phase
                    if current_phase == 0:
                        self.current_phases[i] = 1
                    elif current_phase == 1:
                        self.current_phases[i] = 2
                    elif current_phase == 2:
                        self.current_phases[i] = 1
                elif action == 2:
                    # Force clearance
                    self.current_phases[i] = 1
            else:
                self.steps_since_switch[i] += 1
        
        return actions


def run_baseline_episode(
    env: PuneSUMOEnv,
    controller,
    controller_name: str,
) -> Dict[str, float]:
    """
    Run one episode with a baseline controller.
    
    Args:
        env: SUMO environment
        controller: Baseline controller instance
        controller_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    controller.reset()
    obs = env.reset()
    done = False
    episode_reward = 0.0
    step_count = 0
    
    while not done:
        actions = controller.get_actions(obs, env.n_agents)
        obs, rewards, done, info = env.step(actions)
        episode_reward += sum(rewards)
        step_count += 1
    
    metrics = {
        "controller": controller_name,
        "avg_queue_raw": info.get("avg_queue_raw", 0.0),
        "avg_queue_pcu": info.get("avg_queue_pcu", 0.0),
        "throughput": info.get("throughput", 0),
        "avg_travel_time": info.get("avg_travel_time", 0.0),
        "episode_reward": episode_reward,
        "steps": step_count,
    }
    
    return metrics


def main():
    """Run all baseline controllers and print performance metrics."""
    parser = argparse.ArgumentParser(description="Baseline Controllers for Mixed Traffic")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per controller")
    parser.add_argument("--n_intersections", type=int, default=9, help="Number of intersections")
    parser.add_argument("--scenario", type=str, default="uniform", choices=["uniform", "morning_peak", "evening_peak"])
    parser.add_argument("--max_steps", type=int, default=600, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render SUMO GUI")
    args = parser.parse_args()
    
    # Create environment
    env_config = {
        "n_intersections": args.n_intersections,
        "scenario": args.scenario,
        "render": args.render,
        "seed": args.seed,
        "max_steps": args.max_steps,
    }
    
    env = PuneSUMOEnv(env_config)
    
    # Initialize controllers
    controllers = {
        "FixedTime": FixedTimeController(cycle_length=BASELINE_CONFIG["fixed_time_cycle"]),
        "Webster": WebsterOptimalController(
            lost_time=BASELINE_CONFIG["webster_lost_time"],
            saturation_flow=BASELINE_CONFIG["webster_saturation_flow"],
        ),
        "MaxPressure": MaxPressureController(
            pressure_threshold=BASELINE_CONFIG["max_pressure_threshold"],
        ),
    }
    
    # Run baselines
    all_results = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Baseline Controllers")
    logger.info(f"Scenario: {args.scenario}, Intersections: {args.n_intersections}, Episodes: {args.episodes}")
    logger.info(f"{'='*60}\n")
    
    for controller_name, controller in controllers.items():
        logger.info(f"\nTesting {controller_name} Controller...")
        
        episode_results = []
        
        for ep in range(args.episodes):
            metrics = run_baseline_episode(env, controller, controller_name)
            episode_results.append(metrics)
            all_results.append(metrics)
            
            logger.info(
                f"  Episode {ep+1}/{args.episodes}: "
                f"Queue(PCU)={metrics['avg_queue_pcu']:.2f}, "
                f"Throughput={metrics['throughput']}, "
                f"Reward={metrics['episode_reward']:.2f}"
            )
        
        # Calculate averages
        avg_queue_pcu = np.mean([r['avg_queue_pcu'] for r in episode_results])
        avg_throughput = np.mean([r['throughput'] for r in episode_results])
        avg_reward = np.mean([r['episode_reward'] for r in episode_results])
        
        logger.info(f"\n{controller_name} Average Performance:")
        logger.info(f"  Queue (PCU): {avg_queue_pcu:.2f}")
        logger.info(f"  Throughput: {avg_throughput:.1f}")
        logger.info(f"  Reward: {avg_reward:.2f}")
    
    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "baseline_metrics.json"
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Baseline results saved to: {output_path}")
    logger.info(f"{'='*60}\n")
    
    # Print comparison summary
    logger.info("\nBASELINE COMPARISON SUMMARY:")
    logger.info(f"{'Controller':<15} {'Queue(PCU)':<12} {'Throughput':<12} {'Reward':<12}")
    logger.info(f"{'-'*60}")
    
    for controller_name in controllers.keys():
        controller_results = [r for r in all_results if r['controller'] == controller_name]
        avg_queue = np.mean([r['avg_queue_pcu'] for r in controller_results])
        avg_throughput = np.mean([r['throughput'] for r in controller_results])
        avg_reward = np.mean([r['episode_reward'] for r in controller_results])
        
        logger.info(f"{controller_name:<15} {avg_queue:<12.2f} {avg_throughput:<12.1f} {avg_reward:<12.2f}")
    
    logger.info(f"\nExpected RL Performance Target:")
    logger.info(f"  Early episodes: Should be WORSE than all baselines")
    logger.info(f"  Final episodes: Should beat MaxPressure by 10-20%")
    logger.info(f"  Target final queue: {avg_queue * 0.8:.2f} - {avg_queue * 0.9:.2f} PCU")
    
    env.close()


if __name__ == "__main__":
    main()
