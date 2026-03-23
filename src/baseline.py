"""
Rule-based baseline controllers for traffic signal control.

Three baselines for paper Table IV comparison:
1. Fixed-Time:   Traditional fixed timing plan (most common in Indian cities)
2. Webster:      Optimal cycle length using Webster (1958) formula
3. MaxPressure:  Adaptive pressure-based controller (Varaiya 2013)

All baselines are pure rule-based — no learning, no training.
"""

from __future__ import annotations
from typing import List

try:
    from .config import BASELINE_CONFIG, SUMO_CONFIG
except ImportError:
    from config import BASELINE_CONFIG, SUMO_CONFIG


class FixedTimeController:
    """
    Fixed-time signal controller.
    Runs identical timing plan regardless of traffic state.
    Represents the most common signal control deployed
    at Indian urban intersections today.

    Cycle: NS_GREEN → ALL_RED → EW_GREEN → ALL_RED
    Durations sourced from BASELINE_CONFIG and SUMO_CONFIG.
    """

    def __init__(self, n_agents: int = 9,
                 ns_green_duration: int = BASELINE_CONFIG["fixed_time_cycle"],
                 ew_green_duration: int = BASELINE_CONFIG["fixed_time_cycle"],
                 clearance_duration: int = SUMO_CONFIG["clearance_steps"]):
        self.n_agents = n_agents
        self.ns_green_duration = ns_green_duration
        self.ew_green_duration = ew_green_duration
        self.clearance_duration = clearance_duration
        self.cycle_length = (ns_green_duration + clearance_duration +
                             ew_green_duration + clearance_duration)
        self.step_count = 0

    def act(self, obs) -> List[int]:  # noqa: ARG002
        """Returns keep (0) or switch (1) for each agent at fixed intervals."""
        position_in_cycle = self.step_count % self.cycle_length
        self.step_count += 1

        ns_end = self.ns_green_duration
        ew_end = self.ns_green_duration + self.clearance_duration + self.ew_green_duration

        if position_in_cycle in (ns_end, ew_end):
            return [1] * self.n_agents
        return [0] * self.n_agents

    def reset(self):
        """Reset step counter for new episode."""
        self.step_count = 0


class MaxPressureController:
    """
    MaxPressure rule-based adaptive controller (Varaiya 2013).

    Decision rule:
        If currently serving NS (phase 0):
            switch if (EW_PCU - NS_PCU) > threshold
        If currently serving EW (phase 2):
            switch if (NS_PCU - EW_PCU) > threshold
        If in clearance (phase 1):
            keep (action=0) — environment auto-transitions
            after exactly clearance_steps timesteps

    Uses raw PCU from env.get_raw_queue_pcu() — not observation indices —
    to avoid any normalization mismatch. Threshold from BASELINE_CONFIG.

    Constraints matching RL agent exactly:
        min_green_steps = SUMO_CONFIG["min_green_steps"]
        pressure_threshold = BASELINE_CONFIG["max_pressure_threshold"]
    """

    def __init__(self,
                 n_agents: int = 9,
                 min_green_steps: int = SUMO_CONFIG["min_green_steps"],
                 pressure_threshold: float = BASELINE_CONFIG["max_pressure_threshold"]):
        self.n_agents           = n_agents
        self.min_green          = min_green_steps
        self.pressure_threshold = pressure_threshold

    def act(self, obs) -> List[int]:
        """
        Returns action per agent: 0=keep, 1=switch.
        Never returns action=2 (force clearance) —
        MaxPressure only initiates natural phase transitions.
        """
        actions = []

        for i in range(self.n_agents):
            ns_pcu = float(obs[i][2])
            ew_pcu = float(obs[i][3])
            phase  = int(obs[i][4])
            steps  = int(obs[i][5])

            # Phase 1: clearance — always keep
            # Environment auto-transitions after clearance_steps=2
            if phase == 1:
                actions.append(0)
                continue

            # Minimum green constraint — cannot switch yet
            if steps < self.min_green:
                actions.append(0)
                continue

            # Switch if opposing direction has higher PCU load by more than threshold
            if phase == 0:   # currently NS green
                should_switch = (ew_pcu - ns_pcu) > self.pressure_threshold
            else:            # currently EW green (phase 2)
                should_switch = (ns_pcu - ew_pcu) > self.pressure_threshold

            actions.append(1 if should_switch else 0)

        return actions

    def reset(self):
        """Stateless — nothing to reset."""
        pass


class WebsterController:
    """
    Webster (1958) optimal cycle length and green split controller.

    Computes optimal timing from arrival rates at episode start.
    Still fixed timing but mathematically optimised for the
    observed directional flow asymmetry.

    Webster formula:
        C* = (1.5L + 5) / (1 - Y)
        where L = total lost time per cycle
              Y = sum of critical flow ratios per phase

    Green split proportional to critical flow ratio per phase.
    """

    def __init__(self, n_agents: int = 9,
                 saturation_flow: float = BASELINE_CONFIG["webster_saturation_flow"],
                 lost_time_per_phase: float = BASELINE_CONFIG["webster_lost_time"]):
        self.n_agents = n_agents
        self.saturation_flow = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        self.clearance = SUMO_CONFIG["clearance_steps"]
        self.ns_green = BASELINE_CONFIG["fixed_time_cycle"]
        self.ew_green = BASELINE_CONFIG["fixed_time_cycle"]
        self.step_count = 0
        self.cycle_length = self._compute_cycle_length()

    def compute_timing(self, ns_flow: float, ew_flow: float):
        """Compute optimal cycle length and green split from flow rates (vph)."""
        y_ns = ns_flow / self.saturation_flow
        y_ew = ew_flow / self.saturation_flow
        y_total = y_ns + y_ew
        lost_time = 2 * self.lost_time_per_phase

        if y_total >= 1.0:
            c_star = 120
        else:
            c_star = max(30, min(120, (1.5 * lost_time + 5) / (1 - y_total)))

        g_total = c_star - lost_time

        if y_total > 0:
            self.ns_green = max(10, round(g_total * y_ns / y_total))
            self.ew_green = max(10, round(g_total * y_ew / y_total))
        else:
            self.ns_green = round(g_total / 2)
            self.ew_green = round(g_total / 2)

        # If min-green clamping inflated the sum beyond g_total, scale back
        green_sum = self.ns_green + self.ew_green
        if 0 < g_total < green_sum:
            scale = g_total / green_sum
            self.ns_green = max(10, round(self.ns_green * scale))
            self.ew_green = max(10, round(self.ew_green * scale))

        self.cycle_length = self._compute_cycle_length()

    def _compute_cycle_length(self) -> int:
        return self.ns_green + self.clearance + self.ew_green + self.clearance

    def act(self, obs) -> List[int]:  # noqa: ARG002
        """Switch at Webster-computed intervals regardless of obs."""
        position_in_cycle = self.step_count % self.cycle_length
        self.step_count += 1

        ns_end = self.ns_green
        ew_end = self.ns_green + self.clearance + self.ew_green

        if position_in_cycle in (ns_end, ew_end):
            return [1] * self.n_agents
        return [0] * self.n_agents

    def reset(self):
        """Reset step counter for new episode."""
        self.step_count = 0
