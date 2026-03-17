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


class FixedTimeController:
    """
    Fixed-time signal controller.
    Runs identical timing plan regardless of traffic state.
    Represents the most common signal control deployed
    at Indian urban intersections today.

    Cycle: NS_GREEN (30s) → ALL_RED (2s) → EW_GREEN (30s) → ALL_RED (2s)
    Total cycle length: 64 steps
    """

    def __init__(self, n_agents: int = 9,
                 ns_green_duration: int = 30,
                 ew_green_duration: int = 30,
                 clearance_duration: int = 2):
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
            after exactly clearance_steps=2 timesteps

    Uses observation indices directly matching env_sumo.py:
        obs[i][2] = NS PCU queue
        obs[i][3] = EW PCU queue
        obs[i][4] = current phase (0=NS, 1=clearance, 2=EW)
        obs[i][5] = steps_since_switch

    Constraints matching RL agent exactly:
        min_green_steps = 5  (from SUMO_CONFIG)
        clearance_steps = 2  (from SUMO_CONFIG)
        pressure_threshold = 3.0
    """

    def __init__(self,
                 n_agents: int = 9,
                 min_green_steps: int = 5,
                 pressure_threshold: float = 3.0):
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
                 saturation_flow: float = 1800.0,
                 lost_time_per_phase: float = 3.0):
        self.n_agents = n_agents
        self.saturation_flow = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        self.ns_green = 30
        self.ew_green = 30
        self.clearance = 2
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
