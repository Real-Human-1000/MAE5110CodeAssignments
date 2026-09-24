"""Assignment 2, continuous time: ankle controller, region of attraction, walking.

The walker is the inverted pendulum of ``models/inverted_pendulum_walker.py``.
It walks passively (zero ankle torque) until its state enters the region of
attraction (RoA) of the feedback-linearizing ankle controller; from then on the
controller balances it upright.

Every simulation primitive accepts a batch of states shaped (2, N). The
single-trajectory simulator (``walk``) and the batched Poincare-map engine
(``simulate_to_section``) therefore share one implementation of the event
logic: ``step_walkers``.
"""

from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from functools import cached_property
from typing import NamedTuple

import numpy as np

from integrators import rk4
from models import inverted_pendulum_walker as model

ALPHA_MIN, ALPHA_MAX = np.pi / 8, np.pi / 7  # landing half-angle bounds (rad)
TORQUE_LIMITS_MGL = (-0.10, 0.05)  # ankle-torque bounds as fractions of m*g*l
BALANCE_TOLERANCE = 1e-3  # norm of [theta, velocity] that counts as standing
EVENT_TIME_TOLERANCE = 1e-12  # maximum crossing-time bracket width (s)
MAX_BISECTION_STEPS = 64  # safety cap for event localization


class Outcome(IntEnum):
    """How a simulated segment ended. Integer-valued so arrays can store it."""

    PENDING = 0  # still moving (used inside batched simulations)
    CAPTURED = 1  # entered the RoA of the ankle controller
    RETURNED = 2  # crossed the section theta = 0 with positive velocity
    STABILIZED = 3  # balanced to standstill
    FELL = -1  # reached the ground or diverged
    TIMEOUT = -2  # ran out of simulated time
    NO_POLICY = -3  # the footstep lookup table had no successful action


class EventKind(StrEnum):
    DECISION = "decision"  # landing angle chosen at the section
    SECTION = "section"  # forward crossing of theta = 0
    IMPACT = "impact"  # footstrike (state recorded after the reset)
    CAPTURE = "capture"  # entered the RoA


@dataclass(frozen=True)
class WalkerEvent:
    time: float
    kind: EventKind
    state: np.ndarray
    alpha: float = np.nan


@dataclass
class WalkerTrajectory:
    """Sampled walker motion.

    Each impact is stored as two samples at the same time, before and after
    the reset, so the jump stays visible and ``phase_segments`` can split it.
    ``alphas`` is the landing half-angle in effect (NaN while balancing) and
    ``torques`` is the ankle torque applied at each sample.
    """

    times: np.ndarray  # (T,)
    states: np.ndarray  # (2, T)
    torques: np.ndarray  # (T,)
    alphas: np.ndarray  # (T,)
    events: list[WalkerEvent] = field(default_factory=list)
    footstrikes: int = 0
    outcome: Outcome = Outcome.TIMEOUT

    @classmethod
    def at_rest(cls, state, outcome=Outcome.RETURNED):
        """A single-sample trajectory, used as the start of a stitched rollout."""
        state = np.asarray(state, dtype=float)
        return cls(
            np.zeros(1),
            state[:, None],
            np.zeros(1),
            np.full(1, np.nan),
            outcome=outcome,
        )

    @property
    def final_state(self):
        return self.states[:, -1].copy()

    def event_times(self, kind):
        return np.array([event.time for event in self.events if event.kind == kind])

    def section_velocities(self):
        return np.array(
            [event.state[1] for event in self.events if event.kind == EventKind.SECTION]
        )

    def extended(self, segment):
        """Append a segment that starts at this trajectory's final state.

        The segment's first sample replaces the shared endpoint, so the
        controls recorded there are the ones that act from that instant on.
        """
        offset = self.times[-1]
        return WalkerTrajectory(
            times=np.concatenate([self.times[:-1], offset + segment.times]),
            states=np.concatenate([self.states[:, :-1], segment.states], axis=1),
            torques=np.concatenate([self.torques[:-1], segment.torques]),
            alphas=np.concatenate([self.alphas[:-1], segment.alphas]),
            events=self.events
            + [replace(e, time=offset + e.time) for e in segment.events],
            footstrikes=self.footstrikes + segment.footstrikes,
            outcome=segment.outcome,
        )

    def impact_sample_indices(self):
        """Index of each post-impact sample (the pre-impact sample is just before it)."""
        return np.flatnonzero(np.diff(self.times) == 0) + 1

    def save(self, path):
        np.savez_compressed(
            path,
            times=self.times,
            states=self.states,
            torques=self.torques,
            alphas=self.alphas,
            footstrikes=self.footstrikes,
            outcome=self.outcome.name,
            event_times=[e.time for e in self.events],
            event_kinds=[str(e.kind) for e in self.events],
            event_states=np.array([e.state for e in self.events]).reshape(-1, 2),
            event_alphas=[e.alpha for e in self.events],
        )


class _TrajectoryBuilder:
    """Accumulates samples and events for one WalkerTrajectory."""

    def __init__(self):
        self.times, self.states, self.torques, self.alphas = [], [], [], []
        self.events = []
        self.footstrikes = 0

    @property
    def time(self):
        return self.times[-1]

    def add_sample(self, time, state, *, torque=0.0, alpha=np.nan):
        self.times.append(float(time))
        self.states.append(np.array(state, dtype=float))
        self.torques.append(float(torque))
        self.alphas.append(float(alpha))

    def add_event(self, kind):
        """Log an event at the latest sample; impacts also count a footstrike."""
        self.events.append(WalkerEvent(self.time, kind, self.states[-1].copy()))
        if kind == EventKind.IMPACT:
            self.footstrikes += 1

    def build(self, outcome):
        return WalkerTrajectory(
            np.array(self.times),
            np.array(self.states).T,
            np.array(self.torques),
            np.array(self.alphas),
            self.events,
            self.footstrikes,
            outcome,
        )


# --- Ankle controller -------------------------------------------------------


def torque_limits(params):
    """Return the (lower, upper) ankle-torque bounds in N m."""
    mgl = params["mass"] * params["gravity"] * params["length"]
    return TORQUE_LIMITS_MGL[0] * mgl, TORQUE_LIMITS_MGL[1] * mgl


def compute_ankle_torque(state, params):
    """Feedback-linearizing ankle torque, saturated to the assignment bounds.

    Cancel gravity's (g/l) sin(theta), then impose critically damped linear
    dynamics theta'' = -kp theta - kd theta' with kp = g/l. Works on (2,) or
    (2, N) states.
    """
    theta, velocity = state[0], state[1]
    g, length, mass = params["gravity"], params["length"], params["mass"]
    kp = g / length
    kd = 2 * np.sqrt(kp)
    gravity_compensation = -(g / length) * np.sin(theta)
    torque = mass * length**2 * (gravity_compensation - kp * theta - kd * velocity)
    return np.clip(torque, *torque_limits(params))


def balancing_dynamics(t, state, params):
    controlled_params = {**params, "ankle_torque": compute_ankle_torque(state, params)}
    return model.dynamics(t, state, controlled_params)


def has_fallen(states, params):
    """True where the state is non-finite or the pendulum has reached the ground."""
    states = np.asarray(states, dtype=float)
    reached_ground = np.abs(states[0] - params["incline"]) >= np.pi / 2
    return ~np.isfinite(states).all(axis=0) | reached_ground


def is_balanced(states, tolerance=BALANCE_TOLERANCE):
    """True where the walker stands still upright."""
    return np.linalg.norm(states, axis=0) < tolerance


def max_section_velocity(params):
    """Angular velocity at Froude number 2, sqrt(2 g / l): the velocity-grid bound."""
    return np.sqrt(2 * params["gravity"] / params["length"])


@dataclass
class BalancingRun:
    times: np.ndarray  # (T,) sample times of ``history``, or just the final time
    states: np.ndarray  # (2, N) final states
    alive: np.ndarray  # (N,) never fell
    history: np.ndarray | None = None  # (T, 2, N) when recorded

    @property
    def converged(self):
        return self.alive & is_balanced(self.states)


def integrate_balancing(states, params, *, dt, duration, record=False):
    """Integrate the closed-loop balancing dynamics for (2, N) states.

    Columns stop (freeze) once they fall; only surviving columns are
    integrated. The history is kept only when ``record`` is True.
    """
    states = np.array(states, dtype=float)
    n_steps = int(np.ceil(duration / dt))
    dt = duration / n_steps
    alive = ~has_fallen(states, params)
    history = [states.copy()] if record else None
    for step in range(n_steps):
        indices = np.flatnonzero(alive)
        if not indices.size:
            break
        states[:, indices] = rk4.step(
            balancing_dynamics, step * dt, states[:, indices], dt, params
        )
        alive[indices] = ~has_fallen(states[:, indices], params)
        if record:
            history.append(states.copy())
    if record:
        return BalancingRun(
            dt * np.arange(len(history)), states, alive, np.array(history)
        )
    return BalancingRun(np.array([duration]), states, alive)


# --- Region of attraction ----------------------------------------------------


def _cell_index(grid, values):
    """Index of the grid cell [grid[i], grid[i+1]] that contains each value."""
    return np.clip(np.searchsorted(grid, values, side="right") - 1, 0, len(grid) - 2)


@dataclass
class RegionOfAttraction:
    """Grid estimate of the ankle controller's RoA.

    ``converged`` marks grid points whose closed-loop trajectory reached
    standstill. A state counts as inside only if all four corners of its cell
    converged, which makes the estimate an inner approximation.
    """

    angles: np.ndarray
    velocities: np.ndarray
    converged: np.ndarray  # (len(angles), len(velocities))

    @cached_property
    def accepted_cells(self):
        """(n-1, m-1) mask of cells whose four corners converged."""
        c = self.converged
        return c[:-1, :-1] & c[1:, :-1] & c[:-1, 1:] & c[1:, 1:]

    def contains(self, state):
        """True where a (2,) or (2, N) state lies in an accepted cell."""
        theta, velocity = np.asarray(state, dtype=float)
        inside = (
            (self.angles[0] <= theta)
            & (theta <= self.angles[-1])
            & (self.velocities[0] <= velocity)
            & (velocity <= self.velocities[-1])
        )
        i = _cell_index(self.angles, theta)
        j = _cell_index(self.velocities, velocity)
        return inside & self.accepted_cells[i, j]


def estimate_roa(params, *, grid_size=161, dt=0.01, duration=12.0):
    """Grid search over every state between the fall bounds.

    Angles span the ground-to-ground range incline +/- pi/2 and velocities
    span +/- sqrt(2 g / l), the Froude-2 speed that bounds the walking grid.
    """
    gamma = params["incline"]
    angles = np.linspace(gamma - np.pi / 2, gamma + np.pi / 2, grid_size)
    velocity_bound = max_section_velocity(params)
    velocities = np.linspace(-velocity_bound, velocity_bound, grid_size)
    theta, velocity = np.meshgrid(angles, velocities, indexing="ij")
    run = integrate_balancing(
        np.array([theta.ravel(), velocity.ravel()]), params, dt=dt, duration=duration
    )
    return RegionOfAttraction(angles, velocities, run.converged.reshape(theta.shape))


class RoAValidation(NamedTuple):
    tested: int
    failed: int


def validate_roa(roa, params, *, dt=0.005, duration=15.0):
    """Re-simulate five interior points of every accepted cell at a finer dt."""
    i, j = np.nonzero(roa.accepted_cells)
    angle_steps, velocity_steps = np.diff(roa.angles), np.diff(roa.velocities)
    fractions = ((0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75))
    starts = np.concatenate(
        [
            [
                roa.angles[i] + f * angle_steps[i],
                roa.velocities[j] + h * velocity_steps[j],
            ]
            for f, h in fractions
        ],
        axis=1,
    )
    converged = integrate_balancing(starts, params, dt=dt, duration=duration).converged
    return RoAValidation(tested=int(converged.size), failed=int((~converged).sum()))


def roa_event_guard(state, roa):
    """True where the state is inside the RoA; always False without an RoA."""
    if roa is None:
        return np.zeros(np.shape(state)[1:], dtype=bool)
    return roa.contains(state)


# --- Walking: guards, event localization, and impacts -------------------------


def poincare_guard(previous_state, next_state):
    """True where the flow crosses the section theta = 0 with positive velocity."""
    return (previous_state[0] < 0) & (next_state[0] >= 0) & (next_state[1] > 0)


def validate_landing_angles(alphas, params):
    """Require reset angle < 0 < touchdown angle, i.e. |incline| < alpha < pi/2.

    Then every step crosses the section exactly once, and within one
    integration step the section is always reached before touchdown.
    """
    alphas = np.asarray(alphas, dtype=float)
    if not np.all(
        np.isfinite(alphas) & (abs(params["incline"]) < alphas) & (alphas < np.pi / 2)
    ):
        raise ValueError("Landing angles must satisfy |incline| < alpha < pi/2.")


def _passive_params(params, alphas):
    return {**params, "ankle_torque": 0.0, "angle_of_attack": alphas}


def locate_angle_crossing(states, max_dt, angles, params):
    """Bisect for the time at which each column's theta reaches its guard angle.

    Assumes theta increases monotonically during the step, which holds for
    both guards: the walker only crosses them moving forward. Returns
    (elapsed, crossing_states), with theta snapped onto the guard angle so the
    same guard cannot trigger again on the next step.
    """
    low = np.zeros_like(max_dt, dtype=float)
    high = np.array(max_dt, dtype=float)
    for _ in range(MAX_BISECTION_STEPS):
        if np.all(high - low <= EVENT_TIME_TOLERANCE):
            break
        mid = (low + high) / 2
        before = rk4.step(model.dynamics, 0.0, states, mid, params)[0] < angles
        low = np.where(before, mid, low)
        high = np.where(before, high, mid)
    crossing = rk4.step(model.dynamics, 0.0, states, high, params)
    crossing[0] = angles
    return high, crossing


def advance_to_first_event(states, alphas, max_dt, params):
    """Advance passive (2, N) walkers by up to max_dt, stopping at their first guard.

    Returns (states, elapsed, section, touchdown). Touchdown states are
    pre-impact. The section (theta = 0) lies before touchdown
    (theta = incline + alpha > 0), so a step that crosses both stops at the
    section and touchdown is found on a later step.
    """
    walking = _passive_params(params, alphas)
    elapsed = np.broadcast_to(np.asarray(max_dt, dtype=float), alphas.shape).copy()
    trial = rk4.step(model.dynamics, 0.0, states, elapsed, walking)
    section = poincare_guard(states, trial)
    touchdown = model.event_guard(states, trial, walking) & ~section
    crossed = section | touchdown
    if np.any(crossed):
        guard_angles = np.where(section, 0.0, params["incline"] + alphas)
        elapsed[crossed], trial[:, crossed] = locate_angle_crossing(
            states[:, crossed], elapsed[crossed], guard_angles[crossed], walking
        )
    return trial, elapsed, section, touchdown


def apply_touchdown_impact(states, alphas, touchdown, params):
    """Return a copy of states with the impact reset applied where touchdown is True."""
    reset = states.copy()
    reset[:, touchdown] = model.event_dynamics(
        states[:, touchdown], {**params, "angle_of_attack": alphas[touchdown]}
    )
    return reset


@dataclass
class WalkingStep:
    elapsed: np.ndarray  # (N,) time advanced
    pre_impact: np.ndarray  # (2, N) state at the end of the step, before any reset
    states: np.ndarray  # (2, N) state after the reset
    section: np.ndarray  # (N,) returned to the section without being captured
    impact: np.ndarray  # (N,) footstrike during this step
    captured: np.ndarray  # (N,) inside the RoA at the end of this step


def step_walkers(states, alphas, max_dt, params, roa):
    """One passive integration step with event localization, impact, and capture.

    Capture is checked after the impact: once the swing foot has touched the
    ground, the footstrike has happened.
    """
    pre_impact, elapsed, section, touchdown = advance_to_first_event(
        states, alphas, max_dt, params
    )
    post_impact = apply_touchdown_impact(pre_impact, alphas, touchdown, params)
    captured = roa_event_guard(post_impact, roa)
    return WalkingStep(
        elapsed, pre_impact, post_impact, section & ~captured, touchdown, captured
    )


# --- Walking simulations -------------------------------------------------------


@dataclass
class SectionResult:
    next_velocity: np.ndarray  # (N,) section velocity on return, NaN otherwise
    outcome: np.ndarray  # (N,) Outcome codes
    footstrikes: np.ndarray  # (N,)


def simulate_to_section(states, alphas, params, roa, *, dt=1e-3, duration=10.0):
    """Batched passive walking until each walker is captured or returns to the section.

    Other endings are a fall or the time limit. Each column localizes its own
    events and stops independently. Starting states may be anywhere before
    the section (theta < 0) or on it.
    """
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(duration) and duration > 0):
        raise ValueError("dt and duration must be positive and finite.")
    states = np.array(states, dtype=float)
    if states.ndim != 2 or states.shape[0] != 2 or not np.isfinite(states).all():
        raise ValueError("states must be a finite (2, N) array.")
    n = states.shape[1]
    alphas = np.broadcast_to(np.asarray(alphas, dtype=float), (n,)).copy()
    validate_landing_angles(alphas, params)

    elapsed = np.zeros(n)
    footstrikes = np.zeros(n, dtype=int)
    next_velocity = np.full(n, np.nan)
    outcome = np.full(n, Outcome.PENDING, dtype=int)
    outcome[roa_event_guard(states, roa)] = Outcome.CAPTURED
    # A truncated (event) step does not use a full dt; allow a few extra iterations.
    for _ in range(int(np.ceil(duration / dt)) + 3):
        ids = np.flatnonzero(outcome == Outcome.PENDING)
        if not ids.size:
            break
        step = step_walkers(
            states[:, ids],
            alphas[ids],
            np.minimum(dt, duration - elapsed[ids]),
            params,
            roa,
        )
        states[:, ids] = step.states
        elapsed[ids] += step.elapsed
        footstrikes[ids] += step.impact
        outcome[ids[step.captured]] = Outcome.CAPTURED
        outcome[ids[step.section]] = Outcome.RETURNED
        next_velocity[ids[step.section]] = step.states[1, step.section]
        fell = has_fallen(step.states, params) & ~step.captured & ~step.section
        outcome[ids[fell]] = Outcome.FELL
        timed_out = (elapsed[ids] >= duration - 1e-12) & (
            outcome[ids] == Outcome.PENDING
        )
        outcome[ids[timed_out]] = Outcome.TIMEOUT
    outcome[outcome == Outcome.PENDING] = Outcome.TIMEOUT
    return SectionResult(next_velocity, outcome, footstrikes)


def _validated_state(state):
    state = np.array(state, dtype=float)
    if state.shape != (2,) or not np.isfinite(state).all():
        raise ValueError("A walker state must contain two finite values.")
    return state


def walk(
    initial_state, alpha, params, roa, *, dt=1e-3, duration=10.0, stop_at_section=False
):
    """Passive walking with a fixed landing angle, recorded sample by sample.

    Runs until capture, a fall, or the time limit. With ``stop_at_section``
    it also stops at the next section crossing.
    """
    state = _validated_state(initial_state)
    validate_landing_angles(alpha, params)
    builder = _TrajectoryBuilder()
    builder.add_sample(0.0, state, alpha=alpha)
    if roa_event_guard(state, roa):
        builder.add_event(EventKind.CAPTURE)
        return builder.build(Outcome.CAPTURED)

    alphas = np.array([alpha], dtype=float)
    time = 0.0
    while time < duration:
        step = step_walkers(
            state[:, None], alphas, min(dt, duration - time), params, roa
        )
        time += step.elapsed[0]
        state = step.states[:, 0]
        if step.impact[0]:
            builder.add_sample(time, step.pre_impact[:, 0], alpha=alpha)
        builder.add_sample(time, state, alpha=alpha)
        if step.impact[0]:
            builder.add_event(EventKind.IMPACT)
        if step.captured[0]:
            builder.add_event(EventKind.CAPTURE)
            return builder.build(Outcome.CAPTURED)
        if step.section[0]:
            builder.add_event(EventKind.SECTION)
            if stop_at_section:
                return builder.build(Outcome.RETURNED)
        if has_fallen(state, params):
            return builder.build(Outcome.FELL)
    return builder.build(Outcome.TIMEOUT)


def balance(
    initial_state, params, *, dt=1e-3, duration=10.0, tolerance=BALANCE_TOLERANCE
):
    """Run the ankle controller until the walker stands still, falls, or times out."""
    state = _validated_state(initial_state)
    builder = _TrajectoryBuilder()
    builder.add_sample(0.0, state, torque=compute_ankle_torque(state, params))
    time = 0.0
    while not is_balanced(state, tolerance):
        if time >= duration:
            return builder.build(Outcome.TIMEOUT)
        step_dt = min(dt, duration - time)
        state = rk4.step(balancing_dynamics, time, state, step_dt, params)
        time += step_dt
        builder.add_sample(time, state, torque=compute_ankle_torque(state, params))
        if has_fallen(state, params):
            return builder.build(Outcome.FELL)
    return builder.build(Outcome.STABILIZED)


def simulate_walker(
    initial_state,
    params,
    roa,
    *,
    dt=1e-3,
    duration=10.0,
    stop_at_section=False,
    balance_after_capture=True,
):
    """Walk with ``params["angle_of_attack"]``; once captured, balance to standstill."""
    walking = walk(
        initial_state,
        params["angle_of_attack"],
        params,
        roa,
        dt=dt,
        duration=duration,
        stop_at_section=stop_at_section,
    )
    if walking.outcome != Outcome.CAPTURED or not balance_after_capture:
        return walking
    return walking.extended(
        balance(walking.final_state, params, dt=dt, duration=duration)
    )
