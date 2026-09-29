from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from mae5110.assignment_2_physics import (
    ALPHA_MAX,
    ALPHA_MIN,
    EventKind,
    Outcome,
    WalkerEvent,
    WalkerTrajectory,
    max_section_velocity,
    roa_event_guard,
    simulate_to_section,
    simulate_walker,
    validate_roa,
)

TABLE_DT = 1e-3  # integration step for candidate tables and rollouts (s)
REFERENCE_DT = 5e-4  # integration step for the fine reference table (s)
REFERENCE_GRID = (641, 81)  # (velocity points, angle points)
GRID_CANDIDATES = (
    (21, 6),
    (41, 11),
    (61, 16),
    (81, 21),
    (101, 26),
    (121, 31),
    (161, 41),
    (181, 46),
    (201, 51),
    (221, 56),
    (241, 61),
    (321, 81),
)
VALIDATION_STARTS = 120  # off-grid section velocities used to score each grid
MIN_AGREEMENT = 0.99  # required fraction of matching footstrike counts
CALIBRATION_OFFSET, HOLDOUT_OFFSET = (
    0.37,
    0.83,
)  # sub-cell offsets of the two sample sets


def section_states(velocities):
    """(2, N) states on the section theta = 0."""
    velocities = np.atleast_1d(np.asarray(velocities, dtype=float))
    return np.array([np.zeros_like(velocities), velocities])


def nearest_indices(grid, values):
    """Nearest grid index for each value (ties go low); callers check bounds."""
    upper = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    lower = upper - 1
    return np.where(values - grid[lower] <= grid[upper] - values, lower, upper)


# --- State-action table ------------------------------------------------------


@dataclass
class TransitionTable:
    velocities: np.ndarray  # (n_v,) section velocities
    alphas: np.ndarray  # (n_a,) landing half-angles
    next_velocity: np.ndarray  # (n_v, n_a) velocity at the next section, NaN otherwise
    outcome: np.ndarray  # (n_v, n_a) Outcome codes
    footstrikes: np.ndarray  # (n_v, n_a) footstrikes during the transition

    @property
    def shape(self):
        return self.outcome.shape

    def save(self, path):
        np.savez_compressed(path, **vars(self))


def build_transition_table(params, roa, *, n_velocity=161, n_alpha=41, dt=TABLE_DT):
    """Simulate one step from every (section velocity, landing angle) grid pair."""
    if n_velocity < 2 or n_alpha < 2:
        raise ValueError("Each grid needs at least two points.")
    velocities = np.linspace(0.0, max_section_velocity(params), n_velocity)
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, n_alpha)
    velocity, alpha = np.meshgrid(velocities, alphas, indexing="ij")
    result = simulate_to_section(
        section_states(velocity.ravel()), alpha.ravel(), params, roa, dt=dt
    )
    return TransitionTable(
        velocities,
        alphas,
        result.next_velocity.reshape(velocity.shape),
        result.outcome.reshape(velocity.shape),
        result.footstrikes.reshape(velocity.shape),
    )


# --- Policies ----------------------------------------------------------------


def successor_indices(table):
    """Nearest-grid successor of every returning (velocity, alpha) pair; -1 if none."""
    grid = table.velocities
    returning = (
        (table.outcome == Outcome.RETURNED)
        & (grid[0] <= table.next_velocity)
        & (table.next_velocity <= grid[-1])
    )
    successor = np.full(table.shape, -1, dtype=int)
    successor[returning] = nearest_indices(grid, table.next_velocity[returning])
    return successor


def bellman_backup(table, successor, values, *, unreachable):
    """Footstrikes to standstill for each (velocity, alpha), given successor values."""
    q = np.full(table.shape, unreachable, dtype=float)
    capture = table.outcome == Outcome.CAPTURED
    q[capture] = table.footstrikes[capture]
    returning = successor >= 0
    q[returning] = table.footstrikes[returning] + values[successor[returning]]
    return q


def solve_minimum_steps(table, successor):
    """Fewest footstrikes to standstill per grid velocity (inf: capture unreachable).

    Sweep k adds the states that can reach standstill in k steps, i.e. it
    backs the capture set out one step at a time.
    """
    values = np.full(len(table.velocities), np.inf)
    for _ in range(len(values) + 1):
        updated = bellman_backup(table, successor, values, unreachable=np.inf).min(
            axis=1
        )
        if np.array_equal(updated, values):
            break
        values = updated
    return values


def _propagate_to_predecessors(marked, successor):
    """Also mark every state that has some action leading to a marked state."""
    marked = marked.copy()
    returning = successor >= 0
    while True:
        leads_to_marked = np.zeros(successor.shape, dtype=bool)
        leads_to_marked[returning] = marked[successor[returning]]
        updated = marked | leads_to_marked.any(axis=1)
        if np.array_equal(updated, marked):
            return marked
        marked = updated


def solve_maximum_steps(table, successor):
    """Most footstrikes that still end in capture, per grid velocity.

    -inf means capture is unreachable. +inf means a reachable cycle in the
    grid graph can be repeated before capture; that is a property of the
    discretized graph, not proof of an endless physical walk. Without cycles
    the values settle within n sweeps, because a simple path has at most n
    edges. A state on a cycle of length L <= n grows at least once every L
    sweeps, so growth during sweeps n..2n flags exactly the cycle states.
    """
    n = len(table.velocities)
    values = np.full(n, -np.inf)
    growing = np.zeros(n, dtype=bool)
    for sweep in range(2 * n + 1):
        updated = bellman_backup(table, successor, values, unreachable=-np.inf).max(
            axis=1
        )
        if np.array_equal(updated, values):
            break
        if sweep >= n:
            growing |= updated > values
        values = updated
    values[_propagate_to_predecessors(growing, successor)] = np.inf
    return values


def choose_actions(q, table, *, minimize):
    """Pick the optimal alpha index per velocity; -1 where there is no finite optimum.

    Ties are broken away from outcome boundaries. Prefer a slower successor
    when minimizing and a faster one when maximizing. Among tied capture
    actions, take the middle of the widest contiguous run.
    """
    optimum = np.min(q, axis=1) if minimize else np.max(q, axis=1)
    tied = q == optimum[:, None]
    speed = np.where(np.isfinite(table.next_velocity), table.next_velocity, 0.0)
    score = np.where(tied, -speed if minimize else speed, -np.inf)
    best = np.argmax(score, axis=1)
    capture = table.outcome == Outcome.CAPTURED
    for i in np.flatnonzero(np.any(tied & capture, axis=1)):
        choices = np.flatnonzero(tied[i] & capture[i])
        runs = np.split(choices, np.flatnonzero(np.diff(choices) != 1) + 1)
        run = max(runs, key=len)
        best[i] = run[len(run) // 2]
    best[~np.isfinite(optimum)] = -1
    return best


@dataclass
class StepPolicy:
    table: TransitionTable
    minimum: np.ndarray  # (n_v,) fewest footstrikes to standstill
    min_action: np.ndarray  # (n_v,) alpha index achieving it, -1 if none
    maximum: np.ndarray  # (n_v,) most footstrikes before capture
    max_action: np.ndarray  # (n_v,) alpha index achieving it, -1 if none
    successor: np.ndarray  # (n_v, n_a) nearest-grid successor index, -1 if none

    def actions(self, velocities, *, longest=False):
        """Look up landing angles for continuous section velocities.

        Returns (alphas, usable); alphas is NaN where usable is False.
        """
        grid = self.table.velocities
        velocities = np.asarray(velocities, dtype=float)
        chosen = (self.max_action if longest else self.min_action)[
            nearest_indices(grid, velocities)
        ]
        usable = (grid[0] <= velocities) & (velocities <= grid[-1]) & (chosen >= 0)
        return np.where(usable, self.table.alphas[chosen], np.nan), usable

    def action(self, velocity, *, longest=False):
        """Landing angle for one section velocity; raises if the table has none."""
        alpha, usable = self.actions(velocity, longest=longest)
        if not usable:
            raise ValueError(
                f"No successful action for section velocity {velocity:.4g} rad/s."
            )
        return float(alpha)

    def action_values(self, *, longest=False):
        """(n_v, n_a) footstrikes to standstill after taking each alpha, then following the policy."""
        values, unreachable = (
            (self.maximum, -np.inf) if longest else (self.minimum, np.inf)
        )
        return bellman_backup(
            self.table, self.successor, values, unreachable=unreachable
        )


def solve_step_policy(table):
    """Solve the fewest-step and the most-step policies of a transition table."""
    successor = successor_indices(table)
    minimum = solve_minimum_steps(table, successor)
    maximum = solve_maximum_steps(table, successor)
    min_q = bellman_backup(table, successor, minimum, unreachable=np.inf)
    max_q = bellman_backup(table, successor, maximum, unreachable=-np.inf)
    return StepPolicy(
        table,
        minimum,
        choose_actions(min_q, table, minimize=True),
        maximum,
        choose_actions(max_q, table, minimize=False),
        successor,
    )


# --- Rollouts on the continuous dynamics -----------------------------------------


def rollout_policy(
    velocity, policy, params, roa, *, longest=False, dt=TABLE_DT, max_steps=100
):
    """Walk from [0, velocity], choosing alpha at each section from the lookup table.

    The policy acts on the actual continuous velocity; the state is never
    snapped to the grid. Ends balanced (STABILIZED) or with the failure reason.
    """
    trajectory = WalkerTrajectory.at_rest(section_states(velocity)[:, 0])
    for _ in range(max_steps):
        state = trajectory.final_state
        if roa_event_guard(state, roa):
            return trajectory.extended(simulate_walker(state, params, roa, dt=dt))
        try:
            alpha = policy.action(state[1], longest=longest)
        except ValueError:
            return replace(trajectory, outcome=Outcome.NO_POLICY)
        segment = simulate_walker(
            state,
            {**params, "angle_of_attack": alpha},
            roa,
            dt=dt,
            stop_at_section=True,
        )
        decision = WalkerEvent(0.0, EventKind.DECISION, state, alpha)
        trajectory = trajectory.extended(
            replace(segment, events=[decision, *segment.events])
        )
        if segment.outcome != Outcome.RETURNED:
            return trajectory
    return replace(trajectory, outcome=Outcome.TIMEOUT)


class PolicyEvaluation(NamedTuple):
    footstrikes: np.ndarray  # (N,) footstrikes until the walk ended
    outcome: np.ndarray  # (N,) CAPTURED on success, otherwise the failure reason


def evaluate_policy(
    velocities, policy, params, roa, *, longest=False, dt=TABLE_DT, max_steps=60
):
    """Batched continuous rollouts from section velocities, stopping at capture."""
    velocity = np.array(velocities, dtype=float).ravel()
    footstrikes = np.zeros(velocity.size, dtype=int)
    outcome = np.full(velocity.size, Outcome.PENDING, dtype=int)
    for _ in range(max_steps):
        ids = np.flatnonzero(outcome == Outcome.PENDING)
        if not ids.size:
            break
        captured = roa_event_guard(section_states(velocity[ids]), roa)
        alphas, usable = policy.actions(velocity[ids], longest=longest)
        outcome[ids[captured]] = Outcome.CAPTURED
        outcome[ids[~captured & ~usable]] = Outcome.NO_POLICY
        walking = ~captured & usable
        ids = ids[walking]
        if not ids.size:
            continue
        step = simulate_to_section(
            section_states(velocity[ids]), alphas[walking], params, roa, dt=dt
        )
        footstrikes[ids] += step.footstrikes
        velocity[ids] = step.next_velocity
        finished = step.outcome != Outcome.RETURNED
        outcome[ids[finished]] = step.outcome[finished]
    outcome[outcome == Outcome.PENDING] = Outcome.TIMEOUT
    return PolicyEvaluation(footstrikes, outcome)


@dataclass
class StepsToStandstillMap:
    angles: np.ndarray  # (n_theta,) initial angles, all before the section
    velocities: np.ndarray  # (n_velocity,) initial angular velocities
    footstrikes: np.ndarray  # (n_theta, n_velocity); NaN where the walker falls

    @property
    def captured_fraction(self):
        return float(np.mean(np.isfinite(self.footstrikes)))


def compute_steps_to_standstill_map(
    policy, params, roa, *, n_angle=121, n_velocity=121, min_angle=-0.6, dt=TABLE_DT
):
    """Footstrikes to standstill under the fewest-step policy from states before the section.

    For theta0 < 0 the walker reaches theta = 0 before any touchdown can
    happen, so the first landing angle is decided at the section. The
    approach uses ALPHA_MIN only as a placeholder that never takes effect.
    """
    angles = np.linspace(min_angle, 0.0, n_angle + 1)[:-1]  # strictly theta0 < 0
    velocities = np.linspace(0.0, max_section_velocity(params), n_velocity)
    theta, velocity = np.meshgrid(angles, velocities, indexing="ij")
    approach = simulate_to_section(
        np.array([theta.ravel(), velocity.ravel()]), ALPHA_MIN, params, roa, dt=dt
    )
    footstrikes = np.full(theta.size, np.nan)
    captured = approach.outcome == Outcome.CAPTURED
    footstrikes[captured] = approach.footstrikes[captured]
    returned = np.flatnonzero(approach.outcome == Outcome.RETURNED)
    walk = evaluate_policy(approach.next_velocity[returned], policy, params, roa, dt=dt)
    success = walk.outcome == Outcome.CAPTURED
    footstrikes[returned[success]] = (
        approach.footstrikes[returned[success]] + walk.footstrikes[success]
    )
    return StepsToStandstillMap(angles, velocities, footstrikes.reshape(theta.shape))


# --- Grid-resolution study ------------------------------------------------------


def validation_velocities(params, offset, n=VALIDATION_STARTS):
    """n section velocities spread over (0, v_max), each offset into a grid cell."""
    return (np.arange(n) + offset) / n * max_section_velocity(params)


@dataclass(frozen=True)
class GridTrial:
    velocity_points: int
    angle_points: int
    trials: int
    captured: int  # continuous rollouts that reached standstill
    reference_matches: int  # rollouts matching the fine-grid policy's footstrikes
    prediction_matches: int  # rollouts matching the table's own prediction

    @property
    def cells(self):
        return self.velocity_points * self.angle_points

    @property
    def label(self):
        return f"{self.velocity_points}x{self.angle_points}"

    def failed_criteria(self):
        """Names of the acceptance criteria this grid misses."""
        failures = []
        if self.captured < self.trials:
            failures.append("capture")
        if self.reference_matches / self.trials < MIN_AGREEMENT:
            failures.append("reference agreement")
        if self.prediction_matches / self.trials < MIN_AGREEMENT:
            failures.append("prediction agreement")
        return failures

    @property
    def passed(self):
        return not self.failed_criteria()


def score_grid(policy, starts, reference, params, roa):
    """Compare continuous rollouts of a grid policy with the reference and with its own table."""
    result = evaluate_policy(starts, policy, params, roa)
    predicted = policy.minimum[nearest_indices(policy.table.velocities, starts)]
    success = result.outcome == Outcome.CAPTURED
    n_velocity, n_alpha = policy.table.shape
    return GridTrial(
        velocity_points=n_velocity,
        angle_points=n_alpha,
        trials=len(starts),
        captured=int(success.sum()),
        reference_matches=int(
            np.sum(
                (result.footstrikes == reference.footstrikes)
                & (result.outcome == reference.outcome)
            )
        ),
        prediction_matches=int(np.sum((result.footstrikes == predicted) & success)),
    )


def select_coarsest_passing(trials):
    """Index of the passing grid with the fewest table cells."""
    passing = [i for i, trial in enumerate(trials) if trial.passed]
    if not passing:
        raise RuntimeError("No tested grid met the resolution criteria.")
    return min(passing, key=lambda i: trials[i].cells)


@dataclass
class ResolutionStudy:
    trials: list[GridTrial]  # ordered from coarsest to finest
    policies: list[StepPolicy]
    reference: StepPolicy
    selected: int

    @property
    def selected_policy(self):
        return self.policies[self.selected]

    @property
    def next_coarser(self):
        """The candidate just coarser than the selected one, or None."""
        return self.trials[self.selected - 1] if self.selected else None


def run_grid_resolution_study(params, roa, *, candidates=GRID_CANDIDATES):
    """Find the coarsest candidate grid that meets the acceptance criteria.

    The criteria are fixed before the sweep. On VALIDATION_STARTS off-grid
    section velocities, every rollout must capture, and at least
    MIN_AGREEMENT of them must match both the fine reference policy's
    footstrike count and the table's own prediction.
    """
    starts = validation_velocities(params, CALIBRATION_OFFSET)
    reference = solve_step_policy(
        build_transition_table(
            params,
            roa,
            n_velocity=REFERENCE_GRID[0],
            n_alpha=REFERENCE_GRID[1],
            dt=REFERENCE_DT,
        )
    )
    reference_result = evaluate_policy(starts, reference, params, roa)
    if np.any(reference_result.outcome != Outcome.CAPTURED):
        raise RuntimeError(
            "The fine reference policy failed; cannot score coarser grids."
        )
    trials, policies = [], []
    for n_velocity, n_alpha in sorted(candidates, key=lambda grid: grid[0] * grid[1]):
        policy = solve_step_policy(
            build_transition_table(params, roa, n_velocity=n_velocity, n_alpha=n_alpha)
        )
        trial = score_grid(policy, starts, reference_result, params, roa)
        print(f"  grid {trial.label}: passed={trial.passed} {trial}", flush=True)
        trials.append(trial)
        policies.append(policy)
    return ResolutionStudy(trials, policies, reference, select_coarsest_passing(trials))


@dataclass(frozen=True)
class NumericalChecks:
    roa_interior_samples: int
    roa_failures: int
    table_outcome_changes: int  # cells whose outcome changes at half the dt
    table_count_changes: int  # cells whose footstrike count changes at half the dt
    max_return_velocity_error: float  # rad/s, returning cells, dt vs dt/2
    holdout_trials: int
    holdout_captured: int
    holdout_reference_matches: int
    holdout_timestep_matches: int

    @property
    def passed(self):
        return (
            self.roa_failures == 0
            and self.holdout_captured == self.holdout_trials
            and self.holdout_reference_matches / self.holdout_trials >= MIN_AGREEMENT
            and self.holdout_timestep_matches == self.holdout_trials
        )


def verify_numerics(policy, reference, params, roa):
    """Independent checks: RoA interiors, dt sensitivity, and a held-out velocity set."""
    table = policy.table
    n_velocity, n_alpha = table.shape
    finer = build_transition_table(
        params, roa, n_velocity=n_velocity, n_alpha=n_alpha, dt=REFERENCE_DT
    )
    returned = (table.outcome == Outcome.RETURNED) & (finer.outcome == Outcome.RETURNED)
    velocity_error = np.abs(
        table.next_velocity[returned] - finer.next_velocity[returned]
    )
    holdout = validation_velocities(params, HOLDOUT_OFFSET)
    coarse = evaluate_policy(holdout, policy, params, roa, dt=TABLE_DT)
    half_dt = evaluate_policy(holdout, policy, params, roa, dt=REFERENCE_DT)
    fine = evaluate_policy(holdout, reference, params, roa, dt=REFERENCE_DT)
    roa_check = validate_roa(roa, params)

    def matches(a, b):
        return int(np.sum((a.footstrikes == b.footstrikes) & (a.outcome == b.outcome)))

    return NumericalChecks(
        roa_interior_samples=roa_check.tested,
        roa_failures=roa_check.failed,
        table_outcome_changes=int(np.sum(table.outcome != finer.outcome)),
        table_count_changes=int(np.sum(table.footstrikes != finer.footstrikes)),
        max_return_velocity_error=float(np.max(velocity_error, initial=0.0)),
        holdout_trials=len(holdout),
        holdout_captured=int(np.sum(coarse.outcome == Outcome.CAPTURED)),
        holdout_reference_matches=matches(coarse, fine),
        holdout_timestep_matches=matches(coarse, half_dt),
    )


@dataclass
class Example:
    velocity: float
    shortest: WalkerTrajectory
    longest: WalkerTrajectory


def find_validated_example(
    policy, reference, params, roa, *, preferred_velocity=4.0, min_footstrikes=3
):
    """Pick a section velocity whose example the continuous rollouts confirm.

    The fewest-step count must be at least min_footstrikes and the most-step
    count finite. Rollouts of both policies must stand still with exactly the
    table's counts, and the fine reference policy must agree. The search
    tries preferred_velocity first, then the grid from fast to slow.
    """
    for velocity in np.r_[preferred_velocity, policy.table.velocities[::-1]]:
        i = int(nearest_indices(policy.table.velocities, velocity))
        fewest, most = policy.minimum[i], policy.maximum[i]
        if not (
            np.isfinite(fewest) and fewest >= min_footstrikes and np.isfinite(most)
        ):
            continue
        runs = [
            rollout_policy(velocity, candidate, params, roa, longest=longest)
            for candidate in (policy, reference)
            for longest in (False, True)
        ]
        shortest, longest, fine_shortest, fine_longest = runs
        if (
            all(run.outcome == Outcome.STABILIZED for run in runs)
            and shortest.footstrikes == fewest == fine_shortest.footstrikes
            and longest.footstrikes == most == fine_longest.footstrikes
        ):
            return Example(float(velocity), shortest, longest)
    raise RuntimeError(
        "No validated example with a finite maximum and enough footstrikes."
    )
