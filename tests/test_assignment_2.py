"""Regression checks for Assignment 2: physics, events, planning, and study helpers."""

import numpy as np
import pytest

from integrators import rk4
from mae5110.assignment_2_physics import (
    ALPHA_MAX,
    ALPHA_MIN,
    EventKind,
    Outcome,
    RegionOfAttraction,
    WalkerEvent,
    WalkerTrajectory,
    advance_to_first_event,
    balancing_dynamics,
    compute_ankle_torque,
    estimate_roa,
    locate_angle_crossing,
    poincare_guard,
    simulate_to_section,
    simulate_walker,
    torque_limits,
    walk,
)
from mae5110.assignment_2_planning import (
    GridTrial,
    TransitionTable,
    build_transition_table,
    compute_steps_to_standstill_map,
    rollout_policy,
    select_coarsest_passing,
    solve_step_policy,
)
from models import inverted_pendulum_walker as model


@pytest.fixture(scope="module")
def params():
    return model.generate_params()


@pytest.fixture(scope="module")
def roa(params):
    return estimate_roa(params)


@pytest.fixture(scope="module")
def policy(params, roa):
    return solve_step_policy(
        build_transition_table(params, roa, n_velocity=81, n_alpha=21)
    )


# --- Model and guards -------------------------------------------------------------


def test_batched_guards_and_impact_match_single_states(params):
    rng = np.random.default_rng(0)
    previous = np.array([rng.uniform(-0.6, 0.6, 50), rng.uniform(-1, 4, 50)])
    following = previous + np.array(
        [rng.uniform(-0.2, 0.3, 50), rng.uniform(-0.5, 0.5, 50)]
    )
    alphas = rng.uniform(ALPHA_MIN, ALPHA_MAX, 50)
    batch_params = {**params, "angle_of_attack": alphas}
    touchdown = model.event_guard(previous, following, batch_params)
    section = poincare_guard(previous, following)
    reset = model.event_dynamics(following, batch_params)
    for k in range(50):
        single = {**params, "angle_of_attack": alphas[k]}
        assert touchdown[k] == model.event_guard(
            previous[:, k], following[:, k], single
        )
        assert section[k] == poincare_guard(previous[:, k], following[:, k])
        np.testing.assert_array_equal(
            reset[:, k], model.event_dynamics(following[:, k], single)
        )


def test_angle_crossing_agrees_for_single_state_and_batch(params):
    states = np.array([[-0.1, -0.05, 0.3], [1.0, 2.0, 1.5]])
    angles = np.array([0.0, 0.0, params["incline"] + ALPHA_MIN])
    elapsed, crossing = locate_angle_crossing(states, np.full(3, 0.2), angles, params)
    for k in range(3):
        single_elapsed, single_crossing = locate_angle_crossing(
            states[:, k], 0.2, angles[k], params
        )
        assert elapsed[k] == pytest.approx(single_elapsed, abs=1e-12)
        np.testing.assert_allclose(crossing[:, k], single_crossing, atol=1e-10)
        assert crossing[0, k] == angles[k]


@pytest.mark.parametrize(
    "state, dt, expected",
    [
        ([0.1, 0.2], 0.001, None),
        ([-0.1, 1.0], 0.15, "section"),
        ([0.4, 1.0], 0.1, "touchdown"),
        ([-0.1, 4.0], 0.2, "section"),  # both guards crossed: the section comes first
    ],
)
def test_advance_to_first_event(params, state, dt, expected):
    states = np.array(state)[:, None]
    alphas = np.array([params["angle_of_attack"]])
    trial = rk4.step(model.dynamics, 0.0, states, dt, params)
    reached, elapsed, section, touchdown = advance_to_first_event(
        states, alphas, dt, params
    )
    assert section[0] == (expected == "section")
    assert touchdown[0] == (expected == "touchdown")
    if expected is None:
        assert elapsed[0] == dt
        np.testing.assert_array_equal(reached, trial)
        return
    angle = 0.0 if expected == "section" else params["incline"] + alphas[0]
    assert 0 < elapsed[0] < dt
    assert reached[0, 0] == angle
    exact = rk4.step(model.dynamics, 0.0, states[:, 0], elapsed[0], params)
    assert exact[0] == pytest.approx(angle, abs=1e-10)


def test_landing_angle_must_place_section_between_reset_and_touchdown(params):
    with pytest.raises(ValueError):
        walk([0.0, 1.0], params["incline"] / 2, params, None)


# --- Ankle controller and RoA --------------------------------------------------------


def test_feedback_linearization_and_saturation(params):
    state = np.array([0.001, -0.002])
    kp = params["gravity"] / params["length"]
    expected = -kp * state[0] - 2 * np.sqrt(kp) * state[1]
    assert balancing_dynamics(0, state, params)[1] == pytest.approx(expected)
    np.testing.assert_allclose(torque_limits(params), [-0.981, 0.4905])
    torques = compute_ankle_torque(np.array([[1.0, -1.0], [10.0, -10.0]]), params)
    np.testing.assert_allclose(torques, torque_limits(params))


def test_roa_requires_four_corners():
    roa = RegionOfAttraction(
        np.array([-1.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([[True, True], [True, False]]),
    )
    assert roa.accepted_cells.shape == (1, 1)
    assert not roa.contains([-0.99, -0.99])
    assert not roa.contains([np.nan, 0.0])


# --- Continuous walking ------------------------------------------------------------


def test_capture_endpoint_and_zero_step_equilibrium(params, roa):
    standing = walk([0.0, 0.0], ALPHA_MIN, params, roa)
    assert standing.outcome == Outcome.CAPTURED
    assert standing.footstrikes == 0 and standing.times[-1] == 0
    assert simulate_walker([0.0, 0.0], params, roa).outcome == Outcome.STABILIZED
    one_step = walk([0.0, 0.8], ALPHA_MIN, params, roa)
    assert one_step.outcome == Outcome.CAPTURED and one_step.footstrikes == 1
    assert one_step.times[-1] > 0 and roa.contains(one_step.final_state)


def test_passive_return_matches_energy_balance(params):
    alpha, velocity = ALPHA_MIN, 2.0
    gamma, q = params["incline"], params["gravity"] / params["length"]
    expected = np.sqrt(
        np.cos(2 * alpha) ** 2 * (velocity**2 + 2 * q * (1 - np.cos(gamma + alpha)))
        + 2 * q * (np.cos(gamma - alpha) - 1)
    )
    run = walk([0.0, velocity], alpha, params, None, stop_at_section=True)
    assert run.outcome == Outcome.RETURNED and run.footstrikes == 1
    assert run.section_velocities()[-1] == pytest.approx(expected, abs=1e-8)


def test_impacts_are_recorded_as_pre_and_post_samples(params, roa):
    run = simulate_walker([0.01, 5.0], params, roa)
    impacts = run.impact_sample_indices()
    alpha = params["angle_of_attack"]
    assert run.outcome == Outcome.STABILIZED
    assert len(impacts) == run.footstrikes == len(run.event_times(EventKind.IMPACT))
    np.testing.assert_array_equal(run.times[impacts - 1], run.times[impacts])
    np.testing.assert_allclose(run.states[0, impacts - 1], params["incline"] + alpha)
    np.testing.assert_allclose(run.states[0, impacts], params["incline"] - alpha)
    balancing = np.isnan(run.alphas)
    assert balancing[-1] and not balancing[0]
    assert np.all(run.torques[~balancing] == 0)
    low, high = torque_limits(params)
    assert np.all((run.torques >= low) & (run.torques <= high))


def test_batch_agrees_with_single_trajectory(params, roa):
    velocities = np.array([0.0, 0.3, 0.5, 0.8, 1.0, 2.0, 4.0])
    for alpha in (ALPHA_MIN, ALPHA_MAX):
        batch = simulate_to_section(
            np.array([np.zeros_like(velocities), velocities]),
            alpha,
            params,
            roa,
            dt=0.002,
        )
        for k, velocity in enumerate(velocities):
            run = walk(
                [0.0, velocity], alpha, params, roa, dt=0.002, stop_at_section=True
            )
            assert batch.outcome[k] == run.outcome
            assert batch.footstrikes[k] == run.footstrikes
            if run.outcome == Outcome.RETURNED:
                assert batch.next_velocity[k] == pytest.approx(
                    run.final_state[1], abs=1e-8
                )


def test_trajectory_extension_shifts_time_and_replaces_shared_sample():
    start = WalkerTrajectory.at_rest([0.0, 1.0])
    segment = WalkerTrajectory(
        times=np.array([0.0, 0.5]),
        states=np.array([[0.0, 0.2], [1.0, 0.9]]),
        torques=np.zeros(2),
        alphas=np.full(2, ALPHA_MIN),
        events=[WalkerEvent(0.5, EventKind.IMPACT, np.array([0.2, 0.9]))],
        footstrikes=1,
        outcome=Outcome.RETURNED,
    )
    joined = start.extended(segment).extended(segment)
    np.testing.assert_array_equal(joined.times, [0.0, 0.5, 1.0])
    assert joined.alphas[0] == ALPHA_MIN
    assert joined.footstrikes == 2
    assert [e.time for e in joined.events] == [0.5, 1.0]


# --- Discrete-time planning ------------------------------------------------------


def _table(next_velocity, outcome, footstrikes):
    n = len(outcome)
    return TransitionTable(
        np.arange(n, dtype=float),
        np.array([ALPHA_MIN, ALPHA_MAX]),
        np.array(next_velocity, dtype=float),
        np.array(outcome),
        np.array(footstrikes),
    )


C, R, F = Outcome.CAPTURED, Outcome.RETURNED, Outcome.FELL
nan = np.nan


def test_shortest_longest_and_unbounded_self_loop():
    # State 0 captures; state 1 can capture or self-loop; state 2 reaches state 1.
    table = _table(
        [[nan, nan], [nan, 1.0], [1.0, nan], [nan, nan]],
        [[C, C], [C, R], [R, F], [F, F]],
        [[0, 0], [1, 1], [1, 0], [0, 0]],
    )
    policy = solve_step_policy(table)
    np.testing.assert_equal(policy.minimum, [0.0, 1.0, 2.0, np.inf])
    np.testing.assert_equal(policy.maximum, [0.0, np.inf, np.inf, -np.inf])
    np.testing.assert_equal(policy.max_action[1:], [-1, -1, -1])  # no finite optimum
    with pytest.raises(ValueError):
        policy.action(4.0)


def test_longer_cycle_is_unbounded():
    # 1 -> 2 -> 3 -> 1 is a 3-cycle whose only exit is capture from state 1; 4 feeds it.
    table = _table(
        [[nan, nan], [nan, 2.0], [nan, 3.0], [nan, 1.0], [2.0, nan]],
        [[C, C], [C, R], [F, R], [F, R], [R, F]],
        [[0, 0], [1, 1], [0, 1], [0, 1], [1, 0]],
    )
    policy = solve_step_policy(table)
    np.testing.assert_equal(policy.minimum, [0, 1, 3, 2, 4])
    np.testing.assert_equal(policy.maximum, [0, np.inf, np.inf, np.inf, np.inf])


def test_finite_longest_path():
    table = _table(
        [[nan, nan], [nan, 0.0], [nan, 1.0]],
        [[C, C], [C, R], [C, R]],
        [[0, 0], [1, 1], [1, 1]],
    )
    policy = solve_step_policy(table)
    np.testing.assert_equal(policy.minimum, [0, 1, 1])
    np.testing.assert_equal(policy.maximum, [0, 1, 2])


def test_vectorized_lookup_matches_single_lookup(policy):
    velocities = np.linspace(-0.5, policy.table.velocities[-1] + 0.5, 97)
    for longest in (False, True):
        alphas, usable = policy.actions(velocities, longest=longest)
        assert not usable[velocities < 0].any()
        assert not usable[velocities > policy.table.velocities[-1]].any()
        for velocity, alpha, ok in zip(velocities, alphas, usable):
            if ok:
                assert policy.action(velocity, longest=longest) == alpha
            else:
                with pytest.raises(ValueError):
                    policy.action(velocity, longest=longest)


@pytest.mark.parametrize("longest, expected_steps", [(False, 3), (True, 5)])
def test_policy_rollout_stands_and_respects_bounds(
    params, roa, policy, longest, expected_steps
):
    run = rollout_policy(4.0, policy, params, roa, longest=longest)
    assert run.outcome == Outcome.STABILIZED
    assert run.footstrikes == expected_steps
    assert np.linalg.norm(run.final_state) < 1e-3
    walking = np.isfinite(run.alphas)
    assert np.all(
        (run.alphas[walking] >= ALPHA_MIN) & (run.alphas[walking] <= ALPHA_MAX)
    )
    low, high = torque_limits(params)
    assert np.all((run.torques >= low - 1e-12) & (run.torques <= high + 1e-12))
    kinds = [event.kind for event in run.events]
    assert (
        kinds.count(EventKind.IMPACT)
        == kinds.count(EventKind.DECISION)
        == expected_steps
    )


def test_steps_to_standstill_map(params, roa, policy):
    steps_map = compute_steps_to_standstill_map(
        policy, params, roa, n_angle=12, n_velocity=15
    )
    theta, velocity = np.meshgrid(steps_map.angles, steps_map.velocities, indexing="ij")
    assert np.all(steps_map.angles < 0)
    inside = roa.contains(np.array([theta.ravel(), velocity.ravel()])).reshape(
        theta.shape
    )
    assert inside.any() and np.all(steps_map.footstrikes[inside] == 0)
    # Leaning back with no forward velocity falls backward.
    assert np.all(np.isnan(steps_map.footstrikes[steps_map.angles <= -0.1, 0]))
    finite = steps_map.footstrikes[np.isfinite(steps_map.footstrikes)]
    assert np.all(finite == np.round(finite))
    assert finite.max() <= np.max(policy.minimum[np.isfinite(policy.minimum)])


def test_grid_selection_and_failed_criteria():
    trials = [
        GridTrial(21, 6, 120, 110, 100, 90),
        GridTrial(41, 11, 120, 120, 120, 118),
        GridTrial(81, 21, 120, 120, 120, 119),
        GridTrial(161, 41, 120, 120, 120, 120),
    ]
    assert trials[0].failed_criteria() == [
        "capture",
        "reference agreement",
        "prediction agreement",
    ]
    assert trials[1].failed_criteria() == ["prediction agreement"]
    assert trials[2].passed
    assert select_coarsest_passing(trials) == 2
    with pytest.raises(RuntimeError):
        select_coarsest_passing(trials[:2])
