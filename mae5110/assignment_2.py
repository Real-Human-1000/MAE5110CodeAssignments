"""Assignment 2: ankle-controller RoA, walker animation, and footstep planning.

Run from the repository root:

    uv run python -m mae5110.assignment_2               # everything
    uv run python -m mae5110.assignment_2 --skip-study  # roa.png and walker.gif only

Every artifact is written to output/assignment_2/ (ignored by Git).
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mae5110 import assignment_2_plots as plots
from mae5110.assignment_2_physics import (
    ALPHA_MIN,
    Outcome,
    estimate_roa,
    integrate_balancing,
    simulate_walker,
)
from mae5110.assignment_2_planning import (
    compute_steps_to_standstill_map,
    find_validated_example,
    run_grid_resolution_study,
    verify_numerics,
)
from models import inverted_pendulum_walker as model

OUTPUT_DIR = Path("output/assignment_2")
DEMO_INITIAL_STATE = (0.01, 5.0)  # fixed-alpha demo: fast enough for several steps


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    params = model.generate_params()
    roa = estimate_roa(params)
    print_roa_summary(roa)

    demo = simulate_walker(DEMO_INITIAL_STATE, params, roa)
    print(
        f"Fixed-alpha demo: {demo.outcome.name.lower()} after {demo.footstrikes} "
        f"footstrikes at t = {demo.times[-1]:.2f} s."
    )
    roa_figure = plots.plot_region_of_attraction(
        roa, sample_balancing_runs(params), demo, params
    )
    plots.save_figure(roa_figure, output / "roa.png", show=args.show)
    plots.save_animation(
        plots.animate_walker(demo, params), output / "walker.gif", show=args.show
    )

    if not args.skip_study:
        run_footstep_study(params, roa, output, show=args.show)
    if args.show:
        plt.show()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR, help="output directory"
    )
    parser.add_argument(
        "--skip-study", action="store_true", help="only make roa.png and walker.gif"
    )
    parser.add_argument("--show", action="store_true", help="also display the figures")
    return parser.parse_args(argv)


def sample_balancing_runs(params):
    """Closed-loop runs from the post-impact angle incline - ALPHA_MIN over a velocity fan."""
    velocities = np.linspace(0.5, 1.7, 9)
    starts = np.array(
        [np.full_like(velocities, params["incline"] - ALPHA_MIN), velocities]
    )
    return integrate_balancing(starts, params, dt=0.01, duration=6.0, record=True)


def run_footstep_study(params, roa, output, *, show=False):
    """Discrete-time study: resolution sweep, checks, example, map, figures, and data."""
    print("Scoring candidate grids against the fine reference policy...", flush=True)
    study = run_grid_resolution_study(params, roa)
    policy = study.selected_policy
    checks = verify_numerics(policy, study.reference, params, roa)
    print(f"Independent checks: {checks}")
    if not checks.passed:
        raise RuntimeError(
            "Independent validation failed; refine the grids before reporting."
        )
    example = find_validated_example(policy, study.reference, params, roa)
    steps_map = compute_steps_to_standstill_map(policy, params, roa)

    save_study_data(output, study, example)
    figures = {
        "return_map.png": plots.plot_return_map(policy.table),
        "state_action_table.png": plots.plot_state_action_table(policy),
        "steps_to_standstill.png": plots.plot_steps_to_standstill(
            policy, example.velocity
        ),
        "steps_to_standstill_map.png": plots.plot_steps_to_standstill_map(
            steps_map, roa
        ),
        "example_trajectories.png": plots.plot_example_trajectories(
            example, roa, params
        ),
        "grid_resolution.png": plots.plot_grid_resolution(study),
    }
    for name, figure in figures.items():
        plots.save_figure(figure, output / name, show=show)
    print_study_summary(study, steps_map)
    print(
        f"Example v0 = {example.velocity:g} rad/s: fewest {example.shortest.footstrikes}, "
        f"most {example.longest.footstrikes} footstrikes. Results in {output}."
    )


# --- Data files ------------------------------------------------------------------


def save_study_data(output, study, example):
    policy = study.selected_policy
    policy.table.save(output / "transitions.npz")
    np.savez_compressed(
        output / "policy.npz",
        velocities=policy.table.velocities,
        alphas=policy.table.alphas,
        minimum=policy.minimum,
        maximum=policy.maximum,
        min_action=policy.min_action,
        max_action=policy.max_action,
    )
    example.shortest.save(output / "minimum_trajectory.npz")
    example.longest.save(output / "maximum_trajectory.npz")
    write_grid_resolution_csv(output / "grid_resolution.csv", study.trials)
    write_example_events_csv(output / "example_events.csv", example)


def write_grid_resolution_csv(path, trials):
    rows = [
        {
            **asdict(trial),
            "cells": trial.cells,
            "passed": trial.passed,
            "failed_criteria": "; ".join(trial.failed_criteria()),
        }
        for trial in trials
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_example_events_csv(path, example):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["policy", "time", "event", "theta", "velocity", "alpha"])
        for label, trajectory in (
            ("fewest", example.shortest),
            ("most", example.longest),
        ):
            for event in trajectory.events:
                alpha = "" if np.isnan(event.alpha) else event.alpha
                writer.writerow([label, event.time, event.kind, *event.state, alpha])


# --- Printed summaries ---------------------------------------------------------------


def section_capture_limit(roa):
    """Largest section velocity that is captured immediately."""
    velocities = np.linspace(0, roa.velocities[-1], 4001)
    inside = roa.contains([np.zeros_like(velocities), velocities])
    return velocities[inside].max()


def return_map_slows_down(table):
    """True if every returning transition ends slower than it started (v_{k+1} < v_k)."""
    returned = table.outcome == Outcome.RETURNED
    return bool(
        np.all(
            table.next_velocity[returned]
            < np.broadcast_to(table.velocities[:, None], table.shape)[returned]
        )
    )


def print_roa_summary(roa):
    accepted = int(roa.accepted_cells.sum())
    print(
        f"RoA: {len(roa.angles)} x {len(roa.velocities)} grid, {accepted} of "
        f"{roa.accepted_cells.size} cells accepted; section velocities up to "
        f"{section_capture_limit(roa):.3f} rad/s are captured without a step."
    )


def print_study_summary(study, steps_map):
    selected = study.trials[study.selected]
    print(
        f"Selected grid: {selected.label} ({selected.reference_matches}/{selected.trials} "
        f"reference, {selected.prediction_matches}/{selected.trials} prediction matches)."
    )
    coarser = study.next_coarser
    if coarser is not None:
        print(
            f"Next coarser grid {coarser.label} fails on {', '.join(coarser.failed_criteria())}: "
            f"{coarser.captured}/{coarser.trials} captured, "
            f"{coarser.reference_matches}/{coarser.trials} reference, "
            f"{coarser.prediction_matches}/{coarser.trials} prediction matches."
        )
    slows = return_map_slows_down(study.selected_policy.table)
    print(f"Every returning transition slows the walker (v_k+1 < v_k): {slows}.")
    print(
        f"Steps-to-standstill map: {steps_map.captured_fraction:.0%} of initial conditions "
        f"reach standstill, needing at most {int(np.nanmax(steps_map.footstrikes))} footstrikes."
    )


if __name__ == "__main__":
    main()
