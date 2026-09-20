from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4
from models import inverted_pendulum_walker as model


def compute_ankle_torque(state, params):
    theta, velocity = state
    g, length, mass = params["gravity"], params["length"], params["mass"]
    kp = g / length
    kd = 2 * np.sqrt(kp)
    torque = mass * length**2 * (-kp * np.sin(theta) - kp * theta - kd * velocity)
    return np.clip(torque, -0.1 * mass * g * length, 0.05 * mass * g * length)


def balancing_dynamics(t, state, params):
    controlled_params = {**params, "ankle_torque": compute_ankle_torque(state, params)}
    return model.dynamics(t, state, controlled_params)


@dataclass
class RegionOfAttraction:
    angles: np.ndarray
    velocities: np.ndarray
    converged: np.ndarray

    def contains(self, state):
        theta, velocity = state
        if not (
            self.angles[0] <= theta <= self.angles[-1]
            and self.velocities[0] <= velocity <= self.velocities[-1]
        ):
            return False
        i = np.clip(
            np.searchsorted(self.angles, theta, side="right") - 1,
            0,
            len(self.angles) - 2,
        )
        j = np.clip(
            np.searchsorted(self.velocities, velocity, side="right") - 1,
            0,
            len(self.velocities) - 2,
        )
        return bool(self.converged[i : i + 2, j : j + 2].all())


def estimate_roa(params, *, grid_size=161, dt=0.01, duration=12.0, tolerance=1e-3):

    gamma = params["incline"]
    angles = np.linspace(gamma - np.pi / 2, gamma + np.pi / 2, grid_size)
    vmax = np.sqrt(2 * params["gravity"] / params["length"])
    velocities = np.linspace(-vmax, vmax, grid_size)
    theta, velocity = np.meshgrid(angles, velocities, indexing="ij")
    states = np.array([theta.ravel(), velocity.ravel()])
    alive = np.cos(states[0] - gamma) > 1e-12
    n_steps = int(np.ceil(duration / dt))
    dt = duration / n_steps
    for step in range(n_steps):
        indices = np.flatnonzero(alive)
        if not indices.size:
            break
        states[:, indices] = rk4.step(
            balancing_dynamics, step * dt, states[:, indices], dt, params
        )
        alive[indices] = (
            np.isfinite(states[:, indices]).all(axis=0)
            & (states[0, indices] > angles[0])
            & (states[0, indices] < angles[-1])
        )
    converged = (
        alive & (np.abs(states[0]) < tolerance) & (np.abs(states[1]) < tolerance)
    )
    return RegionOfAttraction(angles, velocities, converged.reshape(theta.shape))


def roa_event_guard(state, roa):
    return roa.contains(state)


def poincare_guard(previous_state, next_state):
    return previous_state[0] < 0 <= next_state[0] and next_state[1] > 0


def _locate_angle_crossing(t, state, dt, angle, params):
    low, high = 0.0, dt
    for _ in range(35):
        mid = (low + high) / 2
        trial = rk4.step(model.dynamics, t, state, mid, params)
        if trial[0] < angle:
            low = mid
        else:
            high = mid
    crossing = rk4.step(model.dynamics, t, state, high, params)
    crossing[0] = angle
    return high, crossing


def poincare_return_map(velocity, alpha, params, roa=None, *, dt=1e-3, duration=10.0):
    if not np.isfinite(velocity) or velocity <= 0:
        raise ValueError("The upright section requires positive finite velocity.")
    if not abs(params["incline"]) < alpha < np.pi / 2:
        raise ValueError("alpha must place touchdown after and reset before upright.")
    samples = []
    _, _, _, reason = simulate_walker(
        [0.0, velocity], {**params, "angle_of_attack": alpha}, roa,
        dt=dt, duration=duration, poincare_samples=samples, stop_at_section=True,
    )
    return (samples[-1][1] if reason == "returned" else None), reason

def step_to_step_transition():
    params = model.generate_params()
    roa = estimate_roa(params)
    g = params["gravity"]
    l = params["length"]

    v_res = 40
    a_res = 40
    velocities = np.linspace(0.04, np.sqrt(2 * g / l), v_res)
    alphas = np.linspace(np.pi/8, np.pi/7, a_res)

    V_grid, A_grid = np.meshgrid(velocities, alphas, indexing="ij")
    status_map = np.zeros_like(V_grid, dtype=int)

    print(f"Simulating {v_res * a_res} step transitions. This may take a minute...")

    for i in range(v_res):
        for j in range(a_res):
            v_k = V_grid[i, j]
            alpha = A_grid[i, j]

            next_v, reason = poincare_return_map(v_k, alpha, params, roa)

            if reason == "captured":
                status_map[i, j] = 1
            elif reason == "returned" and next_v is not None:
                status_map[i, j] = 2
            else:
                # fell, time limit, or invalid
                status_map[i, j] = -1

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    ax.scatter(
        V_grid[status_map == 1], A_grid[status_map == 1],
        color='green', marker='o', label='Captured (1-Step to RoA)'
    )
    ax.scatter(
        V_grid[status_map == 2], A_grid[status_map == 2],
        color='blue', marker='o', label='Continued Walking'
    )
    ax.scatter(
        V_grid[status_map == -1], A_grid[status_map == -1],
        color='red', marker='x', label='Fell / Failed'
    )

    ax.set(
        xlabel="Initial Angular Velocity, $\\dot{\\theta}_k$ (rad/s)",
        ylabel="Angle of Attack, $\\alpha$ (rad)",
        title="Step-to-Step Transition Outcomes"
    )
    ax.legend()

    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "step_to_step_scatter.png", dpi=180)
    print(f"Saved {output / 'step_to_step_scatter.png'}.")
    plt.show()

def simulate_walker(
    initial_state, params, roa, *, dt=1e-3, duration=10.0,
    poincare_samples=None, stop_at_section=False, stop_on_capture=True
    ):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("dt and duration must be positive and finite.")

    state = np.asarray(initial_state, dtype=float).copy()
    walking_params = {**params, "ankle_torque": 0.0}
    times, states = [0.0], [state.copy()]
    completed_steps = 0
    is_captured = False

    if roa is not None and roa_event_guard(state, roa):
        if stop_on_capture:
            return np.array(times), np.array(states).T, completed_steps, "captured"
        else:
            is_captured = True

    reason = "time limit"
    while times[-1] < duration:
        t = times[-1]
        step_dt = min(dt, duration - t)

        # Switch dynamics if captured
        if is_captured:
            next_state = rk4.step(balancing_dynamics, t, state, step_dt, params)
        else:
            next_state = rk4.step(model.dynamics, t, state, step_dt, walking_params)

        # Only check impacts and crossings if still passively walking
        section_crossed = False
        impact = False
        if not is_captured:
            section_crossed = poincare_guard(state, next_state)
            impact = model.event_guard(state, next_state, walking_params)

            if section_crossed and (not impact or 0 < params["incline"] + params["angle_of_attack"]):
                step_dt, next_state = _locate_angle_crossing(
                    t, state, step_dt, 0.0, walking_params
                )
                impact = False
            elif impact:
                step_dt, next_state = _locate_angle_crossing(
                    t, state, step_dt,
                    params["incline"] + params["angle_of_attack"], walking_params,
                )
                section_crossed = False

        if not is_captured and roa is not None and roa_event_guard(next_state, roa):
            if stop_on_capture:
                reason = "captured"
                break
            else:
                is_captured = True

        if not is_captured and impact:
            next_state = model.event_dynamics(next_state, params)
            completed_steps += 1
            if roa is not None and roa_event_guard(next_state, roa):
                if stop_on_capture:
                    reason = "captured"
                    break
                else:
                    is_captured = True

        times.append(t + step_dt)
        states.append(next_state.copy())

        if section_crossed:
            if poincare_samples is not None:
                poincare_samples.append((t + step_dt, float(next_state[1])))
            if stop_at_section:
                reason = "returned"
                break

        if not params["incline"] - np.pi / 2 < next_state[0] < params["incline"] + np.pi / 2:
            reason = "fell"
            break

        state = next_state


    return np.array(times), np.array(states).T, completed_steps, reason


def _roa_figure(roa):
    fig_roa, ax_roa = plt.subplots(layout="constrained")
    ax_roa.pcolormesh(
        roa.angles,
        roa.velocities,
        roa.converged.T,
        shading="nearest",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    ax_roa.set(
        xlabel="Angle (rad)",
        ylabel="Angular velocity (rad/s)",
        title="Estimated ankle-controller RoA (green)",
    )
    return fig_roa, ax_roa


def plot_roa():
    roa = estimate_roa(model.generate_params())
    fig, _ = _roa_figure(roa)
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "roa.png", dpi=180)
    print(f"Saved {output / 'roa.png'}.")
    plt.show()


def main():
    params = model.generate_params()
    roa = estimate_roa(params)
    samples = []
    time_traj, state_traj, steps, reason = simulate_walker(
        [0.01, 5.0], params, roa, poincare_samples=samples, stop_on_capture=False
    )
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    fig_roa, ax_roa = _roa_figure(roa)
    ax_roa.plot(*state_traj, color="tab:blue", linewidth=1, label="Passive walker")
    ax_roa.plot(*state_traj[:, -1], "ro", label=reason)
    ax_roa.axvline(0.0, color="gray", linestyle="--", label="Upright section")
    if samples:
        ax_roa.scatter(
            np.zeros(len(samples)), [v for _, v in samples],
            color="purple", zorder=5, label="Forward upright crossings",
        )
    ax_roa.legend()
    fig_roa.savefig(output / "roa.png", dpi=180)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        current_state = state_traj[:, index]
        if roa.contains(current_state):
            current_torque = compute_ankle_torque(current_state, params)
        else:
            current_torque = 0.0

        model.visualize(current_state, {**params, "ankle_torque": current_torque}, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s; {reason}")

    # Simulate at a small timestep, but render only 25 frames per second.
    fps = 25
    frame_indices = list(range(0, time_traj.size, round(1 / (fps * 1e-3))))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)
    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
    # To save an MP4 instead, install FFmpeg and use:
    # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
    print(f"{reason} at {time_traj[-1]:.3f} s ({steps} footstrikes).")
    print(f"Saved {output / 'roa.png'} and {output / 'walker.gif'}.")
    plt.show()


if __name__ == "__main__":
    main()
