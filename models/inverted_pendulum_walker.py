"""InvertedPendulumWalker starter model, with visualization provided.

Implement the model functions for Assignment 2. The visualizer works independently
of those functions; it draws a supplied state without advancing the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_params(*, incline=0.06):
    params = {
        "gravity": 9.81,  # gravitational acceleration (m/s^2)
        "length": 1.0,  # rod length (m)
        "mass": 1.0,  # point mass at end of rod (kg)
        "incline": incline,  # slope inclination angle (rad)
        "angle_of_attack": np.pi / 8,  # half angle between stance and swing legs
        "ankle_torque": 0.0,  # stance-ankle torque (N m), positive along theta
    }
    return params


def dynamics(t, state, params):
    g = params["gravity"]
    l = params["length"]
    tau = params["ankle_torque"]
    m = params["mass"]

    theta = state[0]
    theta_dot = state[1]

    theta_ddot = (g / l) * np.sin(theta) + tau / (m * (l**2))

    state_derivative = np.array([theta_dot, theta_ddot])

    return state_derivative


def event_guard(previous_state, next_state, params):
    """Return True where a forward swing crosses the touchdown angle.

    Works for single (2,) states and (2, N) batches; ``angle_of_attack`` may be
    a scalar or one value per column.
    """
    theta = previous_state[0]
    next_theta, next_theta_dot = next_state[0], next_state[1]
    theta_touchdown = params["incline"] + params["angle_of_attack"]
    return (
        (theta < theta_touchdown)
        & (theta_touchdown <= next_theta)
        & (next_theta_dot > 0)
    )


def event_dynamics(state, params):
    """Apply the touchdown reset to (2,) or (2, N) pre-impact states."""
    theta_dot = np.asarray(state, dtype=float)[1]
    gamma = params["incline"]
    alpha = params["angle_of_attack"]

    theta_plus = gamma - alpha
    theta_dot_plus = np.cos(2 * alpha) * theta_dot
    return np.stack(np.broadcast_arrays(theta_plus, theta_dot_plus))


def calculate_energy(state, params):
    """Return (kinetic, potential) energies in joules for (2,) or (2, N) states.

    Potential energy is measured relative to the current stance foot's height.
    Comparing it across downhill impacts requires accounting for the change
    in stance-foot height, which is not tracked by this two-state model.
    """
    theta, angular_velocity = np.asarray(state, dtype=float)
    mass = params["mass"]
    length = params["length"]
    gravity = params["gravity"]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(theta)
    return kinetic_energy, potential_energy


def visualize(
    state,
    params,
    ax=None,
    *,
    show_swing=True,
    stance_position=(0.0, 0.0),
    view_limits=None,
):
    """Draw one walker pose and return a Matplotlib Axes.

    Parameters
    ----------
    state : array-like, shape (2,)
        [theta, angular_velocity], in radians and radians/second. Theta is
        measured clockwise from upward vertical; positive x points right.
    params : dict
        ``length`` is the leg length in meters. ``incline`` is the ground's
        downhill slope angle in radians (positive slopes descend to the right).
        ``angle_of_attack`` is HALF the angle between the stance and forward swing
        legs, in radians; it is needed only when show_swing=True.
        ``ankle_torque`` (optional, default 0) is displayed in N m, with positive
        torque acting in the positive theta direction. Other keys are ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to clear and reuse. If omitted, create a figure. This function
        neither shows nor saves it: use plt.show() or ax.figure.savefig(...).
    show_swing : bool
        Draw a straight forward swing leg at the supplied angle_of_attack. Set False
        while the swing leg is held clear or while balancing. Swing motion is
        not part of the two-state model and is not inferred from theta.
    stance_position : pair of floats
        Current stance foot's (x, y) in meters, default (0, 0). The two-state
        model does not track translation; supply foot positions if desired.
        Ground passes through this point at the supplied incline.
    view_limits : (xmin, xmax, ymin, ymax), optional
        Fixed camera bounds in meters. By default the view follows the stance
        foot with bounds that fit both legs at any angle. Supply the same bounds
        each frame for a stationary world view.

    Notes
    -----
    Draws the supplied pose; contact events belong in the simulation.
    Reuse ax for frame sequences; use evenly spaced simulation times for playback
    at a fixed frame rate, and pass the parameters actually used at each frame.
    """
    state = np.asarray(state, dtype=float)
    foot = np.asarray(stance_position, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("state must contain two finite values: [theta, velocity].")
    if foot.shape != (2,) or not np.all(np.isfinite(foot)):
        raise ValueError("stance_position must contain two finite values: [x, y].")
    length = float(params["length"])
    incline = float(params["incline"])
    torque = float(params.get("ankle_torque", 0.0))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive.")
    if not np.isfinite(incline) or abs(incline) >= np.pi / 2:
        raise ValueError("incline must be finite and between -pi/2 and pi/2.")
    if not np.isfinite(torque):
        raise ValueError("ankle_torque must be finite.")
    if show_swing:
        angle_of_attack = float(params["angle_of_attack"])
        if not np.isfinite(angle_of_attack):
            raise ValueError("angle_of_attack must be finite.")

    if view_limits is None:
        radius = 2.15 * length
        view_limits = (
            foot[0] - radius,
            foot[0] + radius,
            foot[1] - radius,
            foot[1] + radius,
        )
    limits = np.asarray(view_limits, dtype=float)
    if (
        limits.shape != (4,)
        or not np.all(np.isfinite(limits))
        or limits[0] >= limits[1]
        or limits[2] >= limits[3]
    ):
        raise ValueError(
            "view_limits must be (xmin, xmax, ymin, ymax) with increasing bounds."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.clear()
    theta, angular_velocity = state
    hub = foot + length * np.array([np.sin(theta), np.cos(theta)])

    ground_x = np.array(limits[:2])
    ground_y = foot[1] - np.tan(incline) * (ground_x - foot[0])
    ax.fill_between(ground_x, ground_y, limits[2], color="#eee7dc", zorder=0)
    ax.plot(ground_x, ground_y, color="#7b6651", linewidth=2, label="Ground")
    ax.plot(
        [foot[0], foot[0]],
        [foot[1], foot[1] + 1.25 * length],
        ":",
        color="0.7",
        linewidth=1,
        label="Vertical",
    )

    if show_swing:
        swing_angle = theta - 2 * angle_of_attack
        swing_foot = hub - length * np.array([np.sin(swing_angle), np.cos(swing_angle)])
        swing_color = "#df8a25"
        ax.plot(
            [hub[0], swing_foot[0]],
            [hub[1], swing_foot[1]],
            "--",
            color=swing_color,
            linewidth=2.5,
            label="Swing leg",
            zorder=3,
        )
        ax.plot(
            *swing_foot,
            "o",
            color=swing_color,
            markersize=7,
            zorder=4,
            label="Swing foot",
        )

    stance_color = "#23699b"
    ax.plot(
        [foot[0], hub[0]],
        [foot[1], hub[1]],
        color=stance_color,
        linewidth=4,
        label="Stance leg",
        zorder=4,
    )
    ax.plot(*foot, "s", color="#333333", markersize=8, zorder=5, label="Stance foot")
    ax.plot(
        *hub,
        "o",
        color=stance_color,
        markeredgecolor="white",
        markersize=17,
        zorder=6,
        label="Hub",
    )
    ax.text(
        0.03,
        0.97,
        f"$\\theta$ = {theta:.3f} rad\n"
        f"$\\dot\\theta$ = {angular_velocity:.3f} rad/s\n"
        f"$\\tau$ = {torque:.3f} N m",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set(
        xlim=limits[:2],
        ylim=limits[2:],
        xlabel="x (m)",
        ylabel="y (m)",
        title="Inverted pendulum walker",
    )
    ax.set_aspect("equal", adjustable="box")
    return ax
