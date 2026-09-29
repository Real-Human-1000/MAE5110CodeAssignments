import functools
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator, NullLocator

from mae5110.assignment_2_physics import (
    ALPHA_MAX,
    ALPHA_MIN,
    EventKind,
    Outcome,
    torque_limits,
)
from mae5110.assignment_2_planning import MIN_AGREEMENT
from models import inverted_pendulum_walker as model

# Palette: series slots, ink, status, and one ordinal blue ramp for step counts.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS_LINE, NEUTRAL_FILL = "#e1e0d9", "#c3c2b7", "#e6e4dd"
WARNING, CRITICAL = "#fab219", "#d03b3b"
ROA_FILL, ROA_EDGE = "#b7d3f6", "#1c5cab"
BLUE_RAMP = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
DPI = 180

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS_LINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS_LINE,
    "ytick.color": AXIS_LINE,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "font.size": 9,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
}

THETA, THETA_DOT, ALPHA = r"$\theta$", r"$\dot\theta$", r"$\alpha$"


def _styled(plot_function):
    """Build the figure under the shared style."""

    @functools.wraps(plot_function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return plot_function(*args, **kwargs)

    return wrapper


def save_figure(figure, path, *, show=False):
    """Write a figure to disk; keep it open only if it will be shown."""
    figure.savefig(path, dpi=DPI)
    if not show:
        plt.close(figure)
    print(f"Saved {path}")


def step_colors(count):
    """``count`` ordered colors from light to dark along the blue ramp."""
    positions = np.linspace(0, len(BLUE_RAMP) - 1, max(count, 1)).round().astype(int)
    return [BLUE_RAMP[i] for i in positions[:count]]


def _step_label(steps):
    return f"{steps} footstrike" + ("" if steps == 1 else "s")


def _with_impact_breaks(trajectory, values):
    """Insert NaN between each impact's pre/post samples so lines do not join the jump."""
    return np.insert(
        np.asarray(values, dtype=float),
        trajectory.impact_sample_indices(),
        np.nan,
        axis=-1,
    )


def _draw_roa(ax, roa, *, outline=True):
    """Fill the accepted RoA cells (the cells the capture guard uses)."""
    cells = np.ma.masked_where(~roa.accepted_cells.T, roa.accepted_cells.T)
    ax.pcolormesh(
        roa.angles,
        roa.velocities,
        cells,
        cmap=ListedColormap([ROA_FILL]),
        shading="flat",
    )
    if outline:
        centers_theta = (roa.angles[:-1] + roa.angles[1:]) / 2
        centers_velocity = (roa.velocities[:-1] + roa.velocities[1:]) / 2
        ax.contour(
            centers_theta,
            centers_velocity,
            roa.accepted_cells.T.astype(float),
            levels=[0.5],
            colors=ROA_EDGE,
            linewidths=0.8,
        )
    return Patch(
        facecolor=ROA_FILL,
        edgecolor=ROA_EDGE,
        linewidth=0.8,
        label="RoA (grid estimate)",
    )


def _draw_walk(ax, trajectory, color, label):
    """Phase-plane line broken at impacts, with faint connectors for the resets."""
    states = _with_impact_breaks(trajectory, trajectory.states)
    ax.plot(states[0], states[1], color=color, label=label)
    for index in trajectory.impact_sample_indices():
        pre, post = trajectory.states[:, index - 1], trajectory.states[:, index]
        ax.plot(
            [pre[0], post[0]],
            [pre[1], post[1]],
            color=color,
            linewidth=0.8,
            linestyle=":",
            alpha=0.6,
        )


def _event_states(trajectory, kind):
    return (
        np.array([e.state for e in trajectory.events if e.kind == kind])
        .reshape(-1, 2)
        .T
    )


# --- Region of attraction and animation --------------------------------------


@_styled
def plot_region_of_attraction(roa, balancing, demo, params):
    """RoA with closed-loop samples (left) and the RoA guard stopping a fixed-alpha walk (right)."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")

    roa_handle = _draw_roa(left, roa)
    theta = np.linspace(roa.angles[0], roa.angles[-1], 400)
    omega = np.sqrt(params["gravity"] / params["length"])
    left.plot(
        theta,
        -2 * omega * np.sin(theta / 2),
        color=INK_MUTED,
        linewidth=1,
        linestyle=(0, (4, 3)),
    )
    for column, converged in enumerate(balancing.converged):
        color = ROA_EDGE if converged else CRITICAL
        left.plot(
            balancing.history[:, 0, column],
            balancing.history[:, 1, column],
            color=color,
            linewidth=1.2,
        )
        left.plot(*balancing.history[0, :, column], "o", color=color, markersize=4)
    section_velocity = np.linspace(0, roa.velocities[-1], 2001)
    captured = section_velocity[
        roa.contains([np.zeros_like(section_velocity), section_velocity])
    ]
    left.axvline(0.0, color=INK_MUTED, linewidth=0.8)
    if not captured.size:
        raise ValueError("The RoA estimate does not contain any upright state.")
    left.plot(
        [0, 0],
        [captured.min(), captured.max()],
        color=INK,
        linewidth=4,
        solid_capstyle="butt",
    )
    left.set(
        xlim=(roa.angles[0], roa.angles[-1]),
        ylim=(roa.velocities[0], roa.velocities[-1]),
        xlabel=f"{THETA} (rad)",
        ylabel=f"{THETA_DOT} (rad/s)",
        title="Ankle-controller region of attraction",
    )
    left.legend(
        handles=[
            roa_handle,
            Line2D(
                [],
                [],
                color=INK_MUTED,
                linewidth=1,
                linestyle=(0, (4, 3)),
                label="Passive separatrix (zero torque)",
            ),
            Line2D(
                [],
                [],
                color=ROA_EDGE,
                marker="o",
                markersize=4,
                label="Closed loop: balanced",
            ),
            Line2D(
                [],
                [],
                color=CRITICAL,
                marker="o",
                markersize=4,
                label="Closed loop: fell",
            ),
            Line2D(
                [],
                [],
                color=INK,
                linewidth=4,
                label=f"Section velocities captured at once (up to {captured.max():.2f} rad/s)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
    )

    _draw_roa(right, roa, outline=False)
    _draw_walk(right, demo, BLUE, "Passive walking")
    balancing_samples = np.isnan(demo.alphas)
    right.plot(
        *demo.states[:, balancing_samples],
        color=ORANGE,
        label="Balancing (ankle torque on)",
    )
    sections = _event_states(demo, EventKind.SECTION)
    right.plot(
        *sections,
        "o",
        color=INK_SECONDARY,
        markerfacecolor="white",
        markersize=6,
        label="Section crossings",
    )
    right.plot(
        *_event_states(demo, EventKind.CAPTURE),
        "X",
        color=INK,
        markersize=8,
        label="Captured by the RoA guard",
    )
    right.axvline(0.0, color=INK_MUTED, linewidth=0.8)
    alpha = demo.alphas[0]
    right.set(
        xlim=(-0.6, 0.6),
        ylim=(-0.6, max(5.5, demo.states[1].max() + 0.3)),
        xlabel=f"{THETA} (rad)",
        ylabel=f"{THETA_DOT} (rad/s)",
        title=(
            f"Fixed {ALPHA} = {alpha:.3f} rad walk from [{demo.states[0, 0]:.2f}, {demo.states[1, 0]:.1f}]: "
            f"{demo.footstrikes} footstrikes, then balanced"
        ),
    )
    right.legend(loc="upper left")
    return figure


class WalkerAnimation(NamedTuple):
    figure: plt.Figure
    animation: FuncAnimation
    fps: int


def _stance_positions(trajectory, params):
    """Stance-foot position after 0, 1, 2, ... footstrikes (each step spans 2 l sin(alpha))."""
    alphas = trajectory.alphas[trajectory.impact_sample_indices() - 1]
    gamma = params["incline"]
    steps = (
        2
        * params["length"]
        * np.sin(alphas)[:, None]
        * np.array([np.cos(gamma), -np.sin(gamma)])
    )
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def animate_walker(trajectory, params, *, fps=25):
    """Animate a trajectory at uniform simulated time, one frame every 1/fps s.

    Torque and landing angle come from the trajectory itself, and the swing
    leg is hidden while the ankle controller balances.
    """
    frame_times = np.append(
        np.arange(0.0, trajectory.times[-1], 1 / fps), trajectory.times[-1]
    )
    frame_samples = np.searchsorted(trajectory.times, frame_times, side="right") - 1
    impact_times = trajectory.event_times(EventKind.IMPACT)
    feet = _stance_positions(trajectory, params)
    figure, ax = plt.subplots(figsize=(6, 6), layout="constrained")

    def draw(frame):
        index = frame_samples[frame]
        footstrikes = int(
            np.searchsorted(impact_times, trajectory.times[index], side="right")
        )
        alpha = trajectory.alphas[index]
        walking = bool(np.isfinite(alpha))
        frame_params = {
            **params,
            "ankle_torque": trajectory.torques[index],
            "angle_of_attack": alpha if walking else params["angle_of_attack"],
        }
        model.visualize(
            trajectory.states[:, index],
            frame_params,
            ax=ax,
            show_swing=walking,
            stance_position=feet[footstrikes],
        )
        phase = "walking" if walking else "balancing"
        ax.set_title(
            f"t = {trajectory.times[index]:.2f} s, footstrikes: {footstrikes}, {phase}"
        )

    animation = FuncAnimation(
        figure, draw, frames=len(frame_samples), interval=1000 / fps, repeat=False
    )
    return WalkerAnimation(figure, animation, fps)


def save_animation(walker_animation, path, *, show=False):
    walker_animation.animation.save(path, writer=PillowWriter(fps=walker_animation.fps))
    if not show:
        plt.close(walker_animation.figure)
    print(f"Saved {path}")


# --- Step-to-step map and policies -------------------------------------------


@_styled
def plot_return_map(table):
    """Section-to-section map v_k -> v_{k+1} for the smallest, middle, and largest alpha."""
    figure, ax = plt.subplots(figsize=(6.5, 5.6), layout="constrained")
    velocities = table.velocities
    captured_by_some = (table.outcome == Outcome.CAPTURED).any(axis=1)
    ax.fill_between(
        velocities,
        0,
        velocities[-1],
        where=captured_by_some,
        step="mid",
        color=ROA_FILL,
        alpha=0.6,
        linewidth=0,
        label=f"Some {ALPHA} captures in one step",
    )
    for color, column in zip(
        (BLUE, ORANGE, AQUA), (0, len(table.alphas) // 2, len(table.alphas) - 1)
    ):
        returned = table.outcome[:, column] == Outcome.RETURNED
        ax.plot(
            velocities,
            np.where(returned, table.next_velocity[:, column], np.nan),
            color=color,
            label=f"{ALPHA} = {table.alphas[column]:.3f} rad",
        )
    ax.plot(
        [0, velocities[-1]],
        [0, velocities[-1]],
        color=INK_MUTED,
        linewidth=1,
        linestyle=(0, (4, 3)),
    )
    ax.annotate(
        "$v_{k+1} = v_k$",
        (0.8 * velocities[-1], 0.8 * velocities[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        color=INK_SECONDARY,
        ha="center",
        va="center",
        rotation=45,
    )
    ax.set(
        xlim=(0, velocities[-1]),
        ylim=(0, velocities[-1]),
        aspect="equal",
        xlabel=f"$v_k$: {THETA_DOT} at the section (rad/s)",
        ylabel=f"$v_{{k+1}}$: {THETA_DOT} at the next section (rad/s)",
        title=rf"Return map on the section {THETA} = 0 ({THETA_DOT} > 0)",
    )
    ax.legend(loc="upper left")
    return figure


def _outcome_categories(outcome):
    categories = [
        (Outcome.CAPTURED, ROA_EDGE, "Captured (standing next)"),
        (Outcome.RETURNED, NEUTRAL_FILL, "Returns to the section"),
        (Outcome.FELL, CRITICAL, "Falls"),
        (Outcome.TIMEOUT, WARNING, "Times out"),
    ]
    return [category for category in categories if np.any(outcome == category[0])]


def _discrete_mesh(ax, x, y, codes, colors, *, bad_color=NEUTRAL_FILL):
    """pcolormesh of integer codes 0..len(colors)-1 with NaN drawn in bad_color."""
    colormap = ListedColormap(colors)
    colormap.set_bad(bad_color)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))
    ax.pcolormesh(
        x, y, np.ma.masked_invalid(codes).T, cmap=colormap, norm=norm, shading="nearest"
    )
    ax.grid(False)


@_styled
def plot_state_action_table(policy):
    """One-step outcomes (left) and the backed-out steps to standstill (right)."""
    table = policy.table
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.8), layout="constrained", sharey=True
    )

    categories = _outcome_categories(table.outcome)
    codes = np.full(table.shape, np.nan)
    for code, (outcome, _, _) in enumerate(categories):
        codes[table.outcome == outcome] = code
    _discrete_mesh(
        left, table.velocities, table.alphas, codes, [c[1] for c in categories]
    )
    left.legend(
        handles=[Patch(facecolor=color, label=label) for _, color, label in categories],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )
    left.set(
        xlabel="$v_k$ (rad/s)",
        ylabel=f"Landing half-angle {ALPHA} (rad)",
        title=f"One-step outcome of each ($v_k$, {ALPHA})",
    )

    steps_to_go = policy.action_values()
    most = int(np.nanmax(np.where(np.isfinite(steps_to_go), steps_to_go, np.nan)))
    colors = step_colors(most + 1)
    _discrete_mesh(
        right,
        table.velocities,
        table.alphas,
        np.where(np.isfinite(steps_to_go), steps_to_go, np.nan),
        colors,
    )
    velocities = table.velocities
    for actions, style, label in (
        (
            policy.min_action,
            {
                "color": "white",
                "path_effects": [withStroke(linewidth=3.2, foreground=INK)],
            },
            "Fewest-steps policy",
        ),
        (policy.max_action, {"color": ORANGE}, "Most-steps policy"),
    ):
        usable = actions >= 0
        right.step(
            velocities[usable],
            table.alphas[actions[usable]],
            where="mid",
            linewidth=1.6,
            label=label,
            **style,
        )
    handles = [
        Patch(facecolor=color, label=_step_label(steps))
        for steps, color in enumerate(colors)
    ]
    handles.append(Patch(facecolor=NEUTRAL_FILL, label="Never stands still"))
    handles.extend(right.get_legend_handles_labels()[0])
    right.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    right.set(
        xlabel="$v_k$ (rad/s)",
        title=f"Footstrikes to standstill after choosing {ALPHA} at $v_k$",
    )
    return figure


@_styled
def plot_steps_to_standstill(policy, example_velocity):
    """Fewest and most footstrikes to standstill, and the chosen alpha, per section velocity."""
    table = policy.table
    velocities = table.velocities
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 7), layout="constrained", sharex=True, height_ratios=(3, 2)
    )
    for values, color, width, label in (
        (policy.maximum, ORANGE, 3.0, "Most footstrikes before capture"),
        (policy.minimum, BLUE, 1.6, "Fewest footstrikes"),
    ):
        top.step(
            velocities,
            np.where(np.isfinite(values), values, np.nan),
            where="mid",
            color=color,
            linewidth=width,
            label=label,
        )
    finite_max = policy.maximum[np.isfinite(policy.maximum)]
    ceiling = (finite_max.max() if finite_max.size else 0) + 1
    unbounded = np.isposinf(policy.maximum)
    if unbounded.any():
        top.plot(
            velocities[unbounded],
            np.full(unbounded.sum(), ceiling),
            "^",
            color=ORANGE,
            markersize=4,
            label="Unbounded in the grid graph",
        )
    unreachable = np.isinf(policy.minimum)
    if unreachable.any():
        top.plot(
            velocities[unreachable],
            np.full(unreachable.sum(), -0.5),
            "x",
            color=CRITICAL,
            markersize=4,
            label="No capture possible",
        )
    top.axvline(example_velocity, color=INK_MUTED, linewidth=0.8)
    top.text(
        example_velocity,
        0.03,
        f"example $v_0$ = {example_velocity:.2f} rad/s ",
        color=INK_SECONDARY,
        ha="right",
        va="bottom",
        transform=top.get_xaxis_transform(),
    )
    top.yaxis.set_major_locator(MaxNLocator(integer=True))
    top.set(
        ylabel="Footstrikes to standstill",
        title=r"Footstrikes to standstill from the section state [0, $v_0$]",
    )
    top.legend(loc="upper left")

    for actions, color, width, label in (
        (policy.max_action, ORANGE, 3.0, "Most-steps policy"),
        (policy.min_action, BLUE, 1.6, "Fewest-steps policy"),
    ):
        usable = actions >= 0
        bottom.step(
            velocities[usable],
            table.alphas[actions[usable]],
            where="mid",
            color=color,
            linewidth=width,
            label=label,
        )
    for bound, name in ((ALPHA_MIN, r"$\pi/8$"), (ALPHA_MAX, r"$\pi/7$")):
        bottom.axhline(bound, color=INK_MUTED, linewidth=0.8)
        bottom.text(velocities[-1], bound, f" {name}", color=INK_SECONDARY, va="center")
    bottom.set(
        xlabel="Section velocity $v_0$ (rad/s)",
        ylabel=f"{ALPHA} (rad)",
        title=f"Landing half-angle {ALPHA} chosen at the section",
    )
    bottom.legend(loc="lower left")
    return figure


@_styled
def plot_steps_to_standstill_map(steps_map, roa):
    """Footstrikes to standstill over initial conditions (theta0, theta_dot0) before the section."""
    figure, ax = plt.subplots(figsize=(8.5, 5.6), layout="constrained")
    finite = steps_map.footstrikes[np.isfinite(steps_map.footstrikes)]
    colors = step_colors(int(finite.max()) + 1)
    _discrete_mesh(
        ax, steps_map.angles, steps_map.velocities, steps_map.footstrikes, colors
    )
    centers_theta = (roa.angles[:-1] + roa.angles[1:]) / 2
    centers_velocity = (roa.velocities[:-1] + roa.velocities[1:]) / 2
    ax.contour(
        centers_theta,
        centers_velocity,
        roa.accepted_cells.T.astype(float),
        levels=[0.5],
        colors=INK,
        linewidths=0.9,
    )
    half_step = (steps_map.angles[1] - steps_map.angles[0]) / 2
    ax.set(
        xlim=(steps_map.angles[0] - half_step, 0.0),
        ylim=(steps_map.velocities[0], steps_map.velocities[-1]),
        xlabel=f"Initial angle {THETA}$_0$ (rad); the section {THETA} = 0 is the right edge",
        ylabel=f"Initial angular velocity {THETA_DOT}$_0$ (rad/s)",
        title=f"Footstrikes to standstill from ({THETA}$_0$, {THETA_DOT}$_0$), fewest-steps policy",
    )
    handles = [
        Patch(facecolor=color, label=_step_label(steps))
        for steps, color in enumerate(colors)
    ]
    handles.append(Patch(facecolor=NEUTRAL_FILL, label="Falls"))
    handles.append(Line2D([], [], color=INK, linewidth=0.9, label="RoA boundary"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return figure


@_styled
def plot_example_trajectories(example, roa, params):
    """Fewest- and most-footstrike walks from the same section state, down to standstill."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    phase, speed, angle, torque = axes.flat
    runs = (
        (example.shortest, BLUE, f"Fewest footstrikes: {example.shortest.footstrikes}"),
        (example.longest, ORANGE, f"Most footstrikes: {example.longest.footstrikes}"),
    )
    _draw_roa(phase, roa)
    for trajectory, color, label in runs:
        _draw_walk(phase, trajectory, color, label)
        phase.plot(
            *_event_states(trajectory, EventKind.DECISION),
            "o",
            color=color,
            markerfacecolor="white",
            markersize=6,
        )
        phase.plot(
            *_event_states(trajectory, EventKind.CAPTURE), "X", color=INK, markersize=8
        )
        speed.plot(trajectory.times, trajectory.states[1], color=color)
        decisions = [e.alpha for e in trajectory.events if e.kind == EventKind.DECISION]
        angle.plot(np.arange(1, len(decisions) + 1), decisions, "o-", color=color)
        torque.plot(trajectory.times, trajectory.torques, color=color)
    phase.axvline(0.0, color=INK_MUTED, linewidth=0.8)
    states = np.hstack([example.shortest.states, example.longest.states])
    phase.set(
        xlim=(states[0].min() - 0.08, states[0].max() + 0.08),
        ylim=(min(states[1].min(), 0) - 0.3, states[1].max() + 0.3),
        xlabel=f"{THETA} (rad)",
        ylabel=f"{THETA_DOT} (rad/s)",
        title="Phase portrait (dotted: impact resets)",
    )
    speed.set(
        xlabel="Time (s)", ylabel=f"{THETA_DOT} (rad/s)", title="Angular velocity"
    )
    for bound, name in ((ALPHA_MIN, r"$\pi/8$"), (ALPHA_MAX, r"$\pi/7$")):
        angle.axhline(bound, color=INK_MUTED, linewidth=0.8)
        angle.text(
            1,
            bound,
            f"{name} ",
            color=INK_SECONDARY,
            ha="right",
            va="center",
            transform=angle.get_yaxis_transform(),
        )
    angle.xaxis.set_major_locator(MaxNLocator(integer=True))
    angle.set(
        xlabel="Step $k$",
        ylabel=f"{ALPHA} (rad)",
        title="Landing half-angle chosen at each section",
    )
    for bound in torque_limits(params):
        torque.axhline(bound, color=INK_MUTED, linewidth=0.8)
    torque.set(
        xlabel="Time (s)",
        ylabel=r"$\tau$ (N m)",
        title="Ankle torque (limits: -0.1 mgl and 0.05 mgl)",
    )

    handles = [Line2D([], [], color=color, label=label) for _, color, label in runs]
    handles += [
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label=f"{ALPHA} decision at the section",
        ),
        Line2D([], [], color=INK, marker="X", linestyle="none", label="Captured"),
        Patch(facecolor=ROA_FILL, edgecolor=ROA_EDGE, label="RoA"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    figure.suptitle(
        f"Walking from [0, {example.velocity:.2f}] to standstill: fewest {example.shortest.footstrikes}, "
        f"most {example.longest.footstrikes} footstrikes",
        color=INK,
    )
    return figure


@_styled
def plot_grid_resolution(study):
    """Agreement of continuous rollouts with each candidate grid's predictions."""
    trials = study.trials
    cells = np.array([trial.cells for trial in trials])
    count = trials[0].trials
    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    metrics = (
        (
            "Prediction agreement (table vs rollout)",
            BLUE,
            "o",
            [t.prediction_matches for t in trials],
        ),
        (
            "Reference agreement (vs fine policy)",
            ORANGE,
            "s",
            [t.reference_matches for t in trials],
        ),
        ("Captured", AQUA, "^", [t.captured for t in trials]),
    )
    for label, color, marker, values in metrics:
        ax.plot(
            cells,
            100 * np.array(values) / count,
            marker=marker,
            color=color,
            label=label,
            markersize=5,
        )
    ax.axhline(
        100 * MIN_AGREEMENT, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3))
    )
    ax.text(
        cells[0],
        100 * MIN_AGREEMENT,
        f" {100 * MIN_AGREEMENT:.0f}% threshold",
        color=INK_SECONDARY,
        va="bottom",
    )
    selected = trials[study.selected]
    selected_y = 100 * selected.prediction_matches / count
    ax.plot(
        selected.cells,
        selected_y,
        "o",
        markersize=14,
        markerfacecolor="none",
        markeredgecolor=INK,
    )
    ax.annotate(
        f"selected {selected.label}",
        (selected.cells, selected_y),
        xytext=(0, -26),
        textcoords="offset points",
        ha="center",
        color=INK,
    )
    coarser = study.next_coarser
    if coarser is not None:
        coarser_y = 100 * coarser.prediction_matches / count
        ax.annotate(
            f"{coarser.label} fails:\n{', '.join(coarser.failed_criteria())}",
            (coarser.cells, coarser_y),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
        )
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(cells, [trial.label for trial in trials], rotation=45, ha="right")
    ax.set(
        xlabel="Table size (velocity points x angle points, log scale)",
        ylabel="Rollouts in agreement (%)",
        title=f"Grid resolution on {count} off-grid starting velocities",
    )
    ax.legend(loc="lower right")
    return figure
