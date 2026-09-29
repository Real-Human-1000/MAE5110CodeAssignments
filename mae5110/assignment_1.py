import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from integrators import rk4
from models import rimless_wheel as model

params = model.generate_params()


def simulate(initial_state, params, t_max, dt_base, *, settle_speed=1e-3):
    if dt_base <= 0 or settle_speed <= 0:
        raise ValueError("dt_base and settle_speed must be positive")
    t = 0.0
    state = np.array(initial_state, dtype=float)

    t_history = [t]
    state_history = [state]
    poincare_sections = []

    while t < t_max:
        dt = min(dt_base, t_max - t)
        next_state = rk4.step(model.dynamics, t, state, dt, params)
        impact_direction = model.impact_guard(next_state, params)
        if impact_direction != 0:
            low, high = 0.0, dt
            for _ in range(30):
                mid = (low + high) / 2
                test_state = rk4.step(model.dynamics, t, state, mid, params)
                if model.impact_guard(test_state, params) == impact_direction:
                    high = mid
                else:
                    low = mid
            fine_state = rk4.step(model.dynamics, t, state, high, params)
            fine_state[0] = params["slope"] + impact_direction * params["half_angle"]
            t += high
            poincare_sections.append(fine_state)
            state = model.reset_dynamics(fine_state, params, impact_direction)

            can_rest = abs(params["slope"]) < params["half_angle"]
            if can_rest and abs(state[1]) < settle_speed:
                state[1] = 0.0
                t_history.append(t)
                state_history.append(state)
                break

            t_history.append(t)
            state_history.append(state)
        else:
            state = next_state
            t += dt

            t_history.append(t)
            state_history.append(state)

    return np.array(t_history), np.array(state_history), np.array(poincare_sections).reshape(-1, 2)


def classify_motion(state_history, poincare_sections):
    if len(poincare_sections) > 0 and state_history[-1, 1] == 0:
        return -1
    recent = poincare_sections[-5:, 1]
    if len(recent) == 5 and np.all(recent > 0):
        if np.ptp(recent) <= 0.01 * np.mean(recent):
            return 1
    return 0


def estimate_roa():
    gamma = params["slope"]
    alpha = params["half_angle"]
    theta_values = np.linspace(gamma - alpha, gamma + alpha, 60)
    theta_dot_values = np.linspace(-5, 5, 60)

    roa_map = np.zeros((60, 60))

    for i, theta_dot in enumerate(theta_dot_values):
        for j, theta in enumerate(theta_values):
            initial_state = [theta, theta_dot]
            _, states, poincare_sections = simulate(initial_state, params, t_max=10, dt_base=0.01)
            roa_map[i, j] = classify_motion(states, poincare_sections)

    theta_grid, theta_dot_grid = np.meshgrid(theta_values, theta_dot_values)
    plt.figure(figsize=(8, 6))

    plt.scatter(theta_grid[roa_map == 1], theta_dot_grid[roa_map == 1],
                color='green', marker='o', label='Rolling')
    plt.scatter(theta_grid[roa_map == -1], theta_dot_grid[roa_map == -1],
                color='red', marker='o', label='Dead Stop')
    plt.scatter(theta_grid[roa_map == 0], theta_dot_grid[roa_map == 0],
                color='gray', marker='o', label='Unresolved at 10 s')

    plt.xlabel('Initial Angle, $\\theta$ (rad)')
    plt.ylabel('Initial Velocity, $\\dot{\\theta}$ (rad/s)')
    plt.title('Region of Attraction: Rimless Wheel')
    plt.legend()
    plt.tight_layout()
    plt.show()



def return_map(params):
    test_velocities = np.linspace(0.5, 4.0, 100)
    v_n = []
    v_next = []

    for velocity in test_velocities:
        initial_state = [0.0, velocity]
        _, _, poincare = simulate(initial_state, params, t_max=3.0, dt_base=0.01)
        poincare = poincare[poincare[:, 1] > 0]
        if len(poincare) >= 2:
            v_n.append(poincare[0][1])
            v_next.append(poincare[1][1])

    return np.array(v_n), np.array(v_next)


def estimate_fixed_point(v_n, v_next, window_radius=2):
    v_n, v_next = np.asarray(v_n), np.asarray(v_next)
    valid = np.isfinite(v_n) & np.isfinite(v_next)
    order = np.argsort(v_n[valid])
    v_n, v_next = v_n[valid][order], v_next[valid][order]
    residual = v_next - v_n
    fixed_velocity = None
    for fixed_idx in range(len(v_n)):
        if residual[fixed_idx] == 0:
            fixed_velocity = v_n[fixed_idx]
            break
        if (fixed_idx + 1 < len(v_n)
                and v_n[fixed_idx + 1] > v_n[fixed_idx]
                and residual[fixed_idx] * residual[fixed_idx + 1] < 0):
            fraction = -residual[fixed_idx] / (residual[fixed_idx + 1] - residual[fixed_idx])
            fixed_velocity = v_n[fixed_idx] + fraction * (v_n[fixed_idx + 1] - v_n[fixed_idx])
            break
    if fixed_velocity is None:
        return None, None

    start_idx = max(0, fixed_idx - window_radius)
    end_idx = min(len(v_n), fixed_idx + max(window_radius + 1, 2))
    local_v = v_n[start_idx:end_idx]

    if np.unique(local_v).size < 2:
        return fixed_velocity, None

    multiplier, _ = np.polyfit(local_v, v_next[start_idx:end_idx], 1)
    return fixed_velocity, multiplier

def return_map_plot(params=None):
    if params is None:
        params = model.generate_params()

    v_n, v_next = return_map(params)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(v_n, v_next, label="Return map")
    all_velocities = np.concatenate([v_n, v_next])
    limits = [all_velocities.min(), all_velocities.max()] if all_velocities.size else [0, 1]
    ax.plot(limits, limits, linestyle="--", color="gray", label="Identity ($y=x$)")

    ax.set_xlabel("Impact Velocity $v_{n}$ (rad/s)")
    ax.set_ylabel("Next Impact Velocity $v_{n+1}$ (rad/s)")
    ax.set_title(
        f"Return Map: {params['n_spokes']:g} Spokes, "
        f"Slope = {params['slope']:.4g} rad"
    )
    fixed_velocity, multiplier = estimate_fixed_point(v_n, v_next)
    if fixed_velocity is not None:
        ax.scatter(
            [fixed_velocity], [fixed_velocity], color="red", s=65, zorder=5,
            label=f"Estimated fixed point ({fixed_velocity:.3f}, {fixed_velocity:.3f})",
        )
        note = (f"Estimated Floquet multiplier: {multiplier:.3f}"
                if multiplier is not None else "Floquet multiplier unavailable")
    else:
        note = "No fixed point detected in sampled range"
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    plt.show()



def _sweep(values, *, parameter, label, title):

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.colormaps["cividis"]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 8), layout="constrained"
    )
    base_params = model.generate_params()
    if parameter == "slope":
        sweep_info = f"{len(values)} slope values swept"
    else:
        sweep_info = (
            f"{len(values)} spoke counts swept\n"
            rf"$\gamma = {base_params['slope']:.4g}$ rad"
        )
    ax1.text(
        0.02, 0.98, sweep_info, transform=ax1.transAxes,
        va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    curves = []
    fixed_velocities = []
    multipliers = []

    for value in values:
        sweep_params = model.generate_params(**{parameter: value})
        v_n, v_next = return_map(sweep_params)
        if v_n.size == 0:
            fixed_velocities.append(np.nan)
            multipliers.append(np.nan)
            continue

        curves.extend([v_n, v_next])
        ax1.plot(v_n, v_next, color=cmap(norm(value)))

        fixed_velocity, multiplier = estimate_fixed_point(v_n, v_next)
        fixed_velocities.append(fixed_velocity if fixed_velocity is not None else np.nan)
        multipliers.append(multiplier if multiplier is not None else np.nan)

    all_velocities = np.concatenate(curves) if curves else np.array([])
    limits = [all_velocities.min(), all_velocities.max()] if all_velocities.size else [0, 1]
    ax1.plot(limits, limits, linestyle="--", color="gray", label="Identity ($y=x$)")

    ax2.plot(values, multipliers, color='red',linestyle='--', linewidth=2, marker='*')
    ax2.set_xlabel(label)
    ax2.set_ylabel("Floquet Multiplier")
    ax2.set_title(f"Floquet Multiplier as a Function of {label}")
    ax2.axhline(1, color="gray", linestyle=":", label="Stability boundaries")
    ax2.axhline(-1, color="gray", linestyle=":")
    ax2.legend()
    ax2.grid(True)
    ax3.plot(values, fixed_velocities, color='blue',linestyle='--', linewidth=2, marker='*')
    ax3.set_xlabel(label)
    ax3.set_ylabel("Fixed Velocity (rad/s)")
    ax3.set_title(f"Fixed Velocity as a Function of {label}")
    ax3.grid(True)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax1, label=label)
    cbar.set_ticks([values.min(), values.max()])
    cbar.set_ticklabels([f"{values.min():.4g} (Min)", f"{values.max():.4g} (Max)"])
    ax1.set_xlabel("Impact Velocity $v_{n}$ (rad/s)")
    ax1.set_ylabel("Next Impact Velocity $v_{n+1}$ (rad/s)")
    ax1.set_title(title)
    ax1.grid(True)
    plt.show()


def spokes_sweep():
    _sweep(
        np.arange(6, 17),
        parameter="n_spokes",
        label="Spokes",
        title="Return Map: Spoke Sweep",
    )


def inclination_sweep():
    _sweep(
        np.linspace(0.05, 0.5, 20),
        parameter="slope",
        label="Slope (rad)",
        title="Return Map: Inclination Sweep",
    )


#testing/sanity checking code below

def test_energy_conservation():

    params = model.generate_params()

    initial_state = np.array([-0.1, 0.5])

    timestep = 1e-4
    sim_time = 10.0
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = rk4.step(model.dynamics, t, state_traj[:, step], timestep, params)

    total_energy = 1/2 * params["mass"] * params["length"]**2 * state_traj[1,:]**2 + params["mass"] * params["gravity"] * params["length"] * np.cos(state_traj[0,:])
    plt.figure()
    plt.plot(time_traj, total_energy, label="energy")

    plt.xlabel("Time (s)")
    plt.ylabel("energy")
    plt.title("Rimless Wheel energy test")
    plt.legend()
    plt.tight_layout()
    plt.ylim(9.88, 9.89)
    plt.show()

def test_flat_ground_dissipation():

    params = model.generate_params()

    params["slope"] = 0.0
    initial_state = np.array([0.0, 3.0])
    timestep = 1e-4
    sim_time = 6.0
    time_traj, states, _ = simulate(initial_state, params, sim_time, timestep)
    state_traj = states.T

    kinetic = 0.5 * params["mass"] * params["length"]**2 * state_traj[1,:]**2
    potential = params["mass"] * params["gravity"] * params["length"] * np.cos(state_traj[0,:])
    total_energy = kinetic + potential

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(time_traj, total_energy, color='blue')
    ax1.set_ylabel("Total Energy (J)")
    ax1.set_title("Flat Ground Dissipation Check")
    ax2.plot(time_traj, state_traj[1,:], color='red')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    plt.tight_layout()
    plt.show()

def test_downhill_limit_cycle():

    params = model.generate_params()

    initial_state = np.array([params["slope"] - params["half_angle"], 3.0])
    timestep = 1e-4
    sim_time = 6.0
    time_traj, states, _ = simulate(initial_state, params, sim_time, timestep)
    state_traj = states.T

    _, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    ax1.plot(state_traj[0,:], state_traj[1,:] , color='blue')
    ax1.set_ylabel("Velocity")
    ax1.set_xlabel("Position")
    ax1.set_title("phase portrait")
    plt.tight_layout()
    plt.show()
