import numpy as np

def integrate(params, model, initial_state, timestep, sim_time):
    # Integrate a system using 4th Order Runge-Kutta methods
    # Subject to errors at large timesteps
    # Inputs:
    #   params - dict of model params, e.g. gravity, lengths, etc.
    #   model - system model, which has a dynamics function
    #   initial_state - numpy vector representing the initial state (nx1)
    #   timestep - size of timesteps, in seconds
    #   sim_time - total time to simulate over, in seconds
    # Output: Time trajectory (1xm) and state trajectory vectors (nxm)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        k1 = model.dynamics(t, state_traj[:, step], params)
        k2 = model.dynamics(t + timestep/2, state_traj[:, step] + k1 * timestep/2, params)
        k3 = model.dynamics(t + timestep/2, state_traj[:,step] + k2 * timestep/2, params)
        k4 = model.dynamics(t + timestep, state_traj[:, step] + k3 * timestep, params)
        state_traj[:, step + 1] = state_traj[:, step] + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return time_traj, state_traj


def integrate_step(params, model, initial_state, t, timestep):
    # Integrate a system using 4th Order Runge-Kutta methods
    # Subject to errors at large timesteps
    # Inputs:
    #   params - dict of model params, e.g. gravity, lengths, etc.
    #   model - system model, which has a dynamics function
    #   initial_state - numpy vector representing the initial state (nx1)
    #   t: current time
    #   timestep - size of timesteps, in seconds
    # Output: Next state vector (2x1 array)

    # simulate step
    k1 = model.dynamics(t, initial_state, params)
    k2 = model.dynamics(t + timestep/2, initial_state + k1 * timestep/2, params)
    k3 = model.dynamics(t + timestep/2, initial_state + k2 * timestep/2, params)
    k4 = model.dynamics(t + timestep, initial_state + k3 * timestep, params)
    return initial_state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


