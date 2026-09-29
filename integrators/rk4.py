import numpy as np


def integrate(params, model, initial_state, timestep, sim_time, dynamics_args={}):
    # Integrate a system using 4th Order Runge-Kutta methods
    # Subject to errors at large timesteps
    # Inputs:
    #   params - dict of model params, e.g. gravity, lengths, etc.
    #   model - system model, which has a dynamics function
    #   initial_state - numpy vector representing the initial state (nx1)
    #   timestep - size of timesteps, in seconds
    #   sim_time - total time to simulate over, in seconds
    #   dynamics_args - arguments to provide to model.dynamics, in the form of a dict of key-value pairs
    # Output: Time trajectory (1xm) and state trajectory vectors (nxm)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        k1 = model.dynamics(t, state_traj[:, step], params, **dynamics_args)
        k2 = model.dynamics(t + timestep/2, state_traj[:, step] + k1 * timestep/2, params, **dynamics_args)
        k3 = model.dynamics(t + timestep/2, state_traj[:,step] + k2 * timestep/2, params, **dynamics_args)
        k4 = model.dynamics(t + timestep, state_traj[:, step] + k3 * timestep, params, **dynamics_args)
        state_traj[:, step + 1] = state_traj[:, step] + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return time_traj, state_traj

