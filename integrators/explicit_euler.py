import numpy as np


def integrate(params, model, initial_state, timestep, sim_time, dynamics_args={}):
    # Integrate a system using explicit Forward Euler integration
    # Subject to errors at large timesteps
    # Inputs:
    #   params - dict of model params, e.g. gravity, lengths, etc.
    #   model - system model, which has a dynamics function
    #   initial_state - numpy vector representing the initial state (nx1)
    #   timestep - size of timesteps, in seconds
    #   sim_time - total time to simulate over, in seconds
    #   dynamics_args - keyword args to pass to model.dynamics, as a dict of key-value pairs
    # Output: Time trajectory (1xm) and state trajectory vectors (nxm)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
            t, state_traj[:, step], params, **dynamics_args
        )

    return time_traj, state_traj

