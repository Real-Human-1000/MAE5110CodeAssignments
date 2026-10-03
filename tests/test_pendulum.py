import numpy as np

import models.pendulum as model
from integrators import rk4 as integrator


def test_no_torque_no_damping():
    # Test that total system energy remains constant when the pendulum has no torque or damping

    params = model.generate_params()
    params["damping_coeff"] = 1

    initial_state = model.generate_initial_condition()
    # Nonzero initial state
    initial_state[0] = 1
    initial_state[1] = 1

    timestep = 1e-1
    sim_time = 10

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(model.dynamics, t, state_traj[:,step], timestep, params)

    initial_kinetic, initial_potential = model.calculate_energy(initial_state, params)
    initial_total_energy = initial_kinetic + initial_potential

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    assert np.all(np.isclose(total_energy, initial_total_energy))


def test_torque():
    # Test the condition where torque perfectly balances gravitational force, with no damping
    # This should result in the pendulum staying still for all eternity

    params = model.generate_params()
    params["damping_coeff"] = 0
    params["torque"] = -params["gravity"] * params["mass"] / params["length"]

    initial_state = model.generate_initial_condition()
    # Nonzero initial state
    initial_state[0] = np.pi/2
    initial_state[1] = 0

    timestep = 1e-1
    sim_time = 10

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(
            model.dynamics, t, state_traj[:, step], timestep, params
        )

    initial_kinetic, initial_potential = model.calculate_energy(initial_state, params)
    initial_total_energy = initial_kinetic + initial_potential

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    assert np.all(np.isclose(total_energy, initial_total_energy))


def test_damping():
    # Test that damping will bring the pendulum to a standstill

    params = model.generate_params()
    params["damping_coeff"] = 1.0

    initial_state = model.generate_initial_condition()
    # Nonzero initial state
    initial_state[0] = np.pi / 2
    initial_state[1] = 0

    timestep = 1e-1
    sim_time = 20  # this only really works asymptotically, so we need to simulate for longer

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.shape[0], n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(
            model.dynamics, t, state_traj[:, step], timestep, params
        )

    # Strictly speaking, I don't know how the potential field is defined, so I will rely on the model for energy calc
    static_kinetic, static_potential = model.calculate_energy(np.array([np.pi,0]), params)
    static_total_energy = static_kinetic + static_potential

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    # This time, we just want to look at the last state
    assert np.isclose(total_energy[-1], static_total_energy)
