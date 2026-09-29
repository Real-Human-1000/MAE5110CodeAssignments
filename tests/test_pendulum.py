import numpy as np

from models import pendulum
from integrators import rk4


def test_energy_conservation():
    params = pendulum.generate_params()
    params["damping_coeff"]=0.0
    params["torque"]=0.0

    initial_state = np.array([0.2,0.0])

    kinetic_energy, potential_energy = pendulum.calculate_energy(initial_state, params)
    initial_total_energy = kinetic_energy + potential_energy

    state = initial_state
    timestep, time = 0.01, 0.0
    for i in range (100):
        state = rk4.integrate_step(pendulum.dynamics, time, state, timestep, params)
        time += timestep

    final_kinetic_energy, final_potential_energy = pendulum.calculate_energy(state, params)
    final_total_energy = final_kinetic_energy + final_potential_energy
    assert np.isclose(initial_total_energy,final_total_energy)

def test_torque():
    params = pendulum.generate_params()
    params["damping_coeff"]=0.0
    state = np.array([0.0,0.0])

    params["torque"] = 0.0
    _,zero_torque_result = pendulum.dynamics(0,state,params)

    params["torque"] = 3.0
    _,positive_torque_result = pendulum.dynamics(0,state,params)

    params["torque"] = 6.0
    _,larger_torque_result = pendulum.dynamics(0,state,params)

    params["torque"] = -3.0
    _,negative_torque_result = pendulum.dynamics(0,state,params)

    assert np.isclose(zero_torque_result,0.0)
    assert positive_torque_result > zero_torque_result
    assert negative_torque_result < zero_torque_result
    assert larger_torque_result > positive_torque_result


def test_damping():
    params = pendulum.generate_params()
    params["torque"]=0.0

    params["damping_coeff"] = 0.0
    state = np.array([0.0,5.0])
    _,zero_damping_result = pendulum.dynamics(0,state,params)

    params["damping_coeff"] = 3.0
    state = np.array([0.0,5.0])
    _,positive_velocity_result = pendulum.dynamics(0,state,params)

    params["damping_coeff"] = 10.0
    state = np.array([0.0,5.0])
    _,larger_damping_result = pendulum.dynamics(0,state,params)

    params["damping_coeff"] = 3.0
    state = np.array([0.0,-5.0])
    _,negative_velocity_result = pendulum.dynamics(0,state,params)

    assert np.isclose(zero_damping_result,0.0)
    assert positive_velocity_result < zero_damping_result
    assert negative_velocity_result > zero_damping_result
    assert larger_damping_result < positive_velocity_result