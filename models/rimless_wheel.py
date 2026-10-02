import numpy as np


def dynamics(t, state, params):
    g = params["gravity"]
    l = params["length"]

    theta = state[0]
    theta_dot = state[1]

    theta_ddot = (g/l) * np.sin(theta)

    state_derivative = np.array([theta_dot, theta_ddot])

    return state_derivative


def generate_params(*, n_spokes=10, slope=0.2):
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "n_spokes": n_spokes,  # number of spokes on rimless wheel
        "slope": slope,    # slope inclination angle in radians
        "half_angle": np.pi / n_spokes # half angle between spokes
    }
    return params


def generate_initial_condition():
    # Generate an initial condition suitable for this model
    # Inputs: None
    # Output: length 2 numpy vector
    return np.array([0,1])


def impact_guard(state, params):

    theta, theta_dot = state
    gamma = params["slope"]
    alpha = params["half_angle"]

    if theta >= gamma + alpha and theta_dot > 0:
        return 1
    if theta <= gamma - alpha and theta_dot < 0:
        return -1
    return 0

def reset_dynamics(state, params, impact_direction):
    theta_minus, theta_dot_minus = state

    alpha = params["half_angle"]
    theta_plus = theta_minus - impact_direction * 2 * alpha
    theta_dot_plus = theta_dot_minus * np.cos(2 * alpha)

    return np.array([theta_plus, theta_dot_plus])
