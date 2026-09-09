import numpy as np


def dynamics(t, state, params, poincare_section=None):
    # Dynamical model of the rimless wheel
    # The contact of each individual spoke can be modeled like an inverted pendulum
    # Whenever a new spoke contacts the ground, we instantaneously switch to that spoke's perspective
    # Angular momentum about the new contact point must be conserved
    # Positive angle is clockwise in this model; angle of 0 is globally vertical
    # If poincare_section is provided (as an empty list), then the state after each collision will be appended to it
    mass = params["mass"]
    gravity = params["gravity"]
    length = params["spoke_length"]
    slope_angle = params["slope_angle"]
    half_spoke_angle = np.pi / float(params["num_spokes"])  # half-angle between spokes, radians

    abs_angle = state[0]
    angular_velocity = state[1]

    # Check for next-spoke contact
    # Angle relative to slope normal
    rel_angle = abs_angle - slope_angle
    if rel_angle > half_spoke_angle:
        # Transition to either the next or the previous spoke
        # The actual transition is handled by our modulus functions, but we need to absorb some energy for the collision
        # Projecting the previous velocity onto the arc of the new spoke to preserve the tangent component:
        angular_velocity *= np.cos(2 * half_spoke_angle)
        # This is poor form, but we can actually just modify the state in-place
        state[0] = -half_spoke_angle + slope_angle
        state[1] = angular_velocity
        if poincare_section is not None:
            poincare_section.append(state)
    if rel_angle < -half_spoke_angle:
        # Assume that the wheel has stopped
        state[0] = -half_spoke_angle
        state[1] = 0
        return np.array([0,0])
    # The dynamics of this wheel assume that it will always be rolling downhill
    # There are obviously initial conditions that could violate this, but we will assume that those won't be applied

    # Continuous inverse-pendulum dynamics
    angular_acceleration = (
           mass * gravity * length * np.sin(abs_angle)
   ) / (mass * length ** 2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravitational acceleration, m/s^2
        "spoke_length": 0.5,  # spoke length, m
        "num_spokes": 8,  # number of spokes around the entire wheel
        "slope_angle": 0.2,  # slope angle, radians
        "mass": 1,  # mass of center of wheel, kg
    }
    # half spoke angle depends on num_spokes, so it will need to be calculated
    return params


def calculate_energy(state, params):
    # Calculate the energy for a certain state of this system
    # Energy for this system is kind of awkward because it's always rotating down a slope
    # So, in one sense, we start with infinite gravitational potential energy and
    # We're also losing energy with every collision because they're modeled as being plastic

    angle = state[0]
    angular_velocity = state[1]

    # We will assume that gravitational potential energy is calculated from the point of contact with the slope
    kinetic_energy = 0.5 * params["mass"] * np.pow(angular_velocity * params["spoke_length"], 2)
    potential_energy = params["gravity"] * params["mass"] * np.cos(angle) * params["spoke_length"]
    return kinetic_energy, potential_energy