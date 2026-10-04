import numpy as np

def dynamics(t, state, params):
    # Model a bouncing ball with a certain radius and coefficient of restitution
    # The ball cannot travel below y=0, the floor, and will bounce when it touches it
    # state needs to have four rows: x, xdot, y, ydot
    # This model assumes that there is no friction between the ground and ball (so it can't spin)

    # We have the issue that we don't know the timestep and can only set the derivatives of our state for one step
    # So we can't instantaneously change the position or velocity, since they only update by a tiny bit

    gravity = params["gravity"]
    radius = params["radius"]
    mass = params["mass"]
    elasticity = params["elasticity"]
    damping = params["damping"]

    x = state[0]  # horizontal position
    xdot = state[1]  # horizontal velocity
    y = state[2]  # vertical position
    ydot = state[3]  # vertical velocity

    ydotdot = -gravity  # vertical acceleration

    if y - radius < 0:
        # The edge of the ball collides with the ground! Bounce it!
        ydotdot += -((y - radius) * elasticity) / mass  # Simulated "springy" ground (or ball)
        ydotdot += -ydot * damping  # Simulated damping losses from heat/friction in the ball
        # If we had direct control of position, we could also pop position of the ball above the ground

    state_deriv = np.array([xdot, 0, ydot, ydotdot])
    return state_deriv


def generate_params():
    # The params for the bouncing ball
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "radius": 0.1,  # ball radius (m)
        "mass": 0.5,  # mass of the ball (kg)
        "elasticity": 1000,  # spring constant for when the ball bounces
        "damping": 0.5  # damping constant for when the ball bounces
    }
    return params

def generate_initial_condition():
    # The initial condition for the bouncing ball
    initial_state=np.array([0.0,0.0,1.0,0.0])
    return initial_state


def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    radius = params["radius"]
    elasticity = params["elasticity"]

    xdot = state[1]  # horizontal velocity
    y = state[2]  # vertical position
    ydot = state[3]  # vertical velocity

    kinetic_energy = 0.5 * mass * ydot**2 + 0.5 * mass * xdot**2
    potential_energy = y * mass * gravity
    potential_energy += np.where(y - radius < 0, (0.5 * elasticity * (y - radius)**2), np.zeros_like(y))  # select only when the ball is contacting the ground

    return kinetic_energy, potential_energy

