import timeit

import matplotlib.pyplot as plt
import numpy as np

from integrators import explicit_euler as integrator
from models import rimless_wheel as model

params = model.generate_params()
params["slope_angle"] = 0.2

# State is stance angle and stance angular velocity
initial_state = np.array([-0.3, 0.5])

timestep = 1e-3
sim_time = 5.0

time_traj, state_traj = integrator.integrate(params, model, initial_state, timestep, sim_time)
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

# Plot the angle over time
plt.figure()
plt.plot(time_traj, state_traj[0], 'bo:', label="Angle")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.title("Rimless Wheel Angle vs Time")
plt.legend()
plt.tight_layout()
plt.show()

# Phase diagram: velocity vs angle
plt.figure()
plt.plot(state_traj[0], state_traj[1], 'bo:', alpha=0.1, label="Trajectory")
plt.xlabel("Angle (radians)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Rimless Wheel Phase Portrait")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the system energy over time
plt.figure()
plt.plot(time_traj, potential_energy, 'bo:', label="Potential energy")
plt.plot(time_traj, kinetic_energy, 'ro:', label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, 'go:', label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()



# Searching for Regions of Attraction
# I want to make this as automated as possible

perform_roa = False
if perform_roa:
    # List to keep attractors we find
    # If a phase trajectory re-visits a point (or gets within a tiny epsilon value),
    # then we can conclude that this forms a limit cycle / attractor
    eps = 5e-2
    attractors = []

    # Search space
    initial_angles = np.linspace(-params["half_spoke_angle"], params["half_spoke_angle"], 5)
    initial_velocities = np.linspace(0, 5, 5)
    print(f"Initial angles: {initial_angles}")
    print(f"Initial velocities: {initial_velocities}")

    # Regions of attraction
    space_size = (initial_angles.size, initial_velocities.size)
    regions = -1 * np.ones(space_size)

    # Search over all initial states
    for pt_idx in range(space_size[0] * space_size[1]):
        print(f"{100 * pt_idx / (space_size[0] * space_size[1]):.1f}% done")
        # Get initial conditions
        angle_idx = pt_idx % space_size[0]
        vel_idx = pt_idx // space_size[1]
        initial_state = np.array([initial_angles[angle_idx], initial_velocities[vel_idx]])

        # Simulate system
        time_traj, state_traj = integrator.integrate(
            params, model, initial_state, timestep, sim_time
        )

        # Verify attractor
        this_attractor = None
        diff = np.linalg.norm(state_traj[:,[-1]] - state_traj, axis=0)
        repeated_points = (diff[:-int(0.1 / timestep)] < eps).nonzero()[0]  # Exclude the last 0.1 seconds
        if repeated_points.size > 0:
            # There is at least one repeated point
            idx = np.max(repeated_points)
            # print(f"This state leads to an attractor of cycle duration {time_traj[-1] - time_traj[idx]:.3f} seconds")
            this_attractor = state_traj[:,idx:]

        if this_attractor is not None:
            # We need to see whether we have already found this attractor
            # print("Looking over our already-found attractors...")
            already_found = False
            for att_idx in range(len(attractors)):
                # Check an attractor that we had already found against every point in the attractor from this state point
                no_match = False
                for this_att_idx in range(this_attractor.shape[1]):
                    all_comparisons = np.linalg.norm(attractors[att_idx] - this_attractor[:,[this_att_idx]], axis=0)
                    if np.min(all_comparisons) > eps:
                        # This point doesn't have a close match in any of this attractor's points
                        # We can move on to the next attractor
                        no_match = True
                        break
                if not no_match:
                    # Every point on this_attractor has a match in an attractor that we had already found!
                    # print("Found a match")
                    already_found = True
                    regions[angle_idx, vel_idx] = att_idx
                    break
            if not already_found:
                # print("Found no matches")
                attractors.append(this_attractor)
                regions[angle_idx, vel_idx] = len(attractors) - 1
        else:
            # print("No attractor")
            regions[angle_idx, vel_idx] = -1

    print("Done!")
    print(regions)
    plt.imshow(regions)
    plt.show()

    for att_idx in range(len(attractors)):
        print(f"Showing attractor {att_idx}")
        plt.plot(attractors[att_idx][0,:], attractors[att_idx][1,:], 'bo:')
        plt.show()


# Poincare Section
initial_state = np.array([0.0, 0.5])
poincare_section = []
time_traj, state_traj = integrator.integrate(params, model, initial_state, timestep, sim_time, dynamics_args={"poincare_section":poincare_section})

# Plot crossings
plt.figure()
max_vel = max([p[1] for p in poincare_section])
plt.plot([0,1.1*max_vel],[0,1.1*max_vel], label="Identity Line")
for i in range(len(poincare_section)-1):
    plt.plot(poincare_section[i][1], poincare_section[i+1][1], 'rx')
    plt.gca().text(poincare_section[i][1], poincare_section[i+1][1], f"{i}", fontsize=12)
plt.title("Return Map for Angular Velocity and Identity Line")
plt.xlabel("Previous value of Angular Velocity (rad/s)")
plt.ylabel("Next value of Angular Velocity (rad/s)")
plt.legend()
plt.show()
print(f"Calculated fixed point: {sum([poincare_section[-i][1] for i in range(5)])/5}")
# Fixed point appears to be about (2.460474, 2.460474)


# plt.figure()
# plt.plot(range(len(poincare_section)), [p[1] for p in poincare_section])
# plt.show()
