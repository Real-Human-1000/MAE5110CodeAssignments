import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import pendulum as model
from models import bouncing_ball as model_bb
from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    "torque": 0.0,  # torque (Nm)
}

# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-3
sim_time = 5.0

time_traj, state_traj = integrator.integrate(params, model, initial_state, timestep, sim_time)

# timeit trials for the two different integrators
# print(timeit.timeit(lambda: integrator.integrate(params, model, initial_state, timestep, sim_time), number=10))
# 10 trials each
# Same timestep 1e-4
# rk4 1e-4: 5.914196699999593
# ee 1e-4: 1.1558105000003707
# Largest timestep with < 1% error
# rk4 17e-2: 0.003637400001025526
# ee 2e-4: 0.5815985000008368

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

# Record the difference in energy observed (initial vs max diff), which we can correlate to numerical stability
# print(potential_energy[0] + kinetic_energy[0])
# print(potential_energy[-1] + kinetic_energy[-1])
# print(np.max(np.abs(potential_energy[0] + kinetic_energy[0] - potential_energy - kinetic_energy)))

# Plot the system energy over time
plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# Forward-Euler timestep stability data from previous experiments
plt.figure()
stability_data = np.array([
    [1e-5, 0.0005905664312792491],  # This data was collected manually
    [1e-4, 0.005900466086736],
    [1e-1, 5.072434753010766],
    [5e-2, 2.3230285667451853],
    [25e-3, 1.1463059377481968],
    [125e-4, 0.6712444854571763],
    [1e-3, 0.05851325093934279],
    [1e-2, 0.5452212444311582],
    [5e-3, 0.2831168543531102],
    [1e0, 130.42786394972003],
    [5e-1, 11.087846645768431],
    [2e-3, 0.11602547623693815],
    [9e-4, 0.05270637042693427],
    [5e-4, 0.02939008259857223],
    [3e-4, 0.017666905443722902],
    [2e-4, 0.011789502410201103]
])
sort_idx = np.argsort(stability_data[:,0], axis=0)
stability_data = stability_data[sort_idx,:]
plt.loglog(stability_data[:,0], np.abs(stability_data[:,1]))
plt.xlabel("Timestep size (log)")
plt.ylabel("Maximum energy error")
plt.title("Energy Error vs Timestep Size for Forward Euler Pendulum Simulation")
plt.show()
# To keep energy within 1% of initial total energy, timestep must be smaller than 2e-4

# Runge-Kutta timestep stability data from previous experiments
plt.figure()
stability_data = np.array([
    [1e-5, 3.7125857943465235e-13],  # This data was collected manually
    [1e-4, 2.7977620220553945e-14],
    [1e-3, 1.255884285455977e-12],
    [1e-2, 1.0973600250707705e-08],
    [5e-2, 3.98738862614767e-05],
    [75e-3, 0.0002715263111844335],
    [1e-1, 0.0010764617586209169],
    [5e-1, 11.087846645768431],
    [1e0, 1.7652982276830052],
    [2e-1, 0.027450702331213273],
    [15e-2, 0.007343784236782502],
    [17e-2, 0.013252958709302765]
])
sort_idx = np.argsort(stability_data[:,0], axis=0)
stability_data = stability_data[sort_idx,:]
plt.loglog(stability_data[:,0], np.abs(stability_data[:,1]))
plt.xlabel("Timestep size (log)")
plt.ylabel("Maximum energy error")
plt.title("Energy Error vs Timestep Size for Runge-Kutta Pendulum Simulation")
plt.show()
# To keep energy within 1% of initial total energy, timestep must be smaller than 17e-2


# Bouncing ball model
params = model_bb.generate_params()

# Initial conditions for the model
initial_state=model_bb.generate_initial_condition()

# We're keeping the timestep and sim length from the pendulum

time_traj, state_traj = integrator.integrate(params, model_bb, initial_state, timestep, sim_time)

# Sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.
kinetic_energy, potential_energy = model_bb.calculate_energy(state_traj, params)

# Plot the system energy over time
plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Ball energy")
plt.legend()
plt.tight_layout()
plt.show()

# Plot y position over kinetic energy
plt.figure()
plt.plot(time_traj, state_traj[2,:], label="Vertical pos")
plt.ylabel("Vertical position")
plt.xlabel("Time (s)")
plt.title("Ball position over Time")
plt.legend()
plt.tight_layout()
plt.show()

