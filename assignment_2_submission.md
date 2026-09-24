# Assignment 2 Submission

To run my assignment 2 code, run the following command in your terminal:
```
uv run assignment_2.py
```
This will generate and save a number of plots, including:
* A gif of the walker walking at `output-assignment2/assignment_2/walker.gif`
* A map showing the region of attraction of the ankle torque controller at `output-assignment/controller_data/stop_map.png`
* Data used by the ankle torque controller at `output-assignment2/assignment_2/bounds_9-81_1-00_1-00_0-06.npy` and `output-assignment2/controller_data/map_9-81_1-00_1-00_0-06.npy`
* A plot of angular velocities at $\theta = 0$, chosen control angles, and the resulting angular velocity next time the Poincare section is reached, at `output-assignment2/assignment_2/next_vel_table.png`
* A plot of the relationships between states in the Poincare section at `output-assignment2/assignment_2/poincare.png`
* A plot of the same steps to reach stability for a number of velocity values at `output-assignment2/assignment_2/least_steps.png`
* A plot of the most steps to reach stability at any given state for a number of velocity values at `output-assignment2/assignment_2/most_steps.png`
* A phase plot of the walker walking from an extreme initial state at `output-assignment2/assignment_2/walking_phase.png`
* A phase plot of the walker walking from the same initial state as above, but trying to take as many steps as possible `output-assignemnt2/assignment_2/walking_most_phase.png`

## Sketches

![Sketch of some walker states](images/sketch.jpg "Sketch")

## Ankle Controller Region of Attraction
In the following plot, yellow represents regions where the ankle torque controller can bring the pendulum to equilibrium, and purple is otherwise.

![Stop Map for Ankle Controller](output-assignment2/controller_data/stop_map.png "Stop Map")

## Poincare Section
The line at $\theta=0$ was chosen as the Poincare section. Orbits of the natural pendulum are transverse through this line and the section isn't dependent on any of the controller inputs. Since the angle of attack controller is based around this section, the angle will have to be updated when the walker passes through the section every step.

## Grid Resolution
A grid with "good-enough" resolution would be able to accurately model an underlying continuous process. A grid with "not-good-enough" resolution will introduce some artifacts. We can expect that the accuracy of the model will increase as the grid resolution increases. With this in mind, we can examine how the number of steps that the walker takes from a given initial condition changes as the grid resolution is increased. After some point, the number of steps will converge, and we will know that the grid has achieved "good-enough" resolution.
We will be looking at the number of discrete velocity and action samples for a simulation starting at $\theta = 0$ and $\dot{\theta} = \sqrt{2 \frac{g}{\ell}}$, since it's relevant for the next part of the assignment

| | N vel = 5 | N vel = 10 | N vel = 25 | N vel = 50 |
|---|---|---|---|---|
|N action = 2 | 5 | 4 | 3 | 3 |
| N action = 5 | 5 | 4 | 4 | 3 |
| N action = 10 | 5 | 4 | 4 | 3 |
| N action = 25 | 5 | 4 | 4 | 3 |
| N action = 50 | 5 | 4 | 4 | 3 |

The control system seems to be much more sensitive to changes in the number of velocity states than the number of control actions.
As a reasonable happy medium, I will use a grid with 50 velocities and 10 control angles.

These dimensions produce the following lookup table:

![Next-velocity lookup table](output-assignment2/assignment_2/next_vel_table.png "Next-velocity grid")

## Trajectory Plot
![Phase plot of the walker walking](output-assignment2/assignment_2/walking_phase.png "Trajectory of 3 steps")

![Time plot of the walker walking](output-assignment2/assignment_2/walking_time.png "Walking plotted against time")

With the same initial condition as above, the walker could take as many as five steps:

![Phase plot of the walker walking for 5 steps](output-assignment2/assignment_2/walking_most_phase.png "Trajectory of 5 steps")

![Time plot of the walker walking for 5 steps](output-assignment2/assignment_2/walking_most_time.png "Walking for 5 steps plotted against time")

## Reaching a Standstill
My implementation of control over the angle of attack reveals that the walker has some options when it comes to the number of steps required to reach a standstill. With careful choices, the walker could come to a stop in only 3 steps, or it could take as many as 5.

![Plot of the least number of steps required to reach a standstill](output-assignment2/assignment_2/least_steps.png "Least steps to reach a standstill")

![Plot of the most number of steps required to reach a standstill](output-assignment2/assignment_2/most_steps.png "Most steps to reach a standstill")

Here is a more fun, but less useful, way of looking at all possible velocity states and their connections:

![Connections of states around the Poincare section](output-assignment2/assignment_2/poincare.png "Connections of states aroudn the Poincare section")

