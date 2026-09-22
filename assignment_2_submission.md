# Assignment 2 Submission

To run my assignment 2 code, run the following command in your terminal:
```
uv run assignment_2.py
```
This will generate and save a number of plots, including:
* A gif of the walker walking at `output-assignment2/assignment_2/walker.gif`
* A map showing the region of attraction of the ankle torque controller at `output-assignment/controller_data/stop_map.png`
* Data used by the ankle torque controller at `output-assignment2/assignment_2/bounds_9-81_1-00_1-00_0-06.npy` and `output-assignment2/controller_data/map_9-81_1-00_1-00_0-06.npy`
* A plot of angular velocities at $\theta = 0$, chosen control angles, and the resulting angular velocity next time the Poincare section is reached, at `output-assignment2/assignment_2/table.png`
* A plot of the relationships between states in the Poincare section at `output-assignment2/assignment_2/poincare.png`
* A plot of the same steps to reach stability for a number of velocity values at `output-assignment2/assignment_2/steps.png`

## Sketches

## Ankle Controller Region of Attraction
![Stop Map for Ankle Controller](output-assignment2/controller_data/stop_map.png "Stop Map")

## Poincare Section
The line at $\theta=0$ was chosen as the Poincare section. Orbits of the natural pendulum are transverse through this line and the section isn't dependent on any of the controller inputs.

## Grid Resolution


## Trajectory Plot


## Reaching a Standstill
![Plot of the number of steps required to reach a standstill](output-assignment2/assignment_2/steps.png "Steps to reach a standstill")

