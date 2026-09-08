#!/usr/bin/env python3
"""Sage no-op robot controller (zero actuation).

Mirrors the MuJoCo harness, which never applies actuator force: the loop
only keeps the robot alive. OmniSim freezes the joints of a Robot whose
controller directory is missing (bodies stay pinned at their initial
pose); a real, forever-running controller keeps the linkage simulated.
"""

from omnisim import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())
while robot.step(timestep) != -1:
    pass