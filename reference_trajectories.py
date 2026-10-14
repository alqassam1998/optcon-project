#
# Newton Method for Optimal Control
# Reference Trajectory Generation
#

import numpy as np
from equilibria import equilibrium


def gen(step_reference, tf, dt, ns, ni):
    """
    Main entry point for trajectory generation.
    """
    if step_reference:
        return gen_step_trajectory(tf, dt, ns, ni)
    else:
        return gen_smooth_trajectory(tf, dt, ns, ni)


def gen_step_trajectory(tf, dt, ns, ni):
    """
    Task 1: Generates a sudden step change.
    0 degrees -> 60 degrees instantly at T/2.
    """
    print('-*-*-*-*-*-')
    print("Task 1 - Trajectory Generation (Step)")

    TT = int(tf/dt)
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # Apply the step change at the halfway mark
    ref_deg = 60
    xx_ref[0, int(TT/2):] = np.deg2rad(ref_deg)

    # Compute valid physics for every point (equilibrium)
    for t in range(TT):
        theta1 = float(xx_ref[0, t])
        xx_e, uu_e = equilibrium(theta1)
        xx_ref[:, t] = xx_e
        uu_ref[:, t] = uu_e

    return xx_ref, uu_ref


def gen_smooth_trajectory(tf, dt, ns, ni):
    """
    Task 2: Generates a smooth curve geometrically.
    Path: 0 -> 60 -> 30 -> 0 degrees.
    Uses 'Smoothstep' interpolation (3x^2 - 2x^3) to ignore dynamics.
    """
    print('-*-*-*-*-*-')
    print("Task 2 - Trajectory Generation (Smooth Kinematic)")

    TT = int(tf/dt)
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    # 1. Define the path we want to follow (Waypoints)
    targets_deg = [0, 60, 30, 0]

    # We have 3 moves (0->60, 60->30, 30->0). Split total time by 3.
    steps_per_move = TT // 3

    current_t = 0

    # 2. Loop through each move
    for i in range(3):
        start_angle = targets_deg[i]
        end_angle = targets_deg[i+1]

        # Get the valid physical state for the Start and End points
        start_state, start_input = equilibrium(np.deg2rad(start_angle))
        end_state, end_input = equilibrium(np.deg2rad(end_angle))

        # Interpolate between Start and End for this segment
        for step in range(steps_per_move):
            if current_t >= TT:
                break

            # 'x' is our progress from 0.0 to 1.0
            x = step / steps_per_move

            # Simple Smoothing Formula (Cubic Smoothstep)
            # This creates the S-curve shape
            s = 3*x**2 - 2*x**3

            # Calculate the intermediate state by blending Start and End
            xx_ref[:, current_t] = start_state + (end_state - start_state) * s
            uu_ref[:, current_t] = start_input + (end_input - start_input) * s

            # Calculate Velocity (needed for tracker)
            if current_t > 0:
                xx_ref[2, current_t] = (
                    xx_ref[0, current_t] - xx_ref[0, current_t-1]) / dt
                xx_ref[3, current_t] = (
                    xx_ref[1, current_t] - xx_ref[1, current_t-1]) / dt

            current_t += 1

    # Fill any remaining time with the final state
    final_state, final_input = equilibrium(np.deg2rad(targets_deg[-1]))
    while current_t < TT:
        xx_ref[:, current_t] = final_state
        uu_ref[:, current_t] = final_input
        current_t += 1

    return xx_ref, uu_ref
