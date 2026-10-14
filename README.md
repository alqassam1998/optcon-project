# OPTIMAL CONTROL OF A FLEXIBLE ROBOTIC LINK

## HOW TO RUN
Execute the main script: python "main_newton_optcon_method.py"

Configuration options are inside "main_newton_optcon_method.py":

**1. SELECTING GENERATION TASK (Task 1, Task 2)**
   - Look for the variable "step_reference" (near top of file).
   - Set "step_reference" = True  ----> Generates Step Reference (Task 1).
   - Set "step_reference" = False ----> Generates Smooth Reference (Task 2).

**2. SELECTING TRACKING CONTROLLER (Task 3, Task 4)**
   - Scroll to the bottom of the main file (approx lines 322-329).
   - Uncomment ONE of the following blocks to select the controller:
     
     For LQR (Task 3)
     xx_track, uu_track = track.track_trajectory_lqr(..., animation_enabled=True)

     For MPC (Task 4)
     xx_track, uu_track = track.track_trajectory_mpc(..., animation_enabled=True)

**3. INITIAL PERTURBATION (Testing Tracking Performance)**
   - To verify the controller works, we introduce an error at t=0.
   - Locate the "xx0" definition around line 320:
     xx0 = xx_star[:, 0] + np.array([np.deg2rad(2), np.deg2rad(2), 0, 0])
   - This adds a 2 degree initial offset.

**4. ANIMATION (Task 5)**
   - Set "visu_animation = True" at the top of the script to enable general animation.
   - Ensure "animation_enabled=True" is passed to the tracking functions.

## RESULTS & PLOTS
After running the script, the following plots are generated sequentially:
1. Descent Direction: Verifies convergence of Newton's method.
2. Cost Evolution: Shows minimization of the cost function J.
3. Optimal Trajectories: Comparison of Reference vs. Optimized path.
4. Input Torque: The optimized control effort u(t).
5. Tracking Error Evolution: plot showing the error e(t) converging to zero over time.

Kind Regards from group (33).

