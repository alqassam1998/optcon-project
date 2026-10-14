#
# Newton method for Optimal Control
# Main
#

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import Flexible Robotic Arm dynamics
import dynamics as dyn

# import cost functions
import cost as cst
from cost import QQt, RRt
# import armijo stepsize selector
import armijo
# import reference trajectory generator
import reference_trajectories as ref_gen
import control as ctrl
import solver_ltv_LQR as solver
import animation
import track_trajectories as track

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({'font.size': 22})

#######################################
# Algorithm parameters
#######################################

max_iters = 10

fixed_stepsize = 1e-4

# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_maxiters = 20  # number of Armijo iterations

term_cond = 1e-6

visu_descent_plot = False
visu_animation = True

#######################################
# Trajectory parameters
#######################################

# tf = 6 # final time in seconds ---- for LQR simulation
tf = 12  # final time in seconds ---- for MPC simulation

dt = dyn.dt   # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt)  # discrete-time samples

######################################
# Reference curve
######################################

step_reference = False

xx_ref, uu_ref = ref_gen.gen(
    step_reference=step_reference, tf=tf, dt=dt, ns=ns, ni=ni)


######################################
# Initial guess
######################################

x0 = xx_ref[:, 0]
xx_init = np.repeat(xx_ref[:, 0].reshape(-1, 1), TT, axis=1)
uu_init = np.repeat(uu_ref[:, 0].reshape(-1, 1), TT, axis=1)

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

deltau = np.zeros((ni, TT, max_iters))  # Du - descent direction
dJ = np.zeros((ni, TT, max_iters))     # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters)  # collect descent direction
descent_arm = np.zeros(max_iters)  # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

xx[:, :, 0] = xx_init
uu[:, :, 0] = uu_init

# Create a figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'hspace': 1})
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'hspace': 1})


for kk in range(max_iters-1):

    JJ[kk] = 0
    # calculate cost
    for tt in range(TT-1):
        temp_cost = cst.stagecost(
            xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
        JJ[kk] += temp_cost

    temp_cost = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[0]
    JJ[kk] += temp_cost

    ##################################
    # Descent direction calculation
    ##################################

    AA = np.zeros((ns, ns, TT))
    BB = np.zeros((ns, ni, TT))
    QQ = np.zeros((ns, ns, TT))
    RR = np.zeros((ni, ni, TT))
    SS = np.zeros((ni, ns, TT))
    qq = np.zeros((ns, TT))
    rr = np.zeros((ni, TT))
    QQf = np.zeros((ns, ns))
    qqf = np.zeros((ns))

    lmbd_temp, QQf = cst.termcost(xx[:, TT-1, kk], xx_ref[:, TT-1])[1:]

    qqf = lmbd_temp.squeeze()

    lmbd[:, TT-1, kk] = lmbd_temp.copy().squeeze()

    for tt in reversed(range(TT-1)):  # integration backward in time

        qt, rt, QQt, RRt = cst.stagecost(
            xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:]
        dfx, dfu = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:]

        At = dfx.T
        Bt = dfu.T

        # AFFINE PROBLEM
        QQ[:, :, tt] = QQt
        RR[:, :, tt] = RRt
        qq[:, tt] = qt.squeeze()
        rr[:, tt] = rt.squeeze()

        AA[:, :, tt] = At
        BB[:, :, tt] = Bt

        lmbd_temp = At.T@lmbd[:, tt+1, kk][:, None] + \
            qt       # costate equation
        dJ_temp = Bt.T@lmbd[:, tt+1, kk][:, None] + \
            rt         # gradient of J wrt u

        lmbd[:, tt, kk] = lmbd_temp.squeeze()
        dJ[:, tt, kk] = dJ_temp.squeeze()

    KK, sigma, PP, deltax, deltau_temp = solver.ltv_LQR(
        AA, BB, QQ, RR, SS, QQf, TT, x0, qq, rr, qqf)

    deltau[:, :, kk] = deltau_temp

    for tt in reversed(range(TT-1)):
        descent[kk] += deltau[:, tt, kk].T@deltau[:,
                                                  tt, kk]     # DESCENT DIRECTION
        descent_arm[kk] += dJ[:, tt, kk].T@deltau[:,
                                                  tt, kk]     # For armijio plot

    ##################################
    # Stepsize selection - ARMIJO
    ##################################

    if Armijo:

        stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                          deltau[:, :, kk], xx_ref, uu_ref, x0,
                                          uu[:, :, kk], JJ[kk], xx[:, :, kk], KK, sigma, descent_arm[kk], visu_descent_plot)
    else:
        stepsize = fixed_stepsize

    ############################
    # Update the current solution
    ############################

    xx_temp = np.zeros((ns, TT))
    uu_temp = np.zeros((ni, TT))

    xx_temp[:, 0] = x0

    for tt in range(TT-1):
        uu_temp[:, tt] = uu[:, tt, kk] + KK[:, :,
                                            tt]@(xx_temp[:, tt] - xx[:, tt, kk]) + stepsize*sigma[:, tt]
        xx_temp[:, tt+1] = dyn.dynamics(xx_temp[:, tt], uu_temp[:, tt])[0]

    uu_temp[:, -1] = uu_temp[:, -2]

    xx[:, :, kk+1] = xx_temp.copy()
    uu[:, :, kk+1] = uu_temp.copy()

   #########################################################
   # PLOT at each iteration#
   ########################################################

    # Figure One

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    # Plot x[0] and x_ref[0] on the first subplot
    axs[0].plot(np.linspace(0, tf, TT), xx[0, :, kk+1],
                "b-", label=r"$\theta_1$")
    axs[0].plot(np.linspace(0, tf, TT), xx_ref[0, :],
                "b--", label=r"$\theta_1$ Reference")
    axs[0].grid()
    axs[0].set_title(r"Generating Optimal Trajectory $\theta_1$")
    axs[0].set_ylabel(r'$\theta_1$')
    axs[0].legend(loc='upper left', fontsize=10)

    # Plot x[2] and x_ref[2] on the second subplot
    axs[1].plot(np.linspace(0, tf, TT), xx[2, :, kk+1],
                "c-", label=r"$\dot{\theta}_1$")
    axs[1].plot(np.linspace(0, tf, TT), xx_ref[2, :],
                "c--", label=r"$\dot{\theta}_1$ Reference")
    axs[1].grid()
    axs[1].set_title(r"Generating Optimal Trajectory $\dot{\theta}_1$")
    axs[1].set_ylabel(r'$\dot{\theta}_1$')
    axs[1].legend(loc='upper left', fontsize=10)

    # Plot u and u_ref on the third subplot
    axs[2].plot(np.linspace(0, tf, TT), uu[0, :, kk+1], "r-", label="U")
    axs[2].plot(np.linspace(0, tf, TT), uu_ref[0, :],
                "r--", label="$U$ Reference")
    axs[2].grid()
    axs[2].set_title(r"Generating Optimal Trajectory $U$")
    axs[2].set_ylabel('$U$')
    axs[2].legend(loc='upper left', fontsize=10)
    axs[2].set_xlabel('Time')

    # ===================================================================================================== #

    # Figure two

    axs2[0].cla()
    axs2[1].cla()
    axs2[2].cla()

    # Plot x[1] and x_ref[1] on the first subplot
    axs2[0].plot(np.linspace(0, tf, TT), xx[1, :, kk+1],
                 "b-", label=r"$\theta_2$")
    axs2[0].plot(np.linspace(0, tf, TT), xx_ref[1, :],
                 "b--", label=r"$\theta_2$ Reference")
    axs2[0].grid()
    axs2[0].set_title(r"Generating Optimal Trajectory $\theta_2$")
    axs2[0].set_ylabel(r'$\theta_2$')
    axs2[0].legend(loc='upper left', fontsize=10)

    # Plot x[3] and x_ref[3] on the second subplot
    axs2[1].plot(np.linspace(0, tf, TT), xx[3, :, kk+1],
                 "c-", label=r"$\dot{\theta}_2$")
    axs2[1].plot(np.linspace(0, tf, TT), xx_ref[3, :],
                 "c--", label=r"$\dot{\theta}_2$ Reference")
    axs2[1].grid()
    axs2[1].set_title(r"Generating Optimal Trajectory $\dot{\theta}_2$")
    axs2[1].set_ylabel(r'$\dot{\theta}_2$')
    axs2[1].legend(loc='upper left', fontsize=10)

    # Plot u and u_ref on the third subplot
    axs2[2].plot(np.linspace(0, tf, TT), uu[0, :, kk+1], "r-", label="u")
    axs2[2].plot(np.linspace(0, tf, TT), uu_ref[0, :],
                 "r--", label="$u_{ref}$")
    axs2[2].grid()
    axs2[2].set_title(r"Generating Optimal Trajectory $U$")
    axs2[2].set_ylabel('$U$')
    axs2[2].legend(loc='upper left', fontsize=10)
    axs2[2].set_xlabel('Time')

    plt.pause(1e-4)

    ############################
    # Termination condition
    ############################

    print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(
        kk, descent[kk], JJ[kk]))

    if descent[kk] <= term_cond:

        max_iters = kk

        print('***************************************')
        print('The descent is less than the threshold!')
        print('***************************************')

        break

xx_star = xx[:, :, max_iters-1]
uu_star = uu[:, :, max_iters-1]
uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes


# xx0 = np.array([0.0, 0.0, 0.0, 0.0])  # initial state for tracking

# Introduce a 2-degree error in theta1 and theta2 to test the controller
xx0 = xx_star[:, 0] + np.array([np.deg2rad(2), np.deg2rad(2), 0, 0])

# Tracking the optimal trajectory using LQR
xx_track, uu_track = track.track_trajectory_lqr(
    xx_star, uu_star, xx0, plot_enabled=True, animation_enabled=True)

# Tracking the optimal trajectory using MPC
# xx_track, uu_track = track.track_trajectory_mpc(
#    xx_star, uu_star, xx0, plot_enabled=True, animation_enabled=False)

############################
# Plots
############################

# ---------------------------------------------------------
# PLOT TRACKING ERROR (Required for Report)
# ---------------------------------------------------------

# Calculate the error: e(t) = x_track(t) - x_ref(t)
limit = min(xx_track.shape[1], xx_star.shape[1])
error_traj = xx_track[:, :limit] - xx_star[:, :limit]
time_axis = np.linspace(0, (limit-1)*dyn.dt, limit)

plt.figure('Tracking Error Evolution', figsize=(10, 8))

# Subplots for each state
state_names = [r'$\theta_1$', r'$\theta_2$',
               r'$\dot{\theta}_1$', r'$\dot{\theta}_2$']

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time_axis, error_traj[i, :], linewidth=2, color='red')
    plt.ylabel(f'Error {state_names[i]} [rad]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    # Add a zero line to show convergence
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.suptitle(
    f'Tracking Error Evolution (Perturbation: {np.degrees(xx0 - xx_star[:, 0]).round(1)} deg)')
plt.tight_layout()
plt.show()


# cost and descent

plt.figure('Descent direction')
plt.plot(np.arange(max_iters-1), descent[:max_iters-1])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show()


plt.figure('Cost')
plt.plot(np.arange(max_iters-1), JJ[:max_iters-1])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show()


if visu_animation:
    print('-*-*-*-*-*-')
    print("Task 5 - Animation of The Robot Tracking Optimal Trajectory")

    # animation.animate_flexible_robot_arm(xx=xx_star, xx_ref=xx_ref, dt=dt) # Animation of The Robot Tracking Reference Trajectory
    # Animation of The Robot Tracking Optimal Trajectory
    animation.animate_flexible_robot_arm(xx=xx_track, xx_ref=xx_star, dt=dt)
