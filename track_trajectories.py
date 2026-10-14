import solver_MPC as solver_mpc
import solver_ltv_LQR as solver_lqr
import animation
import cost as cst
import dynamics as dyn
import numpy as np
import matplotlib.pyplot as plt

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10, 15)
plt.rcParams.update({'font.size': 10})


# TRAJECTORY PARAMETERS #
# tf = 6          # Final time in seconds ---- for LQR simulation
tf = 12        # Final time in seconds ---- for MPC simulation
dt = dyn.dt        # Get discretization step from dynamics
ns = dyn.ns        # Number of states
ni = dyn.ni        # Number of inputs
TT = int(tf/dt)    # Discrete-time samples


def track_trajectory_lqr(xx_star, uu_star, xx_0, plot_enabled, animation_enabled):

    print("-*-*-*-*-*-")
    print("Task 3 - Trajectory tracking via LQR")

    xx = np.zeros((ns, TT))
    uu = np.zeros((ni, TT))

    AA = np.zeros((ns, ns, TT))
    BB = np.zeros((ns, ni, TT))
    QQ = np.zeros((ns, ns, TT))
    RR = np.zeros((ni, ni, TT))
    SS = np.zeros((ni, ns, TT))
    QQf = np.zeros((ns, ns))

    xx0 = xx_0.squeeze()

    xx[:, 0] = xx0

    # 1. Linearize system around reference trajectory (Backward Pass Prep)
    for tt in reversed(range(TT-1)):

        QQ = cst.QQt_track
        RR = cst.RRt_track
        QQf = cst.QQt_track

        # Without Affine term
        qq = None
        rr = None
        qqf = None

        dfx, dfu = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[1:]

        # Jacobian Matrix
        At = dfx.T
        Bt = dfu.T

        AA[:, :, tt] = At
        BB[:, :, tt] = Bt

    # 2. Solve LTV-LQR
    KK = solver_lqr.ltv_LQR(AA, BB, QQ, RR, SS, QQf, TT, xx0, qq, rr, qqf)[0]

    # 3. Apply Control (Forward Simulation)
    for tt in range(TT-1):

        uu[:, tt] = uu_star[:, tt] + \
            KK[:, :, tt] @ (xx[:, tt] - xx_star[:, tt])
        xx[:, tt+1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0]

    # 4. Plot Tracked Trajectory
    if plot_enabled:

        if animation_enabled:
            animation.animate_traj_tracking(xx, xx_star, uu, uu_star, tf)
        else:

            tt_hor = np.linspace(0, tf, TT)
            fig, axs = plt.subplots(3, 1, sharex='all')

            axs[0].plot(tt_hor, xx_star[0, :], 'g--',
                        label=r'$\theta_1$ Reference', linewidth=2)
            axs[0].plot(tt_hor, xx[0, :], 'r',
                        label=r'$\theta_1$ Tracked', linewidth=2)
            axs[0].grid()
            axs[0].set_title(r"Tracking Reference Trajectory $\theta_1$")
            axs[0].set_ylabel(r'$\theta_1$')
            axs[0].legend(loc='upper left', fontsize=10)

            axs[1].plot(tt_hor, xx_star[1, :], 'g--',
                        label=r'$\theta_2$ Reference', linewidth=2)
            axs[1].plot(tt_hor, xx[1, :], 'r',
                        label=r'$\theta_2$ Tracked', linewidth=2)
            axs[1].grid()
            axs[1].set_title(r"Tracking Reference Trajectory $\theta_2$")
            axs[1].set_ylabel(r'$\theta_2$')
            axs[1].legend(loc='upper left', fontsize=10)

            axs[2].plot(tt_hor, uu_star[0, :], 'c--',
                        label=r'$U$ Reference', linewidth=2)
            axs[2].plot(tt_hor, uu[0, :], 'orange',
                        label=r'$U$ Tracked', linewidth=2)
            axs[2].grid()
            axs[2].set_title(r"Tracking Reference Trajectory $U$")
            axs[2].set_ylabel('$U$')
            axs[2].set_xlabel('time')
            axs[2].legend(loc='upper left', fontsize=10)

            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()

            plt.show()

    return xx, uu


def track_trajectory_mpc(xx_star, uu_star, xx_0, plot_enabled, animation_enabled):

    print('-*-*-*-*-*-')
    print("Task 4 - Trajectory tracking via MPC")

    ###############################
    # MPC Parameters Initialization
    ###############################

    T_pred = 10                 # MPC Prediction horizon
    Tsim = TT - T_pred          # Simulation horizon
    # Tsim = TT                   # Simulation horizon
    umax = 30                   # Upper input constraint
    umin = -umax                # Lower input constraint

    AAnom = np.zeros((ns, ns, TT))
    BBnom = np.zeros((ns, ni, TT))

    for tt in range(TT-1):

        dfx, dfu = dyn.dynamics(xx_star[:, tt], uu_star[:, tt])[1:]

        # Jacobian Matrix
        At = dfx.T
        Bt = dfu.T

        AAnom[:, :, tt] = At
        BBnom[:, :, tt] = Bt

    # state cost
    QQ = cst.QQt_track_mpc
    QQf = cst.QQt_track_mpc
    # input cost
    RR = cst.RRt_track_mpc

    xx_real_mpc = np.zeros((ns, Tsim))
    uu_real_mpc = np.zeros((ni, Tsim))

    xx_mpc = np.zeros((ns, T_pred, Tsim))

    xx_real_mpc[:, 0] = xx_0.squeeze()

    for tt in range(Tsim-1):
        # System evolution - real with MPC
        xx_t_mpc = xx_real_mpc[:, tt]  # get initial condition

        # Trajectory values in the prediction horizon
        xx_star_T = xx_star[:, tt:tt+T_pred]
        uu_star_T = uu_star[:, tt:tt+T_pred]

        # Linearization over the prediction horizon
        AA = AAnom[:, :, tt:tt+T_pred]
        BB = BBnom[:, :, tt:tt+T_pred]

        # Solve MPC problem - apply first input

        if tt % 10 == 0:  # print every 10 time instants
            print('MPC:\t t = {}'.format(tt))

        uu_real_mpc[:, tt], xx_mpc[:, :, tt] = solver_mpc.linear_mpc(
            AA, BB, QQ, RR, QQf, xx_t_mpc, xx_star_T, uu_star_T, umax=umax, umin=umin, T_pred=T_pred)[:2]

        xx_real_mpc[:, tt +
                    1] = dyn.dynamics(xx_real_mpc[:, tt], uu_real_mpc[:, tt])[0]

    # Plot Tracked Trajectory

    if plot_enabled:

        if animation_enabled:
            animation.animate_traj_tracking(
                xx_real_mpc, xx_star, uu_real_mpc, uu_star, tf)
        else:

            tt_hor = np.arange(Tsim)
            fig, axs = plt.subplots(3, 1, sharex='all')

            axs[0].plot(tt_hor, xx_star[0, :Tsim], 'g--',
                        label=r'$\theta_1$ Reference', linewidth=2)
            axs[0].plot(tt_hor, xx_real_mpc[0, :], 'r',
                        label=r'$\theta_1$ Tracked', linewidth=2)
            axs[0].grid()
            axs[0].set_title(r"Tracking Reference Trajectory $\theta_1$")
            axs[0].set_ylabel(r'$\theta_1$')
            axs[0].legend(loc='upper left', fontsize=10)

            axs[1].plot(tt_hor, xx_star[1, :Tsim], 'g--',
                        label=r'$\theta_2$ Reference', linewidth=2)
            axs[1].plot(tt_hor, xx_real_mpc[1, :], 'r',
                        label=r'$\theta_2$ Tracked', linewidth=2)
            axs[1].grid()
            axs[1].set_title(r"Tracking Reference Trajectory $\theta_2$")
            axs[1].set_ylabel(r'$\theta_2$')
            axs[1].legend(loc='upper left', fontsize=10)

            axs[2].plot(tt_hor, uu_star[0, :Tsim], 'c--',
                        label=r'$U$ Reference', linewidth=2)
            axs[2].plot(tt_hor, uu_real_mpc[0, :], 'orange',
                        label=r'$U$ Tracked', linewidth=2)
            axs[2].plot(tt_hor, np.ones(Tsim)*umax, 'r',
                        label=r'$U$ Max', linewidth=2)
            axs[2].plot(tt_hor, np.ones(Tsim)*umin, 'r',
                        label=r'$U$ Min', linewidth=2)
            axs[2].grid()
            axs[2].set_title(r"Tracking Reference Trajectory $U$")
            axs[2].set_ylabel('$U$')
            axs[2].set_xlabel('time')
            axs[2].legend(loc='upper left', fontsize=10)

            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()

            plt.show()

    return xx_real_mpc, uu_real_mpc
