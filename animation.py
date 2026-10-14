# Animation

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


def animate_flexible_robot_arm(xx, xx_ref, dt):
    """
    Animates the Flexible Robotic Arm dynamics
    input parameters:
          - Optimal state trajectory xx
          - Reference trajectory xx_ref
          - Sampling time dt
    output arguments:
          None
    """

    # --- PARAMETERS FOR SET 2  ---
    l1 = 1.5
    l2 = 1.5
    # ---------------------------------------

    TT = xx.shape[1]

    # Set up the figure and axis for the animation
    # Total length is now 3.0 (1.5 + 1.5), so limits increased to 4 for clearance
    fig, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Plot elements
    arm_1_line, = ax.plot([], [], 'o-', lw=3, color="blue",
                          label="Optimal Path for Arm 1")
    reference_1_line, = ax.plot(
        [], [], 'o--', lw=2, color="green", label="Reference Path for Arm 2")

    arm_2_line, = ax.plot([], [], 'o-', lw=3,
                          color="blue", label="Optimal Path")
    reference_2_line, = ax.plot(
        [], [], 'o--', lw=2, color="green", label="Reference Path for Arm 2")

    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.legend(fontsize=10)
    ax.set_title(f"Flexible Robotic Arm (Set 2: l1={l1}, l2={l2})")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    # Initial setup function for the animation
    def init():
        arm_1_line.set_data([], [])
        reference_1_line.set_data([], [])

        arm_2_line.set_data([], [])
        reference_2_line.set_data([], [])

        time_text.set_text('')
        return arm_1_line, reference_1_line, arm_2_line, reference_2_line, time_text

    # Update function for each frame of the animation
    def update(frame):
        # --- Kinematics updated for lengths l1 and l2 ---

        # Arm 1 position (optimal solution)
        # x1 = l1 * sin(theta1)
        x_1_opt = l1 * np.sin(xx[0, frame])
        y_1_opt = -l1 * np.cos(xx[0, frame])

        # Reference 1 position
        x_1_ref = l1 * np.sin(xx_ref[0, frame])
        y_1_ref = -l1 * np.cos(xx_ref[0, frame])

        # Arm 2 position (optimal solution) relative to Arm 1 end
        # The second link angle is theta1 + theta2 (relative angle theta2)
        x_2_opt = l2 * np.sin(xx[0, frame] + xx[1, frame])
        y_2_opt = -l2 * np.cos(xx[0, frame] + xx[1, frame])

        # Reference 2 position relative to Arm 1 end
        x_2_ref = l2 * np.sin(xx_ref[0, frame] + xx_ref[1, frame])
        y_2_ref = -l2 * np.cos(xx_ref[0, frame] + xx_ref[1, frame])

        # Update Arm 1 line (Origin to End of Link 1)
        arm_1_line.set_data([0, x_1_opt], [0, y_1_opt])
        reference_1_line.set_data([0, x_1_ref], [0, y_1_ref])

        # Update Arm 2 line (End of Link 1 to End of Link 2)
        # We add the relative vector (x_2_opt) to the end of link 1 (x_1_opt)
        arm_2_line.set_data([x_1_opt, x_1_opt + x_2_opt],
                            [y_1_opt, y_1_opt + y_2_opt])
        reference_2_line.set_data(
            [x_1_ref, x_1_ref + x_2_ref], [y_1_ref, y_1_ref + y_2_ref])

        # Update time text
        time_text.set_text(f'time = {frame*dt:.2f}s')

        return arm_1_line, reference_1_line, arm_2_line, reference_2_line, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=TT,
                        init_func=init, blit=True, interval=0)

    # Display the animation
    plt.show()


def animate_traj_tracking(xx, xx_ref, uu, uu_ref, tf):
    """
    Animates the Flexible Robotic Arm dynamics
    input parameters:
          - track state trajectory xx
          - Reference state trajectory xx_ref
          - track input trajectory uu
          - Reference input trajectory uu_ref
          - final time tf
    output arguments:
          None
    """

    TT = xx_ref.shape[1]

    tt_hor = np.linspace(0, tf, TT)  # Time horizon
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={
                            'hspace': 0.5}, sharex='all')

    theta_1_reference_path, = axs[0].plot(
        tt_hor, xx_ref[0, :], 'g--', label=r'$\theta_1$ Reference', linewidth=2)
    theta_1_track_path, = axs[0].plot(
        [], [], 'r', label=r'$\theta_1$ Tracked', linewidth=2)
    theta_1_track_marker, = axs[0].plot([], [], 'o', markersize=5)
    axs[0].grid()
    axs[0].set_title(r"Tracking Reference Trajectory $\theta_1$")
    axs[0].set_ylabel(r'$\theta_1$')
    # axs[0].set_ylim(min(xx[0,:]) - 0.5, max(xx[0,:]) + 0.5)
    axs[0].legend(loc='upper left', fontsize=10)

    theta_2_reference_path, = axs[1].plot(
        tt_hor, xx_ref[1, :], 'g--', label=r'$\theta_2$ Reference', linewidth=2)
    theta_2_track_path, = axs[1].plot(
        [], [], 'r', label=r'$\theta_2$ Tracked', linewidth=2)
    theta_2_track_marker, = axs[1].plot([], [], 'o', markersize=5)
    axs[1].grid()
    axs[1].set_title(r"Tracking Reference Trajectory $\theta_2$")
    axs[1].set_ylabel(r'$\theta_2$')
    # axs[1].set_ylim(min(xx[1,:]) - 0.5, max(xx[1,:]) + 0.5)
    axs[1].legend(loc='upper left', fontsize=10)

    input_reference_path, = axs[2].plot(
        tt_hor, uu_ref[0, :], 'c--', label=r'$U$ Reference', linewidth=2)
    input_track_path, = axs[2].plot(
        [], [], 'orange', label=r'$U$ Tracked', linewidth=2)
    input_track_marker, = axs[2].plot([], [], 'o', markersize=5)
    axs[2].grid()
    axs[2].set_title(r"Tracking Reference Trajectory $U$")
    axs[2].set_ylabel('$U$')
    # axs[0].set_ylim(min(uu[0,:]) - 0.5, max(uu[0,:]) + 0.5)
    axs[2].set_xlabel('time')
    axs[2].legend(loc='upper left', fontsize=10)

    # plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()

    # time_text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes)

    # Initial setup function for the animation

    def init():

        theta_1_track_path.set_data([], [])
        theta_1_track_marker.set_data([], [])

        theta_2_track_path.set_data([], [])
        theta_2_track_marker.set_data([], [])

        input_track_path.set_data([], [])
        input_track_marker.set_data([], [])

        # time_text.set_text('')

        return theta_1_track_path, theta_2_track_path, theta_1_track_marker, theta_2_track_marker, input_track_path, input_track_marker  # , time_text

    # Update function for each frame of the animation
    def update(frame):

        # Update theta 1 tracking path
        theta_1_track_path.set_data([tt_hor[:frame]], [xx[0, :frame]])
        # Update theta 1 marker
        theta_1_track_marker.set_data([tt_hor[frame]], [xx[0, frame]])

        # Update theta 2 tracking path
        theta_2_track_path.set_data([tt_hor[:frame]], [xx[1, :frame]])
        # Update theta 2 marker
        theta_2_track_marker.set_data([tt_hor[frame]], [xx[1, frame]])

        # Update input tracking path
        input_track_path.set_data([tt_hor[:frame]], [uu[0, :frame]])
        # Update input marker
        input_track_marker.set_data([tt_hor[frame]], [uu[0, frame]])

        # # Update time text
        # time_text.set_text(f'time = {frame*dt:.2f}s')

        return theta_1_track_path, theta_2_track_path, theta_1_track_marker, theta_2_track_marker, input_track_path, input_track_marker  # , time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=TT,
                        init_func=init, blit=True, interval=0)

    # Display the animation
    plt.show()
