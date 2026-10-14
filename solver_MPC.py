import numpy as np
import cvxpy as cp


def linear_mpc(AA, BB, QQ, RR, QQf, xx_t, xx_star, uu_star, umax, umin, T_pred=10):
    """
    Linear MPC solver - Constrained LQR

    Given a measured state xxt measured at t
    gives back the optimal input to be applied at t

    Parameters:
    - AA: ndarray, system dynamics matrix for state evolution.
    - BB: ndarray, control input matrix.
    - QQ: ndarray, state weighting matrix.
    - RR: ndarray, control input weighting matrix.
    - QQf: ndarray, final state weighting matrix.
    - xx_t: ndarray, the initial state of the system.
    - xx_star: ndarray, the desired state trajectory over the prediction horizon.
    - uu_star: ndarray, the desired control input trajectory over the prediction horizon.
    - umax: float, maximum control input
    - umin: float, minimum control input
    - T_pred: int, prediction horizon

    Returns:
    - uu_mpc_opt: ndarray, the first optimal control input.
    - xx_mpc_traj: ndarray, the state trajectory over the prediction horizon.
    - uu_mpc_traj: ndarray, the control input trajectory over the prediction horizon.
    """

    # Squeeze initial state to ensure it is a 1D array #
    xx_t = xx_t.squeeze()

    ns, ni = BB.shape[:2]

    # Declare optimization variables for the state and control input trajectories #
    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    # Initialize cost and constraints #
    cost = 0
    constraints = []

    # Set up the cost function and constraints for each time step #
    for tt in range(T_pred-1):
        # Add the cost for deviations from the optimal trajectory
        # cost += cp.quad_form(xx_mpc[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt], RR)
        cost += cp.quad_form(xx_mpc[:, tt] - xx_star[:, tt], QQ) + \
            cp.quad_form(uu_mpc[:, tt] - uu_star[:, tt], RR)

        # Add system dynamics constraint for state evolution #
        constraints += [
            xx_mpc[:, tt+1] == AA[:, :, tt]@xx_mpc[:, tt] +
            BB[:, :, tt]@uu_mpc[:, tt],  # dynamics constraint
            # xx_mpc[:,tt+1] == AA[:,:,tt]@(xx_mpc[:,tt] - xx_star[:,tt]) + BB[:,:,tt]@(uu_mpc[:,tt] - uu_star[:,tt]),  # dynamics constraint
            uu_mpc[:, tt] <= umax,  # Control input upper bound
            uu_mpc[:, tt] >= umin,  # Control input lower bound
        ]

    # Add the final state cost at the last prediction step #
    # cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)
    cost += cp.quad_form(xx_mpc[:, T_pred-1] - xx_star[:, T_pred-1], QQf)

    # Initial condition constraint (set initial state) #
    constraints += [xx_mpc[:, 0] == xx_t]

    # Define and solve the optimization problem #
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Check if the problem is infeasible #
    if problem.status == "infeasible":
        print("Infeasible problem! Please check your constraints.")

    # Return the first optimal control input, the state trajectory, and the control input trajectory #
    return uu_mpc[:, 0].value, xx_mpc.value, uu_mpc.value
