#
# Newton method for Optimal Control
# Cost functions
#

import numpy as np
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# QQt = 0.1*np.diag([100.0, 100.0, 1.0, 1.0]) # This gives the same weight of tracking the angle 2 (desired = 0),
# which is not stable (even its not an equilibria) and cannot be reched because there is no input for the second arm.
# This leads to no more decrease in the cost and the tracking will be stopped early with large error

# QQt = 0.1*np.diag([100.0, 0, 1.0, 0]) # This gives the weight of tracking the first angle without concerning about the second one which is free to rotate.

# This is the perfect value for Q for trajectory generation
QQt = 0.1*np.diag([1000, 10, 10, 10])
# This is the perfect value for R for trajectory generation
RRt = 0.01*np.eye(ni)

QQt_track = 0.1*np.diag([100.0, 100, 0.01, 10])  # Used for trajectory tracking
RRt_track = 0.001*np.eye(ni)  # Used for trajectory tracking

# Used for trajectory tracking
QQt_track_mpc = 0.1*np.diag([100.0, 100, 0.01, 10])
RRt_track_mpc = 0.001*np.eye(ni)  # Used for trajectory tracking


def stagecost(xx, uu, xx_ref, uu_ref):
    """
      Stage-cost 

      Quadratic cost function 
      l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

      Args
        - xx \in \R^4 state at time t
        - xx_ref \in \R^4 state reference at time t

        - uu \in \R^1 input at time t
        - uu_ref \in \R^1 input reference at time t


      Return 
        - cost at xx,uu
        - gradient of l wrt x, at xx,uu
        - gradient of l wrt u, at xx,uu
        - hessian of l wrt x, at xx,uu
        - hessian of l wrt u, at xx,uu

    """

    xx = xx[:, None]
    uu = uu[:, None]

    xx_ref = xx_ref[:, None]
    uu_ref = uu_ref[:, None]

    ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + \
        0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

    dlx = QQt@(xx - xx_ref)
    dlu = RRt@(uu - uu_ref)

    dlxx = QQt
    dluu = RRt

    return ll.squeeze(), dlx, dlu, dlxx, dluu


def termcost(xT, xT_ref, QQT=QQt):
    """
      Terminal-cost

      Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

      Args
        - xT \in \R^4 state at time t
        - xT_ref \in \R^4 state reference at time t

      Return 
        - cost at xT,uu
        - gradient of l wrt x, at xT,uu
        - hessian of l wrt x, at xT,uu

    """

    llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref)

    dlTx = QQT@(xT - xT_ref)

    dlTxx = QQT

    return llT.squeeze(), dlTx, dlTxx
