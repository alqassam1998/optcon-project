# dynamics.py
# CasADi implementation
# - Continuous-time dynamics xdot = f(x,u)
# - Discretization: RK4
# - Discrete-time Jacobians computed by AD on the discretized map
# - LEGACY return format for your existing codebase:
#     returns dfx = (Ad)^T  (4x4)
#     returns dfu = (Bd)^T  (1x4)
#
#     At = dfx.T ; Bt = dfu.T

import numpy as np
import casadi as ca

# -----------------------------
# Dimensions / discretization
# -----------------------------
ns = 4
ni = 1
dt = 1e-3

# -----------------------------
# Parameters: Set 2
# -----------------------------
m1 = 2.0
m2 = 2.0
l1 = 1.5
l2 = 1.5
lc1 = 0.75
lc2 = 0.75
I1 = 1.5
I2 = 1.5
g = 9.81
f1 = 1.0
f2 = 1.0
k = 0.6

# ------------------------------------------------------------
# Build CasADi functions once (discrete map + discrete Jacobians)
# ------------------------------------------------------------
_x = ca.SX.sym("x", ns)          # [th1, th2, w1, w2]
_u = ca.SX.sym("u", ni)          # [tau]

th1, th2, w1, w2 = _x[0], _x[1], _x[2], _x[3]
tau = _u[0]

c2 = ca.cos(th2)
s2 = ca.sin(th2)

# Mass matrix M(th2)
M11 = I1 + I2 + (lc1**2)*m1 + m2*(l1**2 + 2*l1*lc2*c2 + (lc2**2))
M12 = I2 + lc2*m2*(l1*c2 + lc2)
M22 = I2 + (lc2**2)*m2
M = ca.vertcat(
    ca.horzcat(M11, M12),
    ca.horzcat(M12, M22)
)

# Coriolis matrix C(th2,w1,w2)
C11 = -l1*lc2*m2*w2*s2
C12 = -l1*lc2*m2*(w1 + w2)*s2
C21 = l1*lc2*m2*w1*s2
C22 = 0.0
C = ca.vertcat(
    ca.horzcat(C11, C12),
    ca.horzcat(C21, C22)
)

# Friction matrix
F = ca.vertcat(
    ca.horzcat(f1, 0.0),
    ca.horzcat(0.0, f2)
)

# Gravity + stiffness vector G(th1,th2)
G1 = g*lc1*m1*ca.sin(th1) + g*m2*(l1*ca.sin(th1) + lc2*ca.sin(th1 + th2))
G2 = g*m2*lc2*ca.sin(th1 + th2) + 3.0*k*ca.sin(th2)*(ca.cos(th2)**2)
G = ca.vertcat(G1, G2)

qdot = ca.vertcat(w1, w2)
rhs = ca.vertcat(tau, 0.0) - (C @ qdot) - (F @ qdot) - G
qddot = ca.solve(M, rhs)  # [dd1, dd2]

xdot = ca.vertcat(w1, w2, qddot[0], qddot[1])

f_ct = ca.Function("f_ct", [_x, _u], [xdot])


def _rk4_step(x, u):
    k1 = f_ct(x, u)
    k2 = f_ct(x + 0.5*dt*k1, u)
    k3 = f_ct(x + 0.5*dt*k2, u)
    k4 = f_ct(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


x_next = _rk4_step(_x, _u)

Ad = ca.jacobian(x_next, _x)     # (4x4)
Bd = ca.jacobian(x_next, _u)     # (4x1)

fd_fun = ca.Function("fd",  [_x, _u], [x_next])
Ad_fun = ca.Function("Ad",  [_x, _u], [Ad])
Bd_fun = ca.Function("Bd",  [_x, _u], [Bd])


def dynamics(xx, uu):
    """
    Discrete-time RK4 dynamics with discrete-time Jacobians (LEGACY output).

    Returns:
      x_next : (4,)
      dfx    : (4,4) = (Ad)^T
      dfu    : (1,4) = (Bd)^T
    """
    x = np.array(xx, dtype=float).reshape((ns,))
    u = np.array(uu, dtype=float).reshape((ni,))

    xk1 = np.array(fd_fun(x, u)).astype(float).reshape((ns,))
    Adk = np.array(Ad_fun(x, u)).astype(float).reshape((ns, ns))
    Bdk = np.array(Bd_fun(x, u)).astype(float).reshape((ns, ni))

    dfx = Adk.T              # legacy
    dfu = Bdk.T              # legacy (1x4)

    return xk1, dfx, dfu
