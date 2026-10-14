# equilibria.py
import numpy as np
from scipy.optimize import root_scalar
import dynamics as dyn

def G1(th1, th2):
    """Calculates Torque required at Joint 1 (Gravity compensation)."""
    return dyn.g*dyn.lc1*dyn.m1*np.sin(th1) + \
        dyn.g*dyn.m2*(dyn.l1*np.sin(th1) + dyn.lc2*np.sin(th1+th2))

def G2(th1, th2):
    """Calculates internal forces at Joint 2 (Gravity vs Stiffness)."""
    return dyn.g*dyn.m2*dyn.lc2*np.sin(th1+th2) + \
        3.0*dyn.k*np.sin(th2)*(np.cos(th2)**2)

def equilibrium(th1, x0=None):
    """
    Solves for theta2 given fixed theta1 using Newton-Raphson.
    
    Args:
        th1: The fixed angle for joint 1.
        x0: Initial guess for theta2. If None, defaults to -th1.
    """

    # 1. Define the function f(theta2)
    def f(th2):
        return G2(th1, th2)

    # 2. Define the derivative f'(theta2) for Newton-Raphson
    def f_prime(th2):
        # Term 1 derivative: g*m2*lc2 * cos(th1+th2)
        term1_prime = dyn.g * dyn.m2 * dyn.lc2 * np.cos(th1 + th2)
        
        # Term 2 derivative: 3k * d/dth2 [ sin(th2)*cos^2(th2) ]
        # Product rule: cos(th2)*cos^2(th2) + sin(th2)*[-2cos(th2)sin(th2)]
        #             = cos^3(th2) - 2sin^2(th2)cos(th2)
        term2_prime = 3.0 * dyn.k * (
            np.cos(th2)**3 - 2.0 * (np.sin(th2)**2) * np.cos(th2)
        )
        
        return term1_prime + term2_prime

    # 3. Choose a robust initial guess if none provided
    if x0 is None:
        # A root is typically near -th1 due to the gravity term
        x0 = -float(th1)

    # 4. Solve for theta2 using Newton-Raphson
    # We pass fprime (the derivative) to speed up convergence
    sol = root_scalar(f, x0=x0, fprime=f_prime, method="newton")

    if not sol.converged:
        raise RuntimeError(f"Equilibrium not found for theta1={th1} using Newton-Raphson")

    th2_star = sol.root

    # 5. Calculate required torque
    tau_star = G1(th1, th2_star)

    # 6. Return full state and control vectors
    xx_e = np.array([th1, th2_star, 0.0, 0.0])
    uu_e = np.array([tau_star])

    return xx_e, uu_e