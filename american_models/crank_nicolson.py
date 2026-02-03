import numpy as np
from scipy.linalg import solve_banded

from test_data import american_option_test_params


def price_american_crank_nicolson(S, K, T, r, sigma, Ns, Nt, S_max_mult=3):
    """
    S: Current spot price
    K: Strike price
    T: Time to maturity
    Ns: Number of space steps (stock price)
    Nt: Number of time steps
    """
    S_max = K * S_max_mult
    dt = T / Nt
    ds = S_max / Ns

    # Grid setup
    s_values = np.linspace(0, S_max, Ns + 1)
    # Payoff at maturity (T)
    v = np.maximum(K - s_values, 0)

    # Precompute coefficients for the tridiagonal system
    # We solve: A * v_new = B * v_old
    j = np.arange(1, Ns)
    alpha = 0.25 * dt * (sigma ** 2 * j ** 2 - r * j)
    beta = -0.5 * dt * (sigma ** 2 * j ** 2 + r)
    gamma = 0.25 * dt * (sigma ** 2 * j ** 2 + r * j)

    # Left-hand side matrix (A) - Implicit part
    # We use solve_banded, so we need the matrix in banded form
    A_banded = np.zeros((3, Ns - 1))
    A_banded[0, 1:] = -gamma[:-1]  # Upper diagonal
    A_banded[1, :] = 1 - beta  # Main diagonal
    A_banded[2, :-1] = -alpha[1:]  # Lower diagonal

    # Right-hand side matrix coefficients - Explicit part
    # (1 + beta)v_j + alpha*v_{j-1} + gamma*v_{j+1}

    for _ in range(Nt):
        # Calculate RHS
        rhs = v[1:-1] * (1 + beta) + v[:-2] * alpha + v[2:] * gamma

        # Boundary conditions (Dirichlet)
        # S=0: Put value is K * exp(-r * (T-t))
        # S=S_max: Put value is 0
        rhs[0] += alpha[0] * K  # Simplified S=0 boundary

        # Solve the system for the internal nodes
        v[1:-1] = solve_banded((1, 1), A_banded, rhs)

        # Enforce American early exercise constraint
        v = np.maximum(v, K - s_values)

    return np.interp(S, s_values, v)


if __name__ == '__main__':
# --- Test Execution ---
    # params = {"S": 100, "K": 110, "T": 1.0, "r": 0.05, "sigma": 0.2}
    params = american_option_test_params
    # Using Ns=500 and Nt=500 for high precision
    price_cn = price_american_crank_nicolson(**params, Ns=10000, Nt=10000)
    print(f"Crank-Nicholson American Put: {price_cn:.6f}")
