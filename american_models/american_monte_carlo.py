import numpy as np

from test_data import american_option_test_params, american_option_test_params2


def american_put_lsm(S, K, T, r, sigma, N, M, poly_degree=2):
    """
    S: Initial stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Volatility
    N: Number of time steps
    M: Number of simulated paths
    degree: Degree of the polynomial for regression
    """
    dt = T / N
    df = np.exp(-r * dt)

    np.random.seed(0)
    # 1. Path Generation (Geometric Brownian Motion)
    paths = np.zeros((N + 1, M))
    paths[0] = S
    for t in range(1, N + 1):
        z = np.random.standard_normal(M)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    # 2. Setup Payoff and Value matrices
    # cash_flows stores the actual realized value for each path
    payoffs = np.maximum(K - paths, 0)
    cash_flows = np.copy(payoffs[-1])

    # 3. Backward Induction
    for t in range(N - 1, 0, -1):
        # Identify paths that are currently In-The-Money
        itm = payoffs[t] > 0

        if np.any(itm):
            # Independent variable: Current Stock Price of ITM paths
            X = paths[t, itm]

            # Dependent variable: Discounted FUTURE cash flows of those same paths
            # Note: We use the actual cash_flows from the stopping time,
            # not just the next step's value.
            Y = cash_flows[itm] * df

            # Least Squares Regression
            coeffs = np.polyfit(X, Y, poly_degree)
            continuation_value = np.polyval(coeffs, X)

            # Exercise decision
            exercise_value = payoffs[t, itm]

            # Find where immediate exercise is better than continuation
            early_exercise = exercise_value > continuation_value

            # Update cash flows for ITM paths:
            # If we exercise, the new cash flow is the exercise value at time t.
            # If we wait, we keep the discounted future cash flow.
            # (Non-ITM paths just get their future cash flows discounted).
            cash_flows[itm] = np.where(early_exercise, exercise_value, Y)

        # For paths NOT in the money, we simply discount the future cash flow
        cash_flows[~itm] = cash_flows[~itm] * df

    # Final price is the average of all discounted cash flows at t=1
    return np.mean(cash_flows * df)


if __name__ == "__main__":
    params = american_option_test_params2
    # Example Usage:
    # S0=100, K=110, T=1, r=0.05, sigma=0.2, Steps=50, Paths=100000
    price = american_put_lsm(**params, N=8, M=16)
    print(f"American Put Price (LSM): {price:.4f}")