import numpy as np

from test_data import american_option_test_params


def binomial_american_option(S, K, T, r, sigma, N, option_type='put'):
    """
    S: Current stock price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate (annualized)
    sigma: Volatility (annualized)
    N: Number of binomial steps
    option_type: 'call' or 'put'
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    a = np.exp(r * dt)  # Risk-free growth
    p = (a - d) / (u - d)  # Risk-neutral probability
    q = 1 - p

    # Initialize price levels at maturity (Step N)
    # The prices at maturity are S * u^j * d^(N-j) for j in range(N+1)
    S_at_T = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1))

    # Initialize the option value at maturity
    if option_type == 'call':
        V = np.maximum(S_at_T - K, 0)
    else:
        V = np.maximum(K - S_at_T, 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        # Current stock prices at this step i
        S_current = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))

        # Continuation value (discounted expected value)
        V = (p * V[:-1] + q * V[1:]) / a

        # American Exercise Check
        if option_type == 'call':
            V = np.maximum(V, np.maximum(S_current - K, 0))
        else:
            V = np.maximum(V, np.maximum(K - S_current, 0))

    return V[0]

if __name__ == "__main__":
    # --- Test Case Execution ---
    params = american_option_test_params


    print(f"{'Steps (N)':<10} | {'American Put Price':<20}")
    print("-" * 35)

    for n in [10, 50, 100, 500, 1000]:
        price = binomial_american_option(**params, N=n)
        print(f"{n:<10} | {price:.4f}")