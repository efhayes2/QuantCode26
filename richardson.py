from american_models.american_binomial import binomial_american_option
from american_models.test_data import american_option_test_params


def price_american_richardson(S, K, T, r, sigma, N):
    """
    Uses Richardson Extrapolation to accelerate Binomial Tree convergence.
    Combines a tree of size N and 2N to estimate the continuous limit.
    """
    # 1. Price with N steps
    price_n = binomial_american_option(S, K, T, r, sigma, N)

    # 2. Price with 2N steps
    price_2n = binomial_american_option(S, K, T, r, sigma, 2 * N)

    # 3. Extrapolate to the limit
    # Formula: V_limit = (2^p * V_2n - V_n) / (2^p - 1)
    # For CRR, the convergence order p is 1.
    price_limit = 2 * price_2n - price_n

    return price_limit


if __name__ == "__main__":
# --- Run the Test Case ---
    params = american_option_test_params
    N_base = 1000

    p_n = binomial_american_option(**params, N=N_base)
    p_2n = binomial_american_option(**params, N=N_base * 2)
    p_rich = price_american_richardson(**params, N=N_base)

    print(f"Price (N={N_base}):      {p_n:.6f}")
    print(f"Price (N={N_base * 2}):    {p_2n:.6f}")
    print(f"Richardson Limit:   {p_rich:.6f}")