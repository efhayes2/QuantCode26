import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from test_data import american_option_test_params


def black_scholes_european(S, K, T, r, sigma, option_type='put'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_american_baw(S, K, T, r, sigma, option_type='put'):
    """Barone-Adesi and Whaley Quadratic Approximation."""
    if T <= 0: return max(K - S, 0) if option_type == 'put' else max(S - K, 0)

    # 1. Pre-calculate common terms
    sig2 = sigma ** 2
    k = 2 * r / sig2
    h = 1 - np.exp(-r * T)

    euro_price = black_scholes_european(S, K, T, r, sigma, option_type)

    if option_type == 'call':
        # (Simplified for non-dividend stocks, American Call = European Call)
        return euro_price

        # 2. Solve for Critical Price S_star for Put
    q2 = (-(k - 1) - np.sqrt((k - 1) ** 2 + (4 * k / h))) / 2

    def objective_function(s_star):
        euro = black_scholes_european(s_star, K, T, r, sigma, 'put')
        # d1 at s_star
        d1 = (np.log(s_star / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return (K - s_star) - euro + (s_star / q2) * (1 - norm.cdf(-d1))

    # Boundary search for S_star
    try:
        s_star = brentq(objective_function, 1e-5, K)
    except ValueError:
        return max(K - S, euro_price)

    # 3. Final Valuation
    if S > s_star:
        d1_sstar = (np.log(s_star / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        A2 = -(s_star / q2) * (1 - norm.cdf(-d1_sstar))
        return euro_price + A2 * (S / s_star) ** q2
    else:
        return K - S

if __name__ == "__main__":

    params = american_option_test_params
    # Example Usage:
    # S0=100, K=110, T=1, r=0.05, sigma=0.2, Steps=50, Paths=100000
    # price = price_american_baw(100, 110, 1, 0.05, 0.2, 'put')
    price = price_american_baw(**params)

    # binomial pricer is 11.9051
    print(f"American Put Price (BAW): {price:.4f}")