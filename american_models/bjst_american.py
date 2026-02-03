import numpy as np
from scipy.stats import norm

from test_data import american_option_test_params

import numpy as np
from scipy.stats import norm


def bjerksund_stensland_2002(S, K, T, r, sigma, option_type='put', q=0.0):
    # For a Put, we use the Bjerksund-Stensland symmetry:
    # price_put(S, K, T, r, q, sigma) = price_call(K, S, T, q, r, sigma)
    if option_type.lower() == 'put':
        return _bs_call_engine(K, S, T, q, r, sigma)
    else:
        return _bs_call_engine(S, K, T, r, q, sigma)


def _bs_call_engine(S, K, T, r, q, sigma):
    b = r - q
    v2 = sigma ** 2

    # 1. Check for immediate exercise
    if b >= r:  # Call on non-dividend stock (q=0)
        # European Call = American Call
        d1 = (np.log(S / K) + (b + 0.5 * v2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # 2. Bjerksund-Stensland Parameters
    beta = (0.5 - b / v2) + np.sqrt((b / v2 - 0.5) ** 2 + 2 * r / v2)
    B_inf = (beta / (beta - 1)) * K
    B_0 = max(K, (r / (r - b)) * K)

    # 3. Boundary calculation (The 'I' value)
    # Refined boundary for 2002 model
    h = -(b * T + 2 * sigma * np.sqrt(T)) * (B_0 / (B_inf - B_0))
    I = B_0 + (B_inf - B_0) * (1 - np.exp(h))

    # 4. Trigger early exercise if S is already above boundary
    if S >= I:
        return S - K

    # 5. Core Approximation components
    alpha = (I - K) * (I ** (-beta))

    def phi(S, T, gamma, H, I):
        kappa = (2 * b / v2) + (2 * gamma - 1)
        # lambda is the drift adjustment
        lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * v2) * T
        d = -(np.log(S / H) + (b + (gamma - 0.5) * v2) * T) / (sigma * np.sqrt(T))

        return np.exp(lam) * (S ** gamma) * (norm.cdf(d) -
                                             ((I / S) ** kappa) * norm.cdf(
                    d - 2 * np.log(I / S) / (sigma * np.sqrt(T))))

    # Final Approximation Formula
    price = (alpha * (S ** beta)
             - alpha * phi(S, T, beta, I, I)
             + phi(S, T, 1, I, I)
             - phi(S, T, 1, K, I)
             - K * phi(S, T, 0, I, I)
             + K * phi(S, T, 0, K, I))

    return max(price, S - K)


if __name__ == '__main__':

    params = american_option_test_params

    p = {"S": 100, "K": 110, "T": 1.0, "r": 0.05, "sigma": 0.2}
    # Note: For the Put, we use the transformation: bjerksund_stensland_put
    price_bs = bjerksund_stensland_2002(**params)
    print(f"Bjerksund-Stensland Price: {price_bs:.4f}")