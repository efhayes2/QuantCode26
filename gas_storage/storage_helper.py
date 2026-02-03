import numpy as np
from dataclasses import dataclass


def get_poly_basis(prices):
    """Constructs the design matrix for degree-3 polynomial regression."""
    return np.column_stack([np.ones(len(prices)), prices, prices ** 2, prices ** 3])


def get_poly_basis_2_factor(chi, xi):
    """
    Constructs the design matrix for a bivariate degree-3 polynomial.
    State variables: chi (short-term) and xi (long-term).
    """
    # Linear and Constant
    ones = np.ones(len(chi))

    # Interaction and higher order terms
    # Total of 10 basis functions:
    # 1, chi, xi, chi^2, xi^2, chi*xi, chi^3, xi^3, chi^2*xi, chi*xi^2
    return np.column_stack([
        ones,
        chi, xi,  # Linear
        chi ** 2, xi ** 2,  # Quadratic
        chi * xi,  # Interaction (Critical for 2F)
        chi ** 3, xi ** 3,  # Cubic
        (chi ** 2) * xi,  # Cross-cubic A
        chi * (xi ** 2)  # Cross-cubic B
    ])


import numpy as np
from dataclasses import dataclass, field


@dataclass
class ForwardCurve:
    """Handles the deterministic seasonal curve via interpolation."""
    nodes_days: np.ndarray
    nodes_prices: np.ndarray

    def get_prices(self, T_days):
        all_days = np.arange(T_days + 1)
        return np.interp(all_days, self.nodes_days, self.nodes_prices)


@dataclass
class MarketEnvironment:
    """1-Factor Environment."""
    forward_market: ForwardCurve
    kappa: float = 0.05
    sigma: float = 0.0315
    M: int = 20000
    T_days: int = 365

    @property
    def fwd_curve(self):
        return self.forward_market.get_prices(self.T_days)

    def generate_paths(self):
        fwd = self.fwd_curve
        S = np.zeros((self.T_days + 1, self.M))
        x = np.zeros((self.T_days + 1, self.M))

        x[0] = np.log(fwd[0])
        S[0] = fwd[0]
        phi = np.exp(-self.kappa)
        stdev = np.sqrt((self.sigma ** 2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa)))

        for t in range(1, self.T_days + 1):
            m = np.log(fwd[t]) - (self.sigma ** 2 / (2 * self.kappa))
            x[t] = x[t - 1] * phi + m * (1 - phi) + stdev * np.random.standard_normal(self.M)
            S[t] = np.exp(x[t])
        return S


@dataclass
class MarketEnvironment3F:
    """3-Factor Environment."""
    forward_market: ForwardCurve
    kappa: float = 0.05
    sigma_chi: float = 0.0315
    sigma_xi: float = 0.0100
    rho: float = 0.3
    M: int = 20000
    T_days: int = 365

    @property
    def fwd_curve(self):
        return self.forward_market.get_prices(self.T_days)

    def generate_paths_3f(self):
        fwd = self.fwd_curve
        chi = np.zeros((self.T_days + 1, self.M))
        xi = np.zeros((self.T_days + 1, self.M))
        S = np.zeros((self.T_days + 1, self.M))

        chi[0], xi[0], S[0] = 0.0, 0.0, fwd[0]

        l11, l21, l22 = 1.0, self.rho, np.sqrt(1 - self.rho ** 2)
        phi = np.exp(-self.kappa)
        stdev_chi = np.sqrt((self.sigma_chi ** 2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa)))
        stdev_xi = self.sigma_xi

        for t in range(1, self.T_days + 1):
            z1, z2 = np.random.standard_normal(self.M), np.random.standard_normal(self.M)
            eps_chi, eps_xi = l11 * z1, l21 * z1 + l22 * z2

            chi[t] = chi[t - 1] * phi + stdev_chi * eps_chi
            xi[t] = xi[t - 1] + stdev_xi * eps_xi - 0.5 * (self.sigma_xi ** 2)

            var_chi_t = (self.sigma_chi ** 2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t))
            S[t] = np.exp(chi[t] + xi[t] + np.log(fwd[t]) - 0.5 * var_chi_t)

        return S, chi, xi

@dataclass
class StorageContract:
    min_vol: float = 0.0
    max_vol: float = 250_000.0
    max_inj_rate: float = 2500.0
    max_with_rate: float = 7500.0
    initial_vol: float = 100_000.0
    target_vol: float = 100_000.0
    terminal_price: float = 15.73
    grid_step: float = 2500.0
    n_levels: int = 101

    @property
    def inventory_grid(self):
        return np.linspace(self.min_vol, self.max_vol, self.n_levels)


