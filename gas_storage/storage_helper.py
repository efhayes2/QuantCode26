import numpy as np
from dataclasses import dataclass


def get_fwd_curve(T_days):
    """Utility function to provide a consistent forward curve."""
    # Key Days: 0 (Jun 10), 83 (Sep 1), 216 (Jan 12), 271 (Mar 8), 365 (Jun 10)
    days = [0, 83, 216, 271, 365]
    prices = [15.01, 15.01, 18.00, 25.83, 15.73]
    all_days = np.arange(T_days + 1)
    return np.interp(all_days, days, prices)


@dataclass
class MarketEnvironment:
    kappa: float = 0.05
    sigma: float = 0.0315
    M: int = 2000
    T_days: int = 365

    @property
    def fwd_curve(self):
        return get_fwd_curve(self.T_days)


@dataclass
class MarketEnvironment3F:
    # Short-term factor (Spikes)
    kappa: float = 0.05
    sigma_chi: float = 0.0315

    # Long-term factor (Equilibrium)
    sigma_xi: float = 0.0100

    # Correlation between factors
    rho: float = 0.3

    M: int = 2000
    T_days: int = 365

    @property
    def fwd_curve(self):
        return get_fwd_curve(self.T_days)


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