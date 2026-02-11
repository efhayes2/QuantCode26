import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    CALL = 1
    PUT = -1


@dataclass
class MarketData:
    """Wraps the state of the market at a point in time."""
    spot: float
    rate: float
    vol: float
    div: float = 0.0


class PricingEngine(ABC):
    """Abstract base class for all pricing logic."""

    @abstractmethod
    def calculate(self, instrument, market_data: MarketData) -> float:
        pass


class BlackScholesEngine(PricingEngine):
    """Closed-form Analytic European Engine."""

    def calculate(self, instrument, market: MarketData) -> float:
        S, K = market.spot, instrument.strike
        T, r, sigma, q = instrument.T, market.rate, market.vol, market.div

        if T <= 0:
            return max(0, instrument.option_type.value * (S - K))

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if instrument.option_type == OptionType.CALL:
            price = (S * np.exp(-q * T) * norm.cdf(d1) -
                     K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                     S * np.exp(-q * T) * norm.cdf(-d1))
        return price


class EuropeanOption:
    """The Instrument: Defines the contract, delegates the math."""

    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        self.strike = strike
        self.T = expiry  # Time to maturity in years
        self.option_type = option_type
        self._engine = None

    def set_engine(self, engine: PricingEngine):
        self._engine = engine

    def price(self, market_data: MarketData) -> float:
        if self._engine is None:
            raise ValueError("Pricing engine not set.")
        return self._engine.calculate(self, market_data)


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define Market Environment
    market = MarketData(spot=100.0, rate=0.05, vol=0.20, div=0.02)

    # 2. Define the Instrument (Contract)
    option = EuropeanOption(strike=105.0, expiry=1.0, option_type=OptionType.CALL)

    # 3. Choose the Engine
    option.set_engine(BlackScholesEngine())

    # 4. Get Price
    print(f"Option Price: {option.price(market):.4f}")