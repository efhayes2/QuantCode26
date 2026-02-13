import numpy as np
from scipy.stats import norm


class FTRAuctionBidder:
    def __init__(self, predicted_value, prediction_std):
        self.v_pred = predicted_value
        self.sigma = prediction_std

    def expected_profit(self, bid_price, market_price_mean, market_price_std):
        """
        Calculates expected profit: (V_true - Bid) * P(Bid > Clearing_Price)
        We approximate the clearing price distribution based on market intelligence.
        """
        # Probability that our bid is above the market clearing price
        prob_win = norm.cdf(bid_price, loc=market_price_mean, scale=market_price_std)

        profit_if_win = self.v_pred - bid_price
        return prob_win * profit_if_win

    def optimize_bid(self, market_mean, market_std):
        """Finds the bid price that maximizes expected profit."""
        bids = np.linspace(self.v_pred - 3 * self.sigma, self.v_pred, 100)
        profits = [self.expected_profit(b, market_mean, market_std) for b in bids]

        best_bid = bids[np.argmax(profits)]
        return best_bid


if __name__ == "__main__":
    # Scenario: We think a path (e.g., West-to-East) is worth $15/MW
    # Our model has a $2 uncertainty.
    my_valuation = 15.0
    my_uncertainty = 2.0

    # We estimate based on past auctions that the clearing price
    # usually floats around $12 with a $3 deviation.
    mkt_clearing_mean = 12.0
    mkt_clearing_std = 3.0

    bidder = FTRAuctionBidder(my_valuation, my_uncertainty)

    optimal_bid = bidder.optimize_bid(mkt_clearing_mean, mkt_clearing_std)

    print(f"Predicted Value: ${my_valuation:.2f}")
    print(f"Optimal Strategic Bid: ${optimal_bid:.2f}")
    print(f"Shading Amount: ${my_valuation - optimal_bid:.2f}")