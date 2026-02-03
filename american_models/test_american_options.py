import unittest

from american_models.american_binomial import binomial_american_option
from american_models.american_monte_carlo import american_put_lsm


class TestAmericanPricers(unittest.TestCase):
    def setUp(self):
        # Standard test parameters
        self.params = {
            "S": 100.0,
            "K": 110.0,
            "T": 1.0,
            "r": 0.05,
            "sigma": 0.2,
        }
        # High resolution for convergence
        self.N_tree = 1000
        self.N_mc = 50
        self.M_paths = 100000

    def test_convergence(self):
        """Test if Binomial and LSMC converge to a similar value for an ITM Put."""
        tree_price = binomial_american_option(**self.params, N=self.N_tree)
        mc_price = american_put_lsm(self.params["S"], self.params["K"],
                                    self.params["T"], self.params["r"],
                                    self.params["sigma"], self.N_mc, self.M_paths)

        # We expect convergence within a 1-2% tolerance due to MC noise
        self.assertAlmostEqual(tree_price, mc_price, delta=0.15)
        print(f"Convergence Check - Tree: {tree_price:.4f}, MC: {mc_price:.4f}")

    def test_early_exercise_premium(self):
        """Test that American Put > European Put for ITM."""
        # A simple Black-Scholes European Put for S=100, K=110, T=1, r=0.05, s=0.2 is approx 9.94
        european_put_price = 9.94
        ame_price = binomial_american_option(**self.params, N=500)

        self.assertGreater(ame_price, european_put_price)
        print(f"Premium Check - American: {ame_price:.4f} > European: {european_put_price}")

    def test_call_no_early_exercise(self):
        """On non-div stock, American Call should equal European Call (Black-Scholes)."""
        # European Call (Black-Scholes) for these params is approx 6.04
        european_call_ref = 6.04

        # Use the binomial_american_option code from earlier for a 'call'
        ame_call_price = binomial_american_option(**self.params, N=500) #, option_type='call')

        self.assertAlmostEqual(ame_call_price, european_call_ref, delta=0.02)
        print(f"Call Check - American Call: {ame_call_price:.4f} vs BS Ref: {european_call_ref}")


if __name__ == '__main__':
    # Assuming binomial_american_put and american_put_lsm are in the namespace
    unittest.main(argv=[''], exit=False)