from american_models.american_binomial import binomial_american_option
from american_models.american_monte_carlo import american_put_lsm
from american_models.baw_american import price_american_baw
from american_models.bjst_american import bjerksund_stensland_2002
from richardson import price_american_richardson
from american_models.crank_nicolson import price_american_crank_nicolson
from american_models.test_data import (american_option_test_params, american_option_test_params2,
                                       american_option_test_params3)


if __name__ == "__main__":

    params_list = [american_option_test_params, american_option_test_params2,
              american_option_test_params3]

    binomial_n = 1_000
    Ns, Nt = 500, 500
    for params in params_list:
        baw = price_american_baw(**params)
        binomial = binomial_american_option(**params, N=binomial_n)
        lsmc = american_put_lsm(**params, N=50, M=100_000, poly_degree=3)
        bjst = bjerksund_stensland_2002(**params)
        richardson = price_american_richardson(**params, N=binomial_n)
        crank = price_american_crank_nicolson(**params, Ns=500, Nt=500)

        #baw = 11.905
        #binomial = 11.9728
        print('binomial         :', binomial)
        print('richardson       :', richardson)
        print('crank-nicolson   :', crank)
        print('baw              :', baw)
        print('bjst             :', bjst)
        print('lsmc             :', lsmc)
        print()
        print('--------------------------------------')