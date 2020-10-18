import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco

from .data_inspection_utils import portfolio_annualised_performance
from .utils import compute_perc_change, construct_portfolio_dictionary


def random_portfolios(num_portfolios: int, portfolio: List[pd.DataFrame], names: List[str], risk_free_rate: float,
                      description: List[str]) -> Tuple[np.array, np.array]:
    """
    Creates random portfolios, with random allocations

    :param num_portfolios: int, the number of the portfolios
    :param portfolio: List[pd.DataFrame]: List of the Index data
    :param names: List[str], List of the names of the Indices
    :param risk_free_rate: float, the risk free rate
     (see: https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=billrates)
    :param description: List[str], List of the description of the Indices
    :return: Tuple [np.array, np.array], Tuple of the results and weights of each portfolio
    """
    for idx in range(len(portfolio)):
        portfolio[idx] = compute_perc_change(portfolio[idx])

    num_etfs = len(portfolio)

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for idx in range(num_portfolios):
        # generate num_etfs random numbers
        weights = np.random.random(num_etfs)
        # Make sure that their sum, sums up to 1
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_dict = construct_portfolio_dictionary(portfolio, names, weights, description)

        portfolio_returns, portfolio_std = portfolio_annualised_performance(portfolio_dict)

        results[0, idx] = portfolio_std
        results[1, idx] = portfolio_returns
        results[2, idx] = (portfolio_returns - risk_free_rate) / portfolio_std

    return results, weights_record


def display_simulated_ef_with_random(portfolio: List[pd.DataFrame], names: List[str], num_portfolios: int,
                                     risk_free_rate: float, description: List[str],
                                     directory: str = 'figures/') -> None:
    """
    Displays and stores simulated efficient frontier using random portfolios

    :param portfolio: List[pd.DataFrame]: List of the Index data
    :param names: List[str], List of the names of the Indices
    :param num_portfolios: int, the number of the portfolios
    :param risk_free_rate: float, the risk free rate
     (see: https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=billrates)
    :param description: List[str], List of the description of the Indices
    :param directory: str, the directory with the figures
    :return:
    """
    results, weights = random_portfolios(num_portfolios, portfolio, names, risk_free_rate, description)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=names, columns=['allocation'])
    # max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.apply(lambda x: round(x * 100, 2))
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=names, columns=['allocation'])
    # min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.apply(lambda x: round(x * 100, 2))
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(15, 10))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title(f'Efficient Frontier: {num_portfolios} simulated portfolios', size=25)
    plt.xlabel('Annualised volatility', size=20)
    plt.ylabel('Annualised returns', size=20)
    plt.legend(labelspacing=0.8)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'{directory}efficient_frontier_{num_portfolios}_portfolios.png', dpi=250)

    plt.show()


def neg_sharpe_ratio(weights: List[float], portfolio: List[pd.DataFrame], names: List[str],
                     risk_free_rate: float) -> float:
    """
    Computes the negative sharpe ratio

    :param weights: List[float], List with the allocation weights
    :param portfolio: List[pd.DataFrame], List of the Index data
    :param names: List[str], List of the names of the Indices
    :param risk_free_rate: float, the risk free rate
    :return: float, the negative sharpe ratio
    """

    portfolio_dictionary = construct_portfolio_dictionary(portfolio, names, weights, ["NA"])
    p_ret, p_std = portfolio_annualised_performance(portfolio_dictionary)
    return -(p_ret - risk_free_rate) / p_std


def portfolio_volatility(weights: List[float], portfolio: List[pd.DataFrame], names: List[str]) -> float:
    """
    Computes the portfolio's volatility (standard deviation)

    :param weights: List[float], List with the allocation weights
    :param portfolio: List[pd.DataFrame], List of the Index data
    :param names: List[str], List of the names of the Indices
    :return: float, standard deviation of the portfolio
    """

    portfolio_dictionary = construct_portfolio_dictionary(portfolio, names, weights, ["NA"])
    p_ret, p_std = portfolio_annualised_performance(portfolio_dictionary)
    return p_std


def max_sharpe_ratio(portfolio: List[pd.DataFrame], names: List[str],
                     risk_free_rate: float) -> sco.optimize.OptimizeResult:
    """
        Maximizes the sharpe ratio

    :param portfolio: List[pd.DataFrame], List of the Index data
    :param names: List[str], List of the names of the Indices
    :param risk_free_rate: float, the risk free rate
    :return: float, the maximum
    """

    num_assets = len(portfolio)
    args = (portfolio, names, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def min_variance(portfolio: List[pd.DataFrame], names: List[str]) -> sco.optimize.OptimizeResult:
    """
        Minimizes the variance


    :param portfolio: List[pd.DataFrame], List of the Index data
    :param names: List[str], List of the names of the Indices
    :return:
    """

    num_assets = len(portfolio)
    args = (portfolio, names)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result
