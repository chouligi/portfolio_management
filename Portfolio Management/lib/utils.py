import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import seaborn as sns

# from .data_inspection_utils import compute_covariance_matrix

"""
Credits: https://github.com/tthustla/efficient_frontier/blob/master/Efficient%20_Frontier_implementation.ipynb
"""

plt.rcParams["figure.figsize"] = (15.0, 8.0)

#  Set font sizes
SMALL_SIZE = 20
MEDIUM_SIZE = 23
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sns.set_style('whitegrid')

plt.style.use('fivethirtyeight')


def apply_perc_change(portfolio: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """

    Applies the computation of the percentage change

    :param portfolio: List, List of dataframes with the Index data
    :return: portfolio: List, List of datafarames with applied percentage change
    """
    for idx in range(len(portfolio)):
        portfolio[idx] = compute_change(portfolio[idx])
        portfolio[idx] = compute_perc_change(portfolio[idx])

    return portfolio


def compute_change(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the change in the price (Current - Open)

    :param dataframe: pandas DataFrame, the dataframe with the Index data
    :return: dataframe: pandas Dataframe, dataframe with computed change
    """
    if 'Price' in dataframe.columns:
        dataframe = dataframe[['Date', 'Open', 'Price']][dataframe['Open'] != None]
        dataframe['Open'] = dataframe['Open'].astype(float)
        dataframe['Price'] = dataframe['Price'].astype(float)
        dataframe['Change'] = dataframe['Price'] - dataframe['Open']
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    else:
        dataframe = dataframe[['Date', 'Open', 'Adj Close']][dataframe['Open'] != 'null']
        dataframe['Open'] = dataframe['Open'].astype(float)
        dataframe['Adj Close'] = dataframe['Adj Close'].astype(float)
        dataframe['Change'] = dataframe['Adj Close'] - dataframe['Open']
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    return dataframe


def compute_perc_change(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the percentage change in the price

    :param dataframe: pandas DataFrame, the dataframe with the Index data
    :return: pandas Dataframe, dataframe with computed percentage change
    """
    if 'Price' in dataframe.columns:
        dataframe['perc_change'] = dataframe['Price'].pct_change()
    else:
        dataframe['perc_change'] = dataframe['Adj Close'].pct_change()

    return dataframe


def construct_portfolio_dictionary(dataframes_list: List[pd.DataFrame], names: List[str],
                                   weights: Union[float, np.array],
                                   description: List[str]) -> dict:
    """
    Creates dictionary with the Index data, their names, their description and weights

    :param dataframes_list: List, List of dataframes with the Index data
    :param names: List, List of strings with the Index names
    :param weights: List, List of floats with the Index weights
    :param description: List, List of strings with the Index descriptions
    :return: dictionary
    """
    portfolio_dict = {'names': names,
                      'frames': dataframes_list,
                      'weights': weights,
                      'description': description}

    return portfolio_dict


def compute_return_since_inception(dataframe: pd.DataFrame) -> float:
    """
    Computes the return of an Index since inception
    :param dataframe: pandas Dataframe, Dataframe with the Index data
    :return: float, the return since inception
    """
    if 'Price' in dataframe.columns:
        p0 = float(dataframe[dataframe['Date'] == dataframe['Date'].min()]['Price'])
        p1 = float(dataframe[dataframe['Date'] == dataframe['Date'].max()]['Price'])
    else:
        p0 = float(dataframe[dataframe['Date'] == dataframe['Date'].min()]['Adj Close'])
        p1 = float(dataframe[dataframe['Date'] == dataframe['Date'].max()]['Adj Close'])

    return (p1 - p0) / p0


def print_return_since_inception(dataframe: pd.DataFrame, name: str) -> None:
    """
    Prints the return since inceptipn

    :param dataframe: pandas Dataframe, Dataframe with the Index data
    :param name: str, the index Name
    :return: None
    """
    returns_since_inception = compute_return_since_inception(dataframe)

    print(
        f'The returns since inception {str(dataframe["Date"].min().date())} for "{name}" is '
        f'{round(returns_since_inception * 100, 2)}%')


def find_intersection(list_df: List[pd.DataFrame]) -> Tuple[pd.Series, str, str]:
    """
    Finds the intersection of the dataframes in the List, to avoid NaN in computations
    :param list_df: List of pandas DataFrames, the index data
    :return: Tuple[pd.Series, str, str], Series of common dates, minimum date, maximum date
    """

    # Filter only non-null values
    for idx in range(len(list_df)):
        list_df[idx] = list_df[idx][list_df[idx]['perc_change'].notnull()]

    intersection_list = set(list_df[0]['Date']).intersection(set(list_df[1]['Date']))
    for df in list_df[2:]:
        intersection_list = intersection_list.intersection(set(df['Date']))

    common_dates = pd.Series(list(intersection_list))

    minimum_date = str(common_dates.min().date())
    maximum_date = str(common_dates.max().date())

    return common_dates, minimum_date, maximum_date


def intersect_dataframes(list_df: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Intersects the dataframes

    :param list_df: List of pandas DataFrames, the index data
    :return: List[pd.DataFrame], List of the intersected dataframes
    """
    intersection = list_df.copy()
    common_dates, _, _ = find_intersection(intersection)
    for idx in range(len(intersection)):
        intersection[idx] = intersection[idx][intersection[idx]['Date'].isin(common_dates)]

    return intersection


def compute_mean_daily_returns(portfolio_dictionary: dict) -> List[float]:
    """
    Computes mean daily returns of the Indices

    :param portfolio_dictionary: dictionary, the dictionary with the Index data
    :return: List[float], list with the mean daily returns of each Index
    """
    intersected = intersect_dataframes(portfolio_dictionary['frames'])
    # Compute mean daily return per ETF
    mean_daily_returns = []
    for df in intersected:
        mean_daily_returns.append(df['perc_change'].mean())

    return mean_daily_returns


def compute_annualized_returns_no_intersection(portfolio_dictionary: dict) -> float:
    """
    Computes the annualized returns of the portfolio without intersecting the dataframes (using all the available data
    per index)
         WARNING: Since the returns are computed using different time periods for each Index, this number should not be
    used for any decision, if the date ranges have large discrepancies.


    :param portfolio_dictionary: dictionary, the dictionary with the Index data
    :return: float, the annualized return of the portfolio
    """
    mean_daily_returns = []
    for df in portfolio_dictionary['frames']:
        mean_daily_returns.append(df['perc_change'].mean())

    weights = np.array(portfolio_dictionary['weights'])

    portfolio_returns = np.sum(mean_daily_returns * weights) * 252

    return portfolio_returns


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
    :return:
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
