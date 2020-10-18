from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


