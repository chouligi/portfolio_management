from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import compute_mean_daily_returns, find_intersection, intersect_dataframes


def print_date_range(name: str, dataframe: pd.DataFrame) -> None:
    """
    Prints the date range of the Index data

    :param name: str, the name of the Index
    :param dataframe: pandas Dataframe, the dataframe with the index data
    :return: None
    """
    print(f'Date range for "{name}" is from {str(dataframe["Date"].min())}' + f" to {str(dataframe['Date'].max())}")


def plot_ts(dataframe: pd.DataFrame, index: int) -> None:
    """
    Plots the timeseries of an Index
    :param dataframe: pandas Dataframe, the dataframe with the index data
    :param index: int, the index of the dictionary
    :return: None
    """
    if "Price" in dataframe.columns:
        dataframe.plot(x="Date", y="Price")

    else:
        dataframe.plot(x="Date", y="Adj Close")

    plt.title(index)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")

    plt.show()


def print_length(name: str, dataframe: pd.DataFrame) -> None:
    """
    Prints the length of the Index dataframe

    :param name: str, the name of the Index
    :param dataframe: pandas Dataframe, the dataframe with the index data
    :return: None
    """
    print(f'Number of days in "{name}" is {len(dataframe)}')


def inspect_data(portfolio_dict: dict) -> None:
    # Print length of each dataframe
    print("############### Printing Number of Rows per index...###############")
    for idx in range(len(portfolio_dict["names"])):
        print_length(portfolio_dict["names"][idx], portfolio_dict["frames"][idx])
    print("\n")
    print("############### Printing Data Ranges per index...###############")
    for idx in range(len(portfolio_dict["names"])):
        print_date_range(portfolio_dict["names"][idx], portfolio_dict["frames"][idx])
    print("\n")
    for idx in range(len(portfolio_dict["names"])):
        plot_ts(portfolio_dict["frames"][idx], portfolio_dict["description"][idx])


def print_risk_and_return_portfolio(portfolio_dict: dict) -> None:
    portfolio_returns, portfolio_std = portfolio_annualised_performance(portfolio_dict)
    _, min_date, max_date = find_intersection(portfolio_dict["frames"])

    print(
        f"The annualized return of the portfolio is {round(portfolio_returns * 100, 2)}% "
        f"and the risk is {round(portfolio_std * 100, 2)}%"
    )
    print("\n")
    print(f"The portfolio minimum date is {min_date} and maximum date is {max_date}")


def plot_correlation_matrix(list_df: List[pd.DataFrame], names: List[str]) -> None:
    """
    Plots the correlation matrix

    :param list_df: List of pandas DataFrames, the index data
    :param names: names: List, List of strings with the Index names
    :return: None
    """
    corr = compute_correlation_matrix(list_df)

    corrMatrix = pd.DataFrame(corr, columns=names, index=names)
    if len(list_df) <= 4:
        plt.subplots(figsize=(15, 10))
    else:
        plt.subplots(figsize=(20, 10))

    sns.heatmap(corrMatrix, annot=True, fmt="g")
    plt.title("Correlation matrix")

    plt.show()


def create_matrix(list_df: List[pd.DataFrame]) -> np.array:
    """
    Creates numpy array with the intersected dataframes

    :param list_df: List of pandas DataFrames, the index data
    :return: np.array, the array with the intersected dataframes
    """
    intersected = intersect_dataframes(list_df)
    array_list = []

    for df in intersected:
        array_list.append(np.array(df["perc_change"]))
    matrix = np.array(array_list)

    return matrix


def compute_correlation_matrix(list_df: List[pd.DataFrame]) -> np.array:
    """
    Computes the correlation matrix

    :param list_df: List of pandas DataFrames, the index data
    :return: np.array, the correlation matrix
    """
    matrix = create_matrix(list_df)
    corrMatrix = np.corrcoef(matrix)

    return corrMatrix


def compute_covariance_matrix(list_df: List[pd.DataFrame]) -> np.array:
    """
    Computes the covariance matrix

    :param list_df: List of pandas DataFrames, the index data
    :return: np.array, the correlation matrix
    """
    matrix = create_matrix(list_df)

    covMatrix = np.cov(matrix, bias=True)
    return covMatrix


def plot_covariance_matrix(list_df: List[pd.DataFrame], names: List[str]) -> None:
    """
    Plots the covariance matrix

    :param list_df: List of pandas DataFrames, the index data
    :param names: names: List, List of strings with the Index names
    :return: None
    """
    cov = compute_covariance_matrix(list_df)

    covMatrix = pd.DataFrame(cov, columns=names, index=names)
    if len(list_df) <= 4:
        plt.subplots(figsize=(15, 10))
    else:
        plt.subplots(figsize=(20, 10))

    sns.heatmap(covMatrix, annot=True, fmt="g")
    plt.title("Covariance matrix")

    plt.show()


def portfolio_annualised_performance(portfolio_dictionary: dict) -> Tuple[float, float]:
    """
    Computes the annualized risk and return of a portfolio

    :param portfolio_dictionary: dictionary, the dictionary with the Index data
    :return: Tuple[float, float], the return and the risk (standard deviation) of the portfolio
    """
    cov_matrix = compute_covariance_matrix(portfolio_dictionary["frames"])

    weights = np.array(portfolio_dictionary["weights"])

    mean_daily_returns = compute_mean_daily_returns(portfolio_dictionary)

    portfolio_returns = np.sum(mean_daily_returns * weights) * 252

    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    return portfolio_returns, portfolio_std
