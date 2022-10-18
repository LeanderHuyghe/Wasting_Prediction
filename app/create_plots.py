import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno


def correlation_heatmap_missing(df, path):
    """
    This function creates the plot for missing values
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title("Plot missing Values", size=14)
    msno.matrix(df).plot();
    plt.savefig(path + 'missing_values.png')

def plot_original_conflict_districts(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    districts = np.sort(df.district.unique())
    fig, ax = plt.subplots(nrows=11, ncols=7, figsize=(30,40))
    axs = ax.ravel()
    data = np.arange(0,73)
    for ax, d in zip(axs.ravel(), data):
        ax.plot(df[df.district==df.district[d]]["n_conflict_total"])
        ax.set_title(districts[d])
    plt.savefig(path + 'original_conflict_districts.png')

def plot_original_price_of_water(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title("Original water price data", size=14)
    df["Price of water"].plot();
    plt.savefig(path + 'original_water_price.png')

def plot_imputed_conflict_districts_knn(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    districts = np.sort(df.district.unique())
    fig, ax = plt.subplots(nrows=11, ncols=7, figsize=(30, 40))
    axs = ax.ravel()
    data = np.arange(0, 73)
    for ax, d in zip(axs.ravel(), data):
        ax.plot(df[df.district == df.district[d]]["conflicts"])
        ax.set_title(districts[d])
    plt.savefig(path + 'imputed_conflict_districts_knn.png')

def plot_imputed_price_of_water_MICE(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title("Imputed water price data MICE", size=14)
    df["price_of_water"].plot();
    plt.savefig(path + 'imputed_water_price_MICE.png')

def plot_correlation(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, );
    plt.savefig(path + 'corelation_plot.png')