import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



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

def plot_original_price_of_water_MICE(df, path):
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

def plot_imputed_conflict_districts_spline(df, path):
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
        ax.plot(df[df.district == df.district[d]]["n_conflict_total_spline"])
        ax.set_title(districts[d])
    plt.savefig(path + 'imputed_conflict_districts_spline.png')

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
        ax.plot(df[df.district == df.district[d]]["n_conflict_total_knn"])
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
    df["Price of water MICE"].plot();
    plt.savefig(path + 'imputed_water_price_MICE.png')

def plot_imputed_price_of_water_knn(df, path):
    """
    This function creates the plot for original_conflict_districts
    :param df: The dateframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title("Imputed water price data knn", size=14)
    df["Price of water KNN"].plot();
    plt.savefig(path + 'imputed_water_price_knn.png')

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