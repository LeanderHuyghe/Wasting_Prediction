from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
import itertools


def correlation_heatmap_missing(df, path):
    """
    This function creates the plot for missing values
    :param df: The dataframe that we make the plot from
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
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    districts = np.sort(df.district.unique())
    fig, ax = plt.subplots(nrows=11, ncols=7, figsize=(30,40))
    axs = ax.ravel()
    data = np.arange(0,73)
    for ax, d in zip(axs.ravel(), data):
        ax.plot(df[df.district==df.district[d]]['n_conflict_total'])
        ax.set_title(districts[d])
    plt.savefig(path + 'original_conflict_districts.png')

def plot_original_price_of_water(df, path):
    """
    This function creates the plot for original_water_price
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title("Original water price data", size=14)
    df['Price of water'].plot();
    plt.savefig(path + 'original_water_price.png')

def plot_imputed_conflict_districts_knn(df, path):
    """
    This function creates the plot for imputed_conflict_districts_knn
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    districts = np.sort(df.district.unique())
    fig, ax = plt.subplots(nrows=11, ncols=7, figsize=(30, 40))
    axs = ax.ravel()
    data = np.arange(0, 73)
    for ax, d in zip(axs.ravel(), data):
        ax.plot(df[df.district == df.district[d]]['conflicts'])
        ax.set_title(districts[d])
    plt.savefig(path + 'imputed_conflict_districts_knn.png')

def plot_imputed_price_of_water_MICE(df, path):
    """
    This function creates the plot for imputed_water_price_MICE
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(10, 7))
    plt.title('Imputed water price data MICE', size=14)
    df['price_of_water'].plot();
    plt.savefig(path + 'imputed_water_price_MICE.png')

def plot_correlation(df, path):
    """
    This function creates the plot for correlation_plot
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, );
    plt.savefig(path + 'correlation_plot.png')

def plot_bar_missing(df, path):
    """
    This function creates the plot for bar_missing
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    missing = df.isna().sum().reset_index().rename(
        columns={'index': 'Feature', 0: 'Number of Missing Values'}).sort_values('Number of Missing Values',
                                                                                 ascending=False)
    missing = missing[missing['Number of Missing Values'] > 0].reset_index().iloc[:, 1:]
    missing.loc[2, 'Feature'] = 'Conflicts'
    missing.loc[3, 'Feature'] = 'Numeric Increase'
    missing.loc[4, 'Feature'] = 'Boolean Increase'
    missing.loc[5, 'Feature'] = 'Prevalence Estimate'
    missing.loc[6, 'Feature'] = 'Lagged Prevalence'
    missing.loc[7, 'Feature'] = 'Population'
    missing.loc[8, 'Feature'] = 'NDVI'
    missing.loc[9, 'Feature'] = 'IPC'

    plt.figure(figsize=(16,10))
    ax = sns.barplot(x=missing['Feature'], y=missing['Number of Missing Values'], palette='bwr');
    ax.grid(False)
    ax.spines['left'].set_visible(True)
    plt.box(False)
    plt.xticks(rotation=90);
    plt.xlabel('', weight='bold')
    plt.ylabel('', weight='bold')

    plt.savefig(path + 'bar_missing.png', dpi=300, bbox_inches='tight', transparent=True)

def scatter_ndvi(df, path):
    """
    This function creates the plot for scatter_ndvi
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    df.loc[582, 'imputed'] = 'actual'
    df.loc[583, 'imputed'] = 'actual'
    sns.set_style('whitegrid')
    sns.despine()
    df = df.rename(columns={'imputed': 'Data Type'})
    ax = sns.jointplot(x=df['ndvi'], y=df['next_prevalence'], kind='scatter', hue=df['Data Type'], hue_order=['actual', 'imputed']);
    ax.set_axis_labels('NDVI', 'Prevalence')
    plt.savefig(path + 'scatter_ndvi.png')

def scatter_conflicts(df, path):
    """
    This function creates the plot for scatter_conflicts
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    df.loc[582, 'imputed'] = 'actual'
    df.loc[583, 'imputed'] = 'actual'
    sns.set_style('whitegrid')
    sns.despine()
    df = df.rename(columns={'imputed': 'Data Type'})
    ax = sns.jointplot(x=df['conflicts'], y=df["next_prevalence"], kind='scatter', hue=df['Data Type'], hue_order=['actual', 'imputed']);
    ax.set_axis_labels('Conflicts', 'Prevalence')
    plt.savefig(path + 'scatter_conflicts.png')

def scatter_ipc(df, path):
    """
    This function creates the plot for scatter_ipc
    :param df: The dataframe that we make the plot from
    :param path: The path that we output the plot to
    :return: outputs the plot in the outputs data path
    """
    df.loc[582, 'imputed'] = 'actual'
    df.loc[583, 'imputed'] = 'actual'
    sns.set_style('whitegrid')
    sns.despine()
    df = df.rename(columns={'imputed': 'Data Type'})
    ax = sns.jointplot(x=df['ipc'], y=df['next_prevalence'], kind='scatter', hue=df['Data Type'], hue_order=['actual', 'imputed']);
    ax.set_axis_labels('IPC', 'Prevalence')
    plt.savefig(path + 'scatter_ipc.png')

def make_confusion_matrix(y_true, y_pred, path, name, classes=None, figsize=(10, 10), text_size=15, norm=False,
                          title="Confusion Matrix", cmap=plt.cm.Purples):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    :param true_y: Array of truth labels (must be same shape as y_pred).
    :param pred_y: Array of predicted labels (must be same shape as y_true).
    :param classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    :param figsize: Size of output figure (default=(10, 10)).
    :param text_size: Size of output figure text (default=15).
    :param norm: normalize values or not (default=False).
    :param savefig: save confusion matrix to file (default=False).

    :return: A labelled confusion matrix plot comparing y_true and y_pred.
    """

    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    plt.grid(False)
    cax = ax.matshow(cm, cmap=cmap)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title=title,
           xlabel='Predicted label',
           ylabel='True label',
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black',
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black',
                     size=text_size)

    plt.savefig(path + name)
