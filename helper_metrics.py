# Imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import itertools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer


########################################################################################################
# ROC CURVE
def roc_curve_gen(y_true, y_pred, model_name="Model"):
    """
    Parameters
    ----------
    y_true
    y_pred
    model_name: for title of plot
    Returns
    -------
    ROC curve
    """
    fpr, tpr, threshold = roc_curve(y_true, y_pred)

    plt.plot(fpr, tpr, label="Predicted Increase")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


########################################################################################################
# CONFUSION MATRIX
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False,
                          title="Confusion Matrix",
                          savefig=False,
                          cmap=plt.cm.Blues):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    Parameters
    ----------
    cmap
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
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
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")


########################################################################################################
# SAVE MODEL
def save_model(model, filename):
    """
    Parameters
    ----------
    model: sklearn model
    filename: should be a string in the format "model_name.sav"
    Returns
    -------
    saved pkl model
    """
    joblib.dump(model, filename)


########################################################################################################
# CALCULATE METRICS
def calculate_results(y_true, y_pred, average="weighted"):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


########################################################################################################
# PLOT TIME SERIES DATA
def plot_time_series(timesteps, values, format='-', start=0, end=None, label=None):
    """
      Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
      Parameters
      ---------
      timesteps : array of timesteps
      values : array of values across time
      format : style of plot, default "."
      start : where to start the plot (setting a value will index from start of timesteps & values)
      end : where to end the plot (setting a value will index from end of timesteps & values)
      label : label to show on plot of values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Date")
    plt.ylabel("Next Prevalence")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)


########################################################################################################
# CALCULATE MISSING VALUES
def count_missing_district(df):
    """ This function shows the total number of missing values in each column
    per district"""

    df = df.groupby('district')
    df = df.count().rsub(df.size(), axis=0)
    return df


def count_missing_district_total(df):
    """This function shows the total number of missing values per district
     """

    df = df.groupby('district')
    df = df.count().rsub(df.size(), axis=0)
    df.reset_index()
    df = df.sum(axis=1)
    return df


########################################################################################################
# TEST IMPUTATION PERFORMANCES
def impute_score(df, features, method, scale, n_neighbours=5):
    """

    Parameters
    ----------
    features (string): column to impute on
    df (dataframe): dataframe
    method (string): imputation strategy
    n_neighbours (int): neighbours for knn and mice imputation
    scale (string): the range of values in the feature

    Returns
    -------
    Evaluation of imputation method for a particular feature
    """

    strategy = {'mean': SimpleImputer(strategy='mean'),
                'median': SimpleImputer(strategy='median'),
                'knn': KNNImputer(n_neighbors=n_neighbours),
                'mice': IterativeImputer(max_iter=100, n_nearest_features=n_neighbours, random_state=0)}

    try:
        imputer = strategy[method]
    except:
        raise ValueError(f"Method argument requires one of 'mean','median','knn','mice'. \n {method} is not a valid strategy.")

    # Create copy of dataframe and only include continuous features
    df_test = df.copy()
    df_test = df_test.select_dtypes(exclude=["category", "object"])
    df_test = df_test.dropna(axis=0)

    # Scale the dataframe
    scaler = MinMaxScaler()
    df_test_scaled = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

    # Setting Feature column
    features = [features]

    # Set seed for reproducibility
    np.random.seed(18)

    #  Inserting NaN values into Experiment Group
    for col in df_test_scaled[features]:
        # 20% of the data will be removed (frac = 0.2)
        # Rows may be selected more that once (replace = true)
        df_test_scaled.loc[df_test_scaled.sample(frac=0.2, replace=True).index, col] = np.nan

    # Creating a list of indices
    nan_cols = df_test_scaled[features]
    nan_cols = nan_cols[nan_cols.isna().any(axis=1)]
    null_idx = list(nan_cols.index)

    # Creating Answer key to compare future results against
    answer_key = df_test.iloc[null_idx]

    # Impute
    df_test_imputed = pd.DataFrame(imputer.fit_transform(df_test_scaled), columns=df_test_scaled.columns)

    # Invert scaling
    inverse_df_test_imputed = pd.DataFrame(scaler.inverse_transform(df_test_imputed), columns=df_test_imputed.columns)

    # Subset data to match that of our answer key
    test = inverse_df_test_imputed.iloc[null_idx]

    # Resetting indexes of test and answer_key for iteration
    test = test.reset_index()
    test.drop(['index'], axis=1, inplace=True)
    answer_key = answer_key.reset_index()
    answer_key.drop(['index'], axis=1, inplace=True)

    # Calculate results
    results = pd.DataFrame((round((answer_key - test), 3)))

    # calculate RMSE
    squared_terms = []
    for col in results[features]:
        for i in range(len(results)):
            if results[col][i] != 0.00 or results[col][i] != -0.00:
                error = results[col][i]
                squared_error = error ** 2
                squared_terms.append(squared_error)

    num_nan = df_test_scaled.isna().sum().sum()
    sum_sqr_err = sum(squared_terms)
    mse = sum_sqr_err / num_nan
    rmse = np.round(np.sqrt(mse), 3)
    # return pd.DataFrame({"RMSE": rmse, "SCALE": scale},index=[0])
    print(f"RMSE for {method.upper()} imputation in {features[0]}: {rmse} \nSCALE: {scale}")
