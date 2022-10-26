import time
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import joblib
warnings.filterwarnings("ignore")
from helper_metrics import calculate_results
from create_plots import make_confusion_matrix
from sklearn.experimental    import enable_hist_gradient_boosting
from sklearn.ensemble        import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics         import mean_absolute_error, accuracy_score



def subsets(l: object) -> object:
    subset_list = []
    for i in range(len(l) + 1):
        for j in range(i):
            subset_list.append(l[j: i])
    return subset_list

def hgbr_semiyearly(data_path, df_csv_name, validation_hgbr, model_hgbr, output_path):
    """
    This function run the model from the baseline for all districts
    :param data_path: path to the dataframe
    :param df_csv_name: name of csv
    :param start_time: start time of the function
    :param validation_hgbr: attribute for whether cross-validation runs or not
    :return: prints the results of the model
    """
    start_time = time.time()
    print("Running HGBR model on semiyearly (unimputed) data...")

    df = pd.read_csv(data_path + df_csv_name).iloc[:,1:]
    y = df.next_prevalence.dropna()
    X = df.select_dtypes(exclude=["object", "category"]).iloc[:len(y)].drop(
        ["next_prevalence", "MAM", "increase_numeric", "GAM Prevalence", "Under-Five Population",
         "Average of centy", "Average of centx"], axis=1)


    num_trees_min = 31
    num_trees_max = 64
    depth_min = 2
    depth_max = 7

    if validation_hgbr == 1:
        print("\tRunning the cross-validation...")
        start_cv = time.time()
        parameter_scores = []

        for num_trees in tqdm(range(num_trees_min, num_trees_max)):

            for depth in range(depth_min, depth_max):

                # Investigate every subset of explanatory variables
                for features in subsets(X.columns):
                    # First CV split. The 219 refers to the first 3 observations for the 73 districts in the data.
                    Xtrain = X[:219][features].copy().values
                    ytrain = y[:219]
                    Xtest = X[219:292][features].copy().values
                    ytest = y[219:292]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = HistGradientBoostingRegressor(max_leaf_nodes=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE1 = mean_absolute_error(ytest, predictions)

                    # Second CV split. The 292 refers to the first 4 observations for the 73 districts in the data.
                    Xtrain = X[:292][features].copy().values
                    ytrain = y[:292]
                    Xtest = X[292:365][features].copy().values
                    ytest = y[292:365]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = HistGradientBoostingRegressor(max_leaf_nodes=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE2 = mean_absolute_error(ytest, predictions)

                    # Calculate the mean MAE over the two folds
                    mean_MAE = (MAE1 + MAE2) / 2

                    # Store the mean MAE together with the used hyperparameters in list
                    parameter_scores.append((mean_MAE, num_trees, depth, features))

        # Sort the models based on score and retrieve the hyperparameters of the best model
        parameter_scores.sort(key=lambda x: x[0])
        best_model_score = parameter_scores[0][0]
        best_model_trees = parameter_scores[0][1]
        best_model_depth = parameter_scores[0][2]
        best_model_columns = list(parameter_scores[0][3])
        print(f"\tCross-validation is finished. ({round((time.time() - start_cv), 2)}s)")
    else:
        print("\tCross-validation is skipped. The best model is assigned manually.")
        best_model_columns = ['Total alarms', 'n_conflict_total', 'prevalence_6lag', 'month']
        best_model_trees = 31
        best_model_depth = 3

    y = df['next_prevalence'].values
    X = df[best_model_columns].values

    # If there is only one explanatory variable, the values need to be reshaped for the model
    if len(best_model_columns) == 1:
        X = X.reshape(-1, 1)

    # Peform evaluation on full data
    Xtrain = X[:365]
    ytrain = y[:365]
    Xtest = X[365:]
    ytest = y[365:]

    if model_hgbr == 0:
        print("\tWe do not load the model.")
        clf = HistGradientBoostingRegressor(max_leaf_nodes=best_model_trees, max_depth=best_model_depth, random_state=0,
                                            verbose=0)
        clf.fit(Xtrain, ytrain)
    else:
        print("\tWe load the best HGBR model.")
        clf = joblib.load(data_path + "best_model_hgbr.joblib")
    predictions = clf.predict(Xtest)

    y_true = pd.Series(ytest[:-73]).drop([55, 59], axis=0)
    y_pred = pd.Series(predictions[:-73]).drop([55, 59], axis=0)

    increase = np.where(df.iloc[365:]["next_prevalence"] < df.iloc[365:]["GAM Prevalence"], 0, 1)
    predicted_increase = np.where(predictions < df.iloc[365:]["GAM Prevalence"], 0, 1)
    acc = accuracy_score(increase, predicted_increase)
    MAE = mean_absolute_error(y_true, y_pred)
    scores = calculate_results(y_true=increase, y_pred=predicted_increase)

    make_confusion_matrix(y_true=increase, y_pred=predicted_increase, path=output_path,
                          name="confusion_matrix_hgbr_semiyearly.png", classes=["Increase", "Decrease"],
                          title="Confusion Matrix for Histogram Gradient Boosting Model", figsize=(8, 8))

    # Print model parameters
    print(
        '\tno. of leaves: ' + str(best_model_trees) + '\tmax_depth: ' + str(best_model_depth) + '\n\tcolumns: ' + str(
            best_model_columns))

    # Print model scores
    print(f"\tMAE: {np.round(MAE,4)}, Accuracy: {np.round(acc,4)*100}%, ",
          f"Precision: {np.round(scores['precision'],4)}, Recall: {np.round(scores['recall'],4)}, "
          f"F1 score: {np.round(scores['f1'],4)}")

    print(f"Finished running the HGBR model. ({round((time.time() - start_time), 2)}s)\n")

def hgbr_semiyearly_crop(data_path, df_csv_name, validation_hgbr, model_hgbr, output_path):
    """
    This function run the model from the baseline for all districts
    :param data_path: path to the dataframe
    :param df_csv_name: name of csv
    :param start_time: start time of the function
    :param validation_hgbr: attribute for whether cross-validation runs or not
    :return: prints the results of the model
    """
    start_time = time.time()
    print("Running HGBR model on semiyearly (unimputed) data including crops...")

    df = pd.read_csv(data_path + df_csv_name).iloc[:,2:].drop(["Average of centx","Average of centy"],axis=1)
    y = df.next_prevalence.dropna()
    X = df.select_dtypes(exclude=["object", "category"]).iloc[:len(y)].drop(
        ["MAM", "next_prevalence", "increase_numeric", "GAM Prevalence", "Under-Five Population"], axis=1)

    num_trees_min = 31
    num_trees_max = 64
    depth_min = 2
    depth_max = 7

    if validation_hgbr == 1:
        print("\tRunning the cross-validation...")
        start_cv = time.time()
        parameter_scores = []

        for num_trees in tqdm(range(num_trees_min, num_trees_max)):

            for depth in range(depth_min, depth_max):

                # Investigate every subset of explanatory variables
                for features in subsets(X.columns):
                    # First CV split. The 219 refers to the first 3 observations for the 73 districts in the data.
                    Xtrain = X[:222][features].copy().values
                    ytrain = y[:222]
                    Xtest = X[222:294][features].copy().values
                    ytest = y[222:294]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = HistGradientBoostingRegressor(max_leaf_nodes=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE1 = mean_absolute_error(ytest, predictions)

                    # Second CV split. The 292 refers to the first 4 observations for the 73 districts in the data.
                    Xtrain = X[:292][features].copy().values
                    ytrain = y[:292]
                    Xtest = X[292:367][features].copy().values
                    ytest = y[292:367]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = HistGradientBoostingRegressor(max_leaf_nodes=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE2 = mean_absolute_error(ytest, predictions)

                    # Calculate the mean MAE over the two folds
                    mean_MAE = (MAE1 + MAE2) / 2

                    # Store the mean MAE together with the used hyperparameters in list
                    parameter_scores.append((mean_MAE, num_trees, depth, features))

        # Sort the models based on score and retrieve the hyperparameters of the best model
        parameter_scores.sort(key=lambda x: x[0])
        best_model_score = parameter_scores[0][0]
        best_model_trees = parameter_scores[0][1]
        best_model_depth = parameter_scores[0][2]
        best_model_columns = list(parameter_scores[0][3])
        print(f"\tCross-validation is finished. ({round((time.time() - start_cv), 2)}s)")
    else:
        print("\tCross-validation is skipped. The best model is assigned manually.")
        best_model_columns = ['SAM', 'SAM Prevalence', 'phase3plus_perc', 'rainfall', 'ndvi_score', 'Price of water',
                  'Total alarms', 'n_conflict_total', 'prevalence_6lag', 'month', 'district_encoded', 'Cowpea', 'Maize']
        best_model_trees = 31
        best_model_depth = 2

    y = df['next_prevalence'].values
    X = df[best_model_columns].values

    # If there is only one explanatory variable, the values need to be reshaped for the model
    if len(best_model_columns) == 1:
        X = X.reshape(-1, 1)

    # Peform evaluation on full data
    Xtrain = X[:367]
    ytrain = y[:367]
    Xtest = X[367:]
    ytest = y[367:]

    if model_hgbr == 0:
        print("\tWe do not load the model.")
        clf = HistGradientBoostingRegressor(max_leaf_nodes=best_model_trees, max_depth=best_model_depth, random_state=0,
                                            verbose=0)
        clf.fit(Xtrain, ytrain)
    else:
        print("\tWe load the best HGBR model.")
        clf = joblib.load(data_path + "best_model_hgbr_crop.joblib")
    predictions = clf.predict(Xtest)

    y_true = pd.Series(ytest[:-74]).drop([53, 57], axis=0)
    y_pred = pd.Series(predictions[:-74]).drop([53, 57], axis=0)

    increase = np.where(df.iloc[367:]["next_prevalence"] < df.iloc[367:]["GAM Prevalence"], 0, 1)
    predicted_increase = np.where(predictions < df.iloc[367:]["GAM Prevalence"], 0, 1)
    acc = accuracy_score(increase, predicted_increase)
    MAE = mean_absolute_error(y_true, y_pred)
    scores = calculate_results(y_true=increase, y_pred=predicted_increase)

    make_confusion_matrix(y_true=increase, y_pred=predicted_increase, path=output_path,
                          name="confusion_matrix_hgbr_semiyearly_crop.png", classes=["Increase", "Decrease"],
                          title="Confusion Matrix for Histogram Gradient Boosting Model (with crop)", figsize=(8, 8))

    # Print model parameters
    print('\tno. of leaves: ' + str(best_model_trees) + '\tmax_depth: ' + str(best_model_depth) + '\n\tcolumns: ' +
        str(best_model_columns))

    # Print model scores
    print(f"\tMAE: {np.round(MAE,4)}, Accuracy: {np.round(acc,4)*100}%, ",
          f"Precision: {np.round(scores['precision'],4)}, Recall: {np.round(scores['recall'],4)}, "
          f"F1 score: {np.round(scores['f1'],4)}")

    print(f"Finished running the HGBR model (with crop). ({round((time.time() - start_time), 2)}s)\n")
