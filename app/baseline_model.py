from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from create_plots import make_confusion_matrix
from helper_metrics import calculate_results
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import time

num_trees_min = 64
num_trees_max = 128
depth_min = 2
depth_max = 7

def subsets(l):
    """
    This function returns every possible subset (except the empty set) of the input list l
    :param l: list for wich to return subsets
    :return subset_list: list of subsets
    """
    subset_list = []
    for i in range(len(l) + 1):
        for j in range(i):
            subset_list.append(l[j: i])
    return subset_list

def run_baseline_model(data_path, df_csv_name, atr_cv, atr_model, output_path):
    """
    This function run the model from the baseline for all districts
    :param data_path: path to the dataframe
    :param df_csv_name: name of csv
    :param atr_cv: attribute for whether cross-validation runs or not
    :param atr_model: attribute for whether we load the model or not
    :param output_path: path to the output folder
    :return: prints the results of the model
    """
    start_time = time.time()
    print("Running baseline model for imputed data...")

    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop(
        ['Unnamed: 0', 'Under-Five Population','district_encoded'],axis=1)
    df.dropna(inplace=True)

    if atr_cv == 1:
        print("\tRunning the cross-validation...")
        start_cv = time.time()

        # Create empty list to store model scores
        parameter_scores = []

        # Define target and explanatory variables
        X = df.drop(columns=['increase', 'increase_numeric', 'date', 'district', 'GAM Prevalence',
                             'next_prevalence'])
        y = df['next_prevalence'].values

        for num_trees in tqdm(range(num_trees_min, num_trees_max)):
            for depth in range(depth_min, depth_max):
                for features in subsets(X.columns):
                    # First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
                    Xtrain = X[:220][features].copy().values
                    ytrain = y[:220]
                    Xtest = X[220:293][features].copy().values
                    ytest = y[220:293]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE1 = mean_absolute_error(ytest, predictions)

                    # Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
                    Xtrain = X[:293][features].copy().values
                    ytrain = y[:293]
                    Xtest = X[293:365][features].copy().values
                    ytest = y[293:365]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

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
        best_model_trees = 107
        best_model_depth = 6
        best_model_columns = ['total population', 'GAM']


    X = df[best_model_columns].values
    y = df['next_prevalence'].values

    # If there is only one explanatory variable, the values need to be reshaped for the model
    if len(best_model_columns) == 1:
        X = X.reshape(-1, 1)

    # Peform evaluation on full data
    Xtrain = X[:365]
    ytrain = y[:365]
    Xtest = X[365:]
    ytest = y[365:]

    if atr_model == 0:
        print("\tWe make the random forest with the best model attributes.")
        clf = RandomForestRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, random_state=0)
        clf.fit(Xtrain, ytrain)
    else:
        print("\tWe load the best model.")
        clf = joblib.load(data_path + 'best_model_baseline.joblib')
    predictions = clf.predict(Xtest)

    # Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current
    # prevalence, 1 otherwise.
    increase = np.where(df.iloc[365:]['next_prevalence'] < df.iloc[365:]['GAM Prevalence'], 0, 1)
    predicted_increase = np.where(predictions < df.iloc[365:]['GAM Prevalence'], 0, 1)

    # Calculate scores
    MAE = mean_absolute_error(ytest, predictions)
    acc = accuracy_score(increase, predicted_increase)
    scores = calculate_results(y_true=increase, y_pred=predicted_increase)

    make_confusion_matrix(y_true=increase, y_pred=predicted_increase, path=output_path,
                          name="confusion_matrix_baseline_semiyearly.png", classes=["Increase", "Decrease"],
                          title="Confusion Matrix for Baseline Model", figsize=(8, 8))

    # Print model parameters
    print('\tno. of trees: ' + str(best_model_trees) + '\tmax_depth: ' + str(best_model_depth) + '\n\tcolumns: ' + str(
        best_model_columns))

    # Print model scores
    print(f"\tMAE: {np.round(MAE, 4)}, Accuracy: {np.round(acc, 4) * 100}%, ",
          f"Precision: {np.round(scores['precision'], 4)}, Recall: {np.round(scores['recall'], 4)}, "
          f"F1 score: {np.round(scores['f1'], 4)}")

    print(f"Finished running the baseline model. ({round((time.time() - start_time), 2)}s)\n")

def run_baseline_model_crop(data_path, df_csv_name, atr_cv, atr_model, output_path):
    """
    This function run the model from the baseline for the imputed data with crop
    :param data_path: path to the data folder
    :param df_csv_name: name of csv that contains the imputed combination of semiyearly and crop
    :param atr_cv: attribute for whether cross-validation runs or not
    :param atr_model: attribute for whether we load the model or not
    :param output_path: path to the output folder
    :return: prints the results of the model
    """
    start_time = time.time()
    print("Running baseline model for imputed data including crop...")

    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop(
        ['Unnamed: 0', 'Unnamed: 0.1', 'Under-Five Population', 'district_encoded'], axis=1)
    df.dropna(inplace=True)

    if atr_cv == 1:
        print("\tRunning the cross-validation...")
        start_cv = time.time()

        # Create empty list to store model scores
        parameter_scores = []

        # Define target and explanatory variables
        X = df.drop(columns=['increase', 'increase_numeric', 'date', 'district', 'GAM Prevalence',
                             'next_prevalence'])
        y = df['next_prevalence'].values


        for num_trees in tqdm(range(num_trees_min, num_trees_max)):
            for depth in range(depth_min, depth_max):
                for features in subsets(X.columns):
                    # First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
                    Xtrain = X[:220][features].copy().values
                    ytrain = y[:220]
                    Xtest = X[220:293][features].copy().values
                    ytest = y[220:293]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

                    # Fit to the training data
                    clf.fit(Xtrain, ytrain)

                    # Make a prediction on the test data
                    predictions = clf.predict(Xtest)

                    # Calculate mean absolute error
                    MAE1 = mean_absolute_error(ytest, predictions)

                    # Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
                    Xtrain = X[:293][features].copy().values
                    ytrain = y[:293]
                    Xtest = X[293:365][features].copy().values
                    ytest = y[293:365]

                    # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                    clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

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
        best_model_trees = 66
        best_model_depth = 6
        best_model_columns = ['total population', 'GAM', 'SAM', 'SAM Prevalence', 'ipc', 'rainfall', 'ndvi',
                              'Total alarms', 'conflicts', 'prevalence_6lag', 'month',
                              'price_of_water', 'Cowpea', 'Maize', 'Sorghum', 'crop']


    X = df[best_model_columns].values
    y = df['next_prevalence'].values

    # If there is only one explanatory variable, the values need to be reshaped for the model
    if len(best_model_columns) == 1:
        X = X.reshape(-1, 1)

    # Peform evaluation on full data
    Xtrain = X[:365]
    ytrain = y[:365]
    Xtest = X[365:]
    ytest = y[365:]

    if atr_model == 0:
        print("\tWe make the random forest with the best model attributes.")
        clf = RandomForestRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, random_state=0)
        clf.fit(Xtrain, ytrain)
    else:
        print("\tWe load the best model.")
        clf = joblib.load(data_path + 'best_model_baseline_crop.joblib')
    predictions = clf.predict(Xtest)

    # Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current
    # prevalence, 1 otherwise.
    increase = np.where(df.iloc[365:]['next_prevalence'] < df.iloc[365:]['GAM Prevalence'], 0, 1)
    predicted_increase = np.where(predictions < df.iloc[365:]['GAM Prevalence'], 0, 1)

    # Calculate scores
    MAE = mean_absolute_error(ytest, predictions)
    acc = accuracy_score(increase, predicted_increase)
    scores = calculate_results(y_true=increase, y_pred=predicted_increase)

    make_confusion_matrix(y_true=increase, y_pred=predicted_increase, path=output_path,
                          name="confusion_matrix_baseline_semiyearly_crop.png", classes=["Increase", "Decrease"],
                          title="Confusion Matrix for Baseline Model (with crop)", figsize=(8, 8))

    # Print model parameters
    print('\tno. of trees: ' + str(best_model_trees) + '\tmax_depth: ' + str(best_model_depth) + '\n\tcolumns: ' + str(
        best_model_columns))

    # Print model scores
    print(f"\tMAE: {np.round(MAE, 4)}, Accuracy: {np.round(acc, 4) * 100}%, ",
          f"Precision: {np.round(scores['precision'], 4)}, Recall: {np.round(scores['recall'], 4)}, "
          f"F1 score: {np.round(scores['f1'], 4)}")

    print(f"Finished running the baseline model (with crop). ({round((time.time() - start_time), 2)}s)\n")
