import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

num_trees_min = 64
num_trees_max = 128
depth_min = 2
depth_max = 7

#Function that returns every possible subset (except the empty set) of the input list l
def subsets (l):
    subset_list = []
    for i in range(len(l) + 1):
        for j in range(i):
            subset_list.append(l[j: i])
    return subset_list

def run_baseline_model(data_path, df_csv_name):
    """
    This function run sthe model from the baseline for all districts
    :param data_path: path to the dateframe
    :param df_csv_name: name of csv
    :return: prints the normal data
    """
    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop('Unnamed: 0', axis=1)
    df.dropna(inplace=True)

    '''------------SECTION RANDOM FOREST CROSS VALIDATION--------------'''
    # WARNING: this process can take some time, since there are a lot of hyperparameters to investigate. The search space can be manually reduced to speed up the process.

    # Create empty list to store model scores
    parameter_scores = []

    # Define target and explanatory variables
    X = df.drop(columns=['increase', 'increase_numeric', 'date', 'district', 'GAM Prevalence',
                         'next_prevalence'])  # Note that these columns are dropped, the remaining columns are used as explanatory variables
    y = df['next_prevalence'].values

    for num_trees in range(num_trees_min, num_trees_max):

        for depth in range(depth_min, depth_max):

            # Investigate every subset of explanatory variables
            for features in subsets(X.columns):
                # First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
                Xtrain = X[:99][features].copy().values
                ytrain = y[:99]
                Xtest = X[99:132][features].copy().values
                ytest = y[99:132]

                # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

                # Fit to the training data
                clf.fit(Xtrain, ytrain)

                # Make a prediction on the test data
                predictions = clf.predict(Xtest)

                # Calculate mean absolute error
                MAE1 = mean_absolute_error(ytest, predictions)

                # Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
                Xtrain = X[:132][features].copy().values
                ytrain = y[:132]
                Xtest = X[132:165][features].copy().values
                ytest = y[132:165]

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

    print(best_model_columns)

    '''------------SECTION FINAL EVALUATION--------------'''
    X = df[best_model_columns].values
    y = df['next_prevalence'].values

    # If there is only one explanatory variable, the values need to be reshaped for the model
    if len(best_model_columns) == 1:
        X = X.reshape(-1, 1)

    # Peform evaluation on full data
    Xtrain = X[:165]
    ytrain = y[:165]
    Xtest = X[165:]
    ytest = y[165:]

    clf = RandomForestRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, random_state=0)
    clf.fit(Xtrain, ytrain)
    predictions = clf.predict(Xtest)

    # Calculate MAE
    MAE = mean_absolute_error(ytest, predictions)

    # Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current prevalence, 1 otherwise.
    increase = [0 if x < y else 1 for x in df.iloc[165:]['next_prevalence'] for y in df.iloc[165:]['GAM Prevalence']]
    predicted_increase = [0 if x < y else 1 for x in predictions for y in df.iloc[165:]['GAM Prevalence']]

    # Calculate accuracy of predicted boolean increase/decrease
    acc = accuracy_score(increase, predicted_increase)

    # Print model parameters
    print('no. of trees: ' + str(best_model_trees) + '\nmax_depth: ' + str(best_model_depth) + '\ncolumns: ' + str(
        best_model_columns))

    # Print model scores
    print(MAE, acc)

    joblib.dump(clf, data_path + "model_74.joblib")

# loaded_model = joblib.load(filename)