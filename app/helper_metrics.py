import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler


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
        raise ValueError(f'Method argument requires one of \'mean\',\'median\',\'knn\',\'mice\'. '
                         f'\n {method} is not a valid strategy.')

    # Create copy of dataframe and only include continuous features
    df_test = df.copy()
    df_test = df_test.select_dtypes(exclude=["category", "object"])
    df_test = df_test.dropna(axis=0)

    # Scale the dataframe
    scaler = MinMaxScaler()
    df_test_1 = df_test.copy()
    df_test_1 = df_test_1.drop(['date'], axis=1)
    df_test_scaled = pd.DataFrame(scaler.fit_transform(df_test_1), columns=df_test_1.columns)

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
    return f"RMSE for {method.upper()} imputation in {features[0]}: {rmse} \nSCALE: {scale}"

def spline_conflicts(i, df):
    """
    This function runs the evaluation for the spline interpolation for conflicts
    :param i: district we run it for
    :param df: the dataframe we run it on
    :return: rmse score
    """

    name = df.district[i]
    df_test = df[df.district == name].drop(
        ['date', 'district', 'Average of centy', 'Average of centx', 'Price of water'], axis=1)

    # Scale the dataframe
    scaler = MinMaxScaler()
    df_test_scaled = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

    # Setting Feature column
    features = ['n_conflict_total']

    # Set seed for reproducibility
    np.random.seed(18)

    #  Inserting NaN values into Experiment Group
    for col in df_test_scaled[features]:
        # 20% of the data will be removed (frac = 0.2)
        # Rows may be selected more that once (replace = true) - only useful if you have more than one column in features
        df_test_scaled.loc[df_test_scaled.sample(frac=0.2, replace=True).index, col] = np.nan

    # Creating a list of indices
    nan_cols = df_test_scaled[features]
    nan_cols = nan_cols[nan_cols.isna().any(axis=1)]
    null_idx = list(nan_cols.index)

    # Creating Answer key to compare future results against
    answer_key = df_test.iloc[null_idx]

    # Interpolate and fill any missing values with a reasonable estimate
    df_imputed = df_test_scaled.interpolate("spline", order=1).bfill()

    # Get the imputed values
    test = df_imputed.iloc[null_idx]

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
    return rmse

def distric_wise_KNN(i, df):
    """
    This function runs the evaluation for the splin einterpolation for conflicts
    :param i: district we run it for
    :param df: the dataframe we run it on
    :return: rmse score
    """
    name = df.district[i]
    df_test = df[df.district == name].drop(
        ['date', 'district', 'Average of centy', 'Average of centx', 'Price of water'], axis=1)

    # Scale the dataframe
    scaler = MinMaxScaler()
    df_test_scaled = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

    # Setting Feature column
    features = ['n_conflict_total']

    # Set seed for reproducibility
    np.random.seed(18)

    #  Inserting NaN values into Experiment Group
    for col in df_test_scaled[features]:
        # 20% of the data will be removed (frac = 0.2)
        # Rows may be selected more that once (replace = true) - only useful if you have more than one column in features
        df_test_scaled.loc[df_test_scaled.sample(frac=0.2, replace=True).index, col] = np.nan

    # Creating a list of indices
    nan_cols = df_test_scaled[features]
    nan_cols = nan_cols[nan_cols.isna().any(axis=1)]
    null_idx = list(nan_cols.index)

    # Creating Answer key to compare future results against
    answer_key = df_test.iloc[null_idx]

    # Interpolate and fill any missing values with a reasonable estimate
    # Impute
    imputer = KNNImputer(n_neighbors=5)
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
    return rmse