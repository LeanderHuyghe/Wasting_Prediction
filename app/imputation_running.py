import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import create_plots as plots
import time


def impute_values(df):
    """
    This function runs all the imputations for a dataframe
    :param df: dataframe to run the imputations
    :return: the new dataframe
    """
    # Imputing the missing MAM
    df.MAM = df.GAM - df.SAM

    # Define a subset X that will only be used for conflict imputation
    X = df.drop(["date", "increase", "Average of centx", "Price of water","Average of centy"], axis=1).copy()

    # KNN for conflicts (district-wise imputation)
    knn_imputer = KNNImputer(n_neighbors=5)
    num_districts = len(df.district.unique())

    for i in range(num_districts):
        # retrieve district name
        name = X.district[i]
        # retrieve conflict data
        data = X[X.district == name].drop('district', axis=1)
        id = data.index
        # retrieve indices of the missing values
        index = data[data['n_conflict_total'].isna()].index.tolist()
        # impute and fill any missing values with a reasonable estimate KNN
        imputed_data = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns, index=id)['n_conflict_total']
        # retrieve interpolated values at the required indices KNN
        conflict = imputed_data[index].values.tolist()
        # change value at the indexed location KNN
        df.loc[index, 'n_conflict_total'] = conflict


    # Redefine subset X for the rest of the imputations
    X = df.drop(['date', 'district', 'Average of centy', 'Average of centx'], axis=1)
    knn_df = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)
    ndvi_score = knn_df["ndvi_score"]
    ipc = knn_df["phase3plus_perc"]

    # MICE imputation
    mice_imputer = IterativeImputer(n_nearest_features=5, max_iter=100).fit_transform(X)
    price_of_water_MICE = pd.DataFrame(mice_imputer, columns=X.columns)["Price of water"]

    # Change columns to imputed features
    df["ndvi_score"] = ndvi_score
    df["phase3plus_perc"] = ipc
    df["price_of_water"] = price_of_water_MICE

    # Rename and dropped unwanted features
    df = df.rename(columns={"ndvi_score": "ndvi", "phase3plus_perc": "ipc", "n_conflict_total": "conflicts"})
    df = df.drop(["Average of centx", "Average of centy", "Price of water", "MAM"], axis=1)

    return df


def impute_dummy(df, feature, index):
    """
    This function returns the dataframe with imputed vs actual data
    :param df: dataframe we use
    :param feature: the feature we look for
    :param index: the index of actual data
    :return: the dataframe where we know what is imputed or actual
    """
    data = df[['date','district','next_prevalence',feature]]
    data['imputed'] = np.arange(0,len(data))
    for i in range(len(data)):
        if i in index:
            data.loc[i, 'imputed'] = 'imputed'
        else:
            data.loc[i, 'imputed'] = 'actual'
    return data


def index_data(df):
    """
    This function gets the indexes of actual data for several features
    :param df: teh dataframe form where we take index
    :return: the 4 index lists
    """
    conflict_index = df[df['n_conflict_total'].isna() == True].index
    ndvi_index = df[df['ndvi_score'].isna() == True].index
    ipc_index = df[df['phase3plus_perc'].isna() == True].index
    return (conflict_index, ndvi_index, ipc_index)


def actual_vs_imputed_df(df, index_actual):
    """
    This function makes the dataframe assigns for the features the dataframes with actual vs real data
    :param df: the dataframe that we start from
    :param index_actual: the list of indices
    :return: the 4 data frames
    """
    df_ndvi = impute_dummy(df, "ndvi", index_actual[1])
    df_conflict = impute_dummy(df, "conflicts", index_actual[0])
    df_ipc = impute_dummy(df, "ipc", index_actual[2])
    return (df_ndvi, df_conflict, df_ipc)


def imputations_and_visulisations(data_path, df_csv_name, outputs_path, new_df_csv_name, start_time):
    """
    This function initializes a dataframe and runs all the imputations and creates the plots
    :param data_path: path to folder of data
    :param df_csv_name: name of csv that we read as df
    :param outputs_path: path to outputs folder
    :return: nothing, instead it stores the new csv with imputed columns
    """
    print("Imputing the missing values ...")

    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop('Unnamed: 0', axis=1)
    index_actual = index_data(df)

    plots.correlation_heatmap_missing(df, outputs_path)
    plots.plot_original_conflict_districts(df, outputs_path)
    plots.plot_original_price_of_water(df, outputs_path)
    plots.plot_bar_missing(df, outputs_path)

    df = impute_values(df)
    df_ndvi, df_conlict, df_ipc = actual_vs_imputed_df(df, index_actual)

    plots.scatter_ndvi(df_ndvi, outputs_path)
    plots.scatter_conflicts(df_conlict, outputs_path)
    plots.scatter_ipc(df_ipc, outputs_path)

    plots.plot_imputed_conflict_districts_knn(df, outputs_path)
    plots.plot_imputed_price_of_water_MICE(df, outputs_path)
    plots.plot_correlation(df, outputs_path)

    df.to_csv(data_path + new_df_csv_name)

    print(f"'{new_df_csv_name}' has the imputed missing values and the visualizations were saved in the output folder.",
          f"({round((time.time() - start_time), 2)}s)\n")