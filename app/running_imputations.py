import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import create_plots as plots


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

    df['n_conflict_total_KNN'] = df['n_conflict_total']
    df['n_conflict_total_spline'] = df['n_conflict_total']

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
        df.loc[index, 'n_conflict_total_KNN'] = conflict

        # retrieve district name
        name = df.district.unique()[i]
        # retrieve conflict data spline
        data2 = df[df.district == name]['n_conflict_total']
        # retrieve indices of the missing values spline
        index_spline = data2[data2.isna()].index.tolist()
        # impute and fill any missing values with a reasonable estimate spline
        data_spline = data2.interpolate("spline", order=1).bfill()
        # retrieve interpolated values at the required indices spline
        conflict = data_spline[index_spline].values.tolist()
        # change value at the indexed location spline
        df.loc[index, 'n_conflict_total_spline'] = conflict


    # Redefine subset X for the rest of the imputations
    X = df.drop(['date', 'district', 'Average of centy', 'Average of centx'], axis=1)
    knn_df = pd.DataFrame(knn_imputer.fit_transform(X), columns=X.columns)
    ndvi_score = knn_df["ndvi_score"]
    ipc = knn_df["phase3plus_perc"]
    price_of_water_KNN = knn_df["Price of water"]

    # MICE imputation
    mice_imputer = IterativeImputer(n_nearest_features=5, max_iter=100).fit_transform(X)
    price_of_water_MICE = pd.DataFrame(mice_imputer, columns=X.columns)["Price of water"]

    # Change columns to imputed features
    df["ndvi_score"] = ndvi_score
    df["phase3plus_perc"] = ipc
    df["price_of_water_KNN"] = price_of_water_KNN
    df["price_of_water_MICE"] = price_of_water_MICE

    # Rename and dropped unwanted features
    df = df.rename(columns={"ndvi_score": "ndvi", "phase3plus_perc": "ipc", "n_conflict_total_KNN": "conflicts_KNN",
                            "n_conflict_total_spline": "conflicts_spline"})
    df = df.drop(["Average of centx", "Average of centy", "Price of water", "n_conflict_total"], axis=1)

    return df



def running_all_imputations(data_path, df_csv_name, outputs_path, new_df_csv_name):
    """
    This function initializez a dataframe and runs all the imputations and creates the plots
    :param data_path: path to folder of data
    :param df_csv_name: name of csv that we read as df
    :param outputs_path: path to outputs folder
    :return: nothing, instead it stores the new csv with imputed columns
    """

    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop('Unnamed: 0', axis=1)

    plots.correlation_heatmap_missing(df, outputs_path)
    plots.plot_original_conflict_districts(df, outputs_path)
    plots.plot_original_price_of_water_MICE(df, outputs_path)

    df = impute_values(df)

    plots.plot_imputed_conflict_districts_spline(df, outputs_path)
    plots.plot_imputed_conflict_districts_knn(df, outputs_path)
    plots.plot_imputed_price_of_water_MICE(df, outputs_path)
    plots.plot_imputed_price_of_water_knn(df, outputs_path)
    plots.plot_correlation(df, outputs_path)

    df.to_csv(data_path + new_df_csv_name)