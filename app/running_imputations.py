import pandas as pd
import imputation_functions as imp
import create_plots as plots

def running_all_imputations(data_path, df_csv_name, outputs_path, new_df_csv_name):
    """
    This function initializez a dataframe and runs all the imputations and creates the plots
    :param data_path: path to folder of data
    :param df_csv_name: name of csv that we read as df
    :param outputs_path: path to outputs folder
    :return: nothing, instead it stores the new csv with imputed columns
    """

    df = pd.read_csv(data_path + df_csv_name, parse_dates=['date']).drop('Unnamed: 0', axis=1)

    plots.plot_original_conflict_districts(df, outputs_path)
    plots.plot_original_price_of_water_MICE(df, outputs_path)

    df = imp.update_MAM(df)
    df = imp.update_total_population(df)
    df = imp.update_n_conflict_total_spline(df)
    df = imp.update_price_of_water_MICE(df)
    df = imp.update_knn(df)

    plots.plot_imputed_conflict_districts_spline(df, outputs_path)
    plots.plot_imputed_conflict_districts_knn(df, outputs_path)
    plots.plot_imputed_price_of_water_MICE(df, outputs_path)
    plots.plot_imputed_price_of_water_knn(df, outputs_path)
    plots.plot_correlation(df, outputs_path)

    df.to_csv(data_path + new_df_csv_name)