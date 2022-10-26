import pandas as pd
import time


def add_imputed_crop_to_imputed_semi(df_semi, df_crop, name_imp, path):
    """
    This function adds the imputed data for the crops to the imputed semiyearly and also exports it to csv
    :param df_semi: imputed semiyearly df
    :param df_crop: imputed crop df
    :param name_imp: name of output csv
    :param path: path for the export
    :return: doesn't return but saves the df to a CSV
    """
    df_imputed_semi_crop = df_semi.copy()
    df_imputed_semi_crop['Cowpea'] = df_crop['Cowpea']
    df_imputed_semi_crop['Maize'] = df_crop['Maize']
    df_imputed_semi_crop['Sorghum'] = df_crop['Sorghum']
    df_imputed_semi_crop['crop'] = df_crop['crop']
    df_imputed_semi_crop.to_csv(path + name_imp)

def add_crop_to_semi(df_semi, df_crop, name_agg, path):
    """
    This function adds the data for the crops to the semiyearly and also exports it to csv
    :param df_semi: aggregated semiyearly df
    :param df_crop: district cleaned production df
    :param name_agg: name of output csv
    :param path: path for the export
    :return: doesn't return but saves the df to a CSV
    """
    df_crop = df_crop[df_crop['date'] >= '2017-01-01']
    df_crop_all = df_crop.copy()
    df_crop_all['date'] = df_crop_all['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_crop_all['date'] = df_crop_all['date'].astype('datetime64')
    df_crop = df_crop_all[['date', 'district', 'Cowpea', 'Maize', 'Sorghum']]
    df_semi_crop = df_semi.merge(df_crop, on=['date', 'district'], how='left')
    df_semi_crop['crop'] = (df_semi_crop.Maize + df_semi_crop.Sorghum + df_semi_crop.Cowpea) / 3
    df_semi_crop.to_csv(path + name_agg)

def make_crop_csvs(path, semi_imp, semi_agg, name_agg, name_imp, run_imputations):
    """
    This function calls and makes the aggregated data, the imputed crop data and the aggregated imputed data
    :param path: path to the data folder
    :param semi_imp: the name of the semiyearly imputed data
    :param semi_agg: the name of the semiyearly aggregated data
    :return: doesn't return anything
    """
    start_time = time.time()
    print("We are creating the production CSVs...")
    df_semi_agg = pd.read_csv(path + semi_agg, parse_dates=['date'])
    df_semi_imp = pd.read_csv(path + semi_imp, parse_dates=['date'])
    df_crop = pd.read_csv(path + 'production.csv', parse_dates=['date'])

    add_crop_to_semi(df_semi_agg, df_crop, name_agg, path)

    if run_imputations == 1:
        from imputation_crop import imputing_crop
        df_crop_imp = imputing_crop(path, name_agg, 'imputed_production.csv')
    else:
        df_crop_imp = pd.read_csv(path + 'imputed_production.csv')

    add_imputed_crop_to_imputed_semi(df_semi_imp, df_crop_imp, name_imp, path)

    print(f"Finished the task sand the CSV are saved in the data folder. ({round((time.time() - start_time), 2)}s)\n")
