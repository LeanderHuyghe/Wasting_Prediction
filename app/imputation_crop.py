import datawig
from datawig.utils import random_split 
import pandas as pd
import numpy as np
from datawig import SimpleImputer

def imputing_crop(path, name_initial, name_output):
    """
    The function imputes data for the missing values for production.
    :param path: the path to the data folder
    :param name_initial: name of the initial csv
    :param name_output: name of the output csv
    """

    df = pd.read_csv(path + name_initial)
    predictions_num = df.copy()

    # Hide empty cells
    columns = ['Cowpea']
    df_not_null = df[df[columns].notnull().all(1)]
    # Create training/test sets
    df_train, df_test = random_split(df_not_null, split_ratios=[0.8, 0.2])
    hide_proportion = 0.25
    df_test_missing = df_test.mask(np.random.rand(*df_test.shape) > (1 - hide_proportion))
    # Define columns with useful info for to-be-imputed column
    input_cols = ['date', 'phase3plus_perc', 'GAM', 'MAM', 'SAM', 'n_conflict_total', 'next_prevalence',
                  'prevalence_6lag', 'ndvi_score', 'total population', 'district_encoded', 'Price of water', 'Sorghum',
                  'Maize']
    # Define columns to be imputed
    output_col_num = 'Cowpea'  # Numerical output col
    # Initialize imputer for numerical imputation
    imputer_num = SimpleImputer(
        input_columns=input_cols,
        output_column=output_col_num,  # Column to be imputed
        output_path='../artifacts/imputer_model_num')
    # Numerical imputation model fit
    imputer_num.fit_hpo(train_df=df_train,
                        learning_rate_candidates=[1e-2, 1e-3, 1e-4, 1e-5],
                        numeric_latent_dim_candidates=[10, 20, 50, 100],
                        numeric_hidden_layers_candidates=[0, 1, 2],
                        final_fc_hidden_units=[[100], [150]])
    # Predict missing values
    predictions_num1 = imputer_num.predict(df)
    # Replace negative values by 0
    predictions_num1['Cowpea_imputed'][predictions_num1['Cowpea_imputed'] < 0] = 0
    # Move predicted values to empty cells of Cowpea Column
    predictions_num1.Cowpea.fillna(predictions_num1.Cowpea_imputed, inplace=True)
    # Add Full column to final dataset
    predictions_num['Cowpea'] = predictions_num1['Cowpea']


    # Hide empty cells
    columns = ['Sorghum']
    df_not_null = df[df[columns].notnull().all(1)]
    # Create training/test sets
    df_train, df_test = random_split(df_not_null, split_ratios=[0.8, 0.2])
    hide_proportion = 0.25
    df_test_missing = df_test.mask(np.random.rand(*df_test.shape) > (1 - hide_proportion))
    # Define columns with useful info for to-be-imputed column
    input_cols = ['date', 'phase3plus_perc', 'GAM', 'MAM', 'SAM', 'n_conflict_total', 'next_prevalence',
                  'prevalence_6lag', 'ndvi_score', 'total population', 'district_encoded', 'Price of water', 'Maize',
                  'Cowpea']
    # Define columns to be imputed
    output_col_num = 'Sorghum'  # Numerical output col
    # Initialize imputer for numerical imputation
    imputer_num = SimpleImputer(
        input_columns=input_cols,
        output_column=output_col_num,  # Column to be imputed
        output_path='../artifacts/imputer_model_num')
    ## Numerical imputation model fit
    imputer_num.fit_hpo(train_df=df_train,
                        learning_rate_candidates=[1e-2, 1e-3, 1e-4, 1e-5],
                        numeric_latent_dim_candidates=[10, 20, 50, 100],
                        numeric_hidden_layers_candidates=[0, 1, 2],
                        final_fc_hidden_units=[[100], [150]] )
    # Predict missing values
    predictions_num1 = imputer_num.predict(df)
    # Replace negative values by 0
    predictions_num1['Sorghum_imputed'][predictions_num1['Sorghum_imputed'] < 0] = 0
    # Move predicted values to empty cells of Cowpea Column
    predictions_num1.Sorghum.fillna(predictions_num1.Sorghum_imputed, inplace=True)
    # Add Full column to final dataset
    predictions_num['Sorghum'] = predictions_num1['Sorghum']


    # Hide empty cells
    columns = ['Maize']
    df_not_null = df[df[columns].notnull().all(1)]
    # Create training/test sets
    df_train, df_test = random_split(df_not_null, split_ratios=[0.8, 0.2])
    hide_proportion = 0.25
    df_test_missing = df_test.mask(np.random.rand(*df_test.shape) > (1 - hide_proportion))
    # Define columns with useful info for to-be-imputed column
    input_cols = ['date', 'phase3plus_perc', 'GAM', 'MAM', 'SAM', 'n_conflict_total', 'next_prevalence',
                  'prevalence_6lag', 'ndvi_score', 'total population', 'district_encoded', 'Price of water', 'Sorghum',
                  'Cowpea']
    # Define columns to be imputed
    output_col_num = 'Maize'  # Numerical output col
    # Initialize imputer for numerical imputation
    imputer_num = SimpleImputer(
        input_columns=input_cols,
        output_column=output_col_num,  # Column to be imputed
        output_path='../artifacts/imputer_model_num')
    ## Numerical imputation model fit
    imputer_num.fit_hpo(train_df=df_train,
                        learning_rate_candidates=[1e-2, 1e-3, 1e-4, 1e-5],
                        numeric_latent_dim_candidates=[10, 20, 50, 100],
                        numeric_hidden_layers_candidates=[0, 1, 2],
                        final_fc_hidden_units=[[100], [150]])
    # Predict missing values
    predictions_num1 = imputer_num.predict(df)
    # Replace negative values by 0
    predictions_num1['Maize_imputed'][predictions_num1['Maize_imputed'] < 0] = 0
    # Move predicted values to empty cells of Cowpea Column
    predictions_num1.Maize.fillna(predictions_num1.Maize_imputed, inplace=True)
    # Add Full column to final dataset
    predictions_num['Maize'] = predictions_num1['Maize']

    predictions_num.to_csv(path + name_output)
    return predictions_num