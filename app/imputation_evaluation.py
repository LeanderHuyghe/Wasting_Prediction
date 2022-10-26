from helper_metrics import impute_score, spline_conflicts, distric_wise_KNN
import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")


def evaluating_imputations(path_df, path_out):
    """
    This functions runs the evaluations for the imputation methods and outputs the results in a txt file
    :param path_df: path to the dataframe
    :param path_out: path to the output file
    :return: doesn't return anything
    """
    start_time = time.time()
    print("Evaluating the imputation methods ...")

    df = pd.read_csv(path_df, parse_dates=['date']).drop('Unnamed: 0', axis=1)

    evaluation = open(path_out, 'w')

    evaluation_metrics = ['mean', 'median', 'knn', 'mice']

    evaluation.write(f'Evaluation of different imputation methods for PRICE OF WATER\n')
    for method in evaluation_metrics:
        evaluation.write(f"{impute_score(df,features='Price of water', method=method, scale='2-100')}\n")

    evaluation.write(f'\nEvaluation of different imputation methods for NDVI\n')
    for method in evaluation_metrics:
        evaluation.write(f"{impute_score(df, features='ndvi_score', method=method, scale='0 - 0.61')}\n")

    evaluation.write(f'\nEvaluation of different imputation methods for IPC\n')
    for method in evaluation_metrics:
        evaluation.write(f"{impute_score(df, features='phase3plus_perc', method=method, scale='0 - 0.58')}\n")

    evaluation.write(f'\nEvaluation of different imputation methods for CONFLICTS\n')
    for method in evaluation_metrics:
        evaluation.write(f"{impute_score(df, features='n_conflict_total', method=method, scale='1-8')}\n")

    evaluation.write(f'SPLINE interpolation n_conflict_total\n')
    nonempty_conflict_districts = []
    for i in range(74):
        name = df.district[i]
        data = df[df.district == name]['n_conflict_total']
        if data.isna().sum().sum() == 0:
            nonempty_conflict_districts.append(i)
    rmse_scores = []
    evaluation_spline_conflicts = ''
    for i in nonempty_conflict_districts:
        name = df.district[i]
        rmse = spline_conflicts(i, df)
        rmse_scores.append(rmse)
        evaluation_spline_conflicts += f"RMSE for SPLINE imputation in {name} is: {rmse}\n"
    evaluation.write(f'Average RMSE for district-wise spline interpolation for conflicts: {round(np.average(rmse_scores),3)}\n')

    evaluation.write(f'\nTest district-wise KNN imputation\n')
    rmse_scores_knn = []
    evaluation_knn = ''
    for i in nonempty_conflict_districts:
        name = df.district[i]
        rmse = distric_wise_KNN(i, df)
        rmse_scores_knn.append(rmse)
        evaluation_knn += f"RMSE for SPLINE imputation in {name} is: {rmse}\n"
    evaluation.write(
        f'Average RMSE for district-wise KNN imputation for conflicts: {round(np.average(rmse_scores_knn),3)}\n')

    evaluation.write(f'\nSPLINE interpolation n_conflict_total pe district\n')
    evaluation.write(f'{evaluation_spline_conflicts}\n')
    evaluation.write(f'Test district-wise KNN imputation pe district\n')
    evaluation.write(f'{evaluation_knn}\n')

    print(f"The imputation evaluation was written in a text file and saved in the output folder. ",
          f"({round((time.time() - start_time), 2)}s)\n")