import time
start_time = time.time()

import pathlib
from baseline_model import run_baseline_model, run_baseline_model_crop
from imputation_running import imputations_and_visulisations
from imputation_evaluation import evaluating_imputations
from cleaning_districts import make_clean_district_data
from hgbr import hgbr_semiyearly, hgbr_semiyearly_crop
from summary_missing_values import writing_summary
from aggregated_data import make_aggrgated_csv
from crop import make_crop_csvs

validation_baseline = 0
validation_baseline_crop = 0
validation_hgbr = 0
validation_hgbr_crop = 0
model_baseline = 1
model_baseline_crop = 1
model_hgbr = 1
model_hgbr_crop = 1
run_imp_crop = 0

name_aggregated = 'semiyearly_chosen_columns.csv'
name_aggregated_crop = 'semiyearly_chosen_columns_with_crop.csv'
name_imputations = 'imputed_semiyearly.csv'
name_imputations_crop = 'imputed_semiyearly_with_crop.csv'
name_summary = 'summary_missing.txt'
name_evaluation = 'evaluation_imputations.txt'

current_path = str(pathlib.Path(__file__).parent.resolve())
project_path = current_path[:-4]
initial_data = project_path + "\\data_initial\\"
data_path = project_path + "\\data\\"
output_path = project_path + "\\output\\"

make_clean_district_data(initial_data, data_path)
make_aggrgated_csv(data_path, name_aggregated)
writing_summary(data_path + name_aggregated, output_path + name_summary)
evaluating_imputations(data_path + name_aggregated, output_path + name_evaluation)
imputations_and_visulisations(data_path, name_aggregated, output_path, name_imputations)
make_crop_csvs(data_path, name_imputations, name_aggregated, name_aggregated_crop, name_imputations_crop, run_imp_crop)
run_baseline_model(data_path, name_imputations, validation_baseline, model_baseline, output_path)
run_baseline_model_crop(data_path, name_imputations_crop, validation_baseline_crop, model_baseline_crop, output_path)
hgbr_semiyearly(data_path, name_aggregated, validation_hgbr, model_hgbr, output_path)
hgbr_semiyearly_crop(data_path, name_aggregated_crop, validation_hgbr_crop, model_hgbr_crop, output_path)


print("--- %s seconds ---" % (time.time() - start_time))
