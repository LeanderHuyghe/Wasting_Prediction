import time
start_time = time.time()

import pathlib
import cleaning_districts as clean
import aggregated_data as aggreg
import summary_missing_values as summary
import imputation_running as imputing
import imputation_evaluation as eval
import baseline_model as model
import hgbr

validation_baseline = 0
validation_hgbr = 0
validation_hgbr_crop = 0
model_baseline = 1
model_hgbr = 0
model_hgbr_crop = 0

name_aggregated = 'semiyearly_chosen_columns.csv'
name_aggregated_crop = 'semiyearly_chosen_columns_with_crop.csv'
name_summary = 'summary_missing.txt'
name_evaluation = 'evaluation_imputations.txt'
name_imputations = 'imputed_semiyearly.csv'

current_path = str(pathlib.Path(__file__).parent.resolve())
project_path = current_path[:-4]
initial_data = project_path + "\\data_initial\\"
data_path = project_path + "\\data\\"
output_path = project_path + "\\output\\"

clean.make_clean_district_data(initial_data, data_path, time.time())
aggreg.make_aggrgated_csv(data_path, name_aggregated, time.time())
summary.writing_summary(data_path + name_aggregated, output_path + name_summary, time.time())
eval.evaluating_imputations(data_path + name_aggregated, output_path + name_evaluation, time.time())
imputing.imputations_and_visulisations(data_path, name_aggregated, output_path, name_imputations, time.time())
model.run_baseline_model(data_path, name_imputations, time.time(), validation_baseline, model_baseline)
hgbr.hgbr_semiyearly(data_path, name_aggregated, time.time(), validation_hgbr, model_hgbr, output_path)
hgbr.hgbr_semiyearly_crop(data_path, name_aggregated_crop, time.time(), validation_hgbr, model_hgbr, output_path)


print("--- %s seconds ---" % (time.time() - start_time))
