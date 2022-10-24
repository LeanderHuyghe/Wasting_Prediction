import time
start_time = time.time()

import pathlib
import cleaning_districts as clean
import aggregated_data as aggreg
import summary_missing_values as summary
import imputation_running as imputing
import imputation_evaluation as eval
import baseline_model as model

run_validation = 0
load_model = 1

name_aggregated = 'semiyearly_chosen_columns.csv'
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
model.run_baseline_model(data_path, name_imputations, time.time(), run_validation, load_model)

print("--- %s seconds ---" % (time.time() - start_time))
