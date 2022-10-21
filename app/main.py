import time
start_time = time.time()

import pathlib
import cleaning_districts as clean
import aggregated_data as aggreg
import summary_missing_values as summary
import imputation_running as imputing
import imputation_evaluation as evaluation
import baseline_model as model

current_path = str(pathlib.Path(__file__).parent.resolve())
project_path = current_path[:-4]
initial_data = project_path + "\\data_initial\\"
data_path = project_path + "\\data\\"
output_path = project_path + "\\output\\"

clean.make_clean_district_data(initial_data,data_path)
aggreg.make_aggrgated_csv(data_path)
summary.writing_summary_missing_values(data_path + 'semiyearly_chosen_columns.csv', output_path + 'summary_missing.txt')
evaluation.evaluating_imputations(data_path + 'semiyearly_chosen_columns.csv', output_path + 'evaluation_imputations.txt')
imputing.imputations_and_visulisations(data_path, 'semiyearly_chosen_columns.csv', output_path, 'imputed_semiyearly.csv')

model.run_baseline_model(data_path, 'imputed_semiyearly.csv')

print("--- %s seconds ---" % (time.time() - start_time))