import ditrict_cleaning_functions as clean


def make_clean_district_data (initial_data, data_path):
    """
    This function loads the necessary csvs and calls the necessary functions in order to clean the district names and
    then store th enew csvs in the data folder
    :param initial_data: the path to the folder to read the csv
    :param data_path: the path for the output folder
    """

    clean.district_name_cleaning(initial_data + 'admissions.csv', data_path + 'admissions.csv')
    clean.district_name_cleaning(initial_data + 'FSNAU_riskfactors.csv', data_path + 'FSNAU_riskfactors.csv')
    clean.district_name_cleaning(initial_data + 'ipc2.csv', data_path + 'ipc2.csv')
    clean.district_name_cleaning(initial_data + 'prevalence_estimates.csv', data_path + 'prevalence_estimates.csv')
    clean.district_name_cleaning(initial_data + 'conflict.csv', data_path + 'conflict.csv')
    clean.district_name_cleaning(initial_data + 'locations.csv', data_path + 'locations.csv')
    clean.district_name_cleaning(initial_data + 'ipc.csv', data_path + 'ipc.csv')
