import pandas as pd
import numpy as np
import time

districts = ['Adan Yabaal', 'Afgooye', 'Afmadow', 'Baardheere', 'Badhaadhe', 'Baidoa',
 'Baydhaba/Bardaale', 'Baki', 'Balcad', 'Banadir', 'Bandarbeyla', 'Baraawe', 'Belet Weyne', 'Belet Weyne (Mataban)','Belet Xaawo',
'Berbera', 'Borama', 'Bossaso', 'Bu\'aale', 'Bulo Burto', 'Burco', 'Burtinle', 'Buuhoodle',
'Buur Hakaba', 'Cabudwaaq', 'Cadaado', 'Cadale', 'Caluula', 'Caynabo', 'Ceel Afweyn', 'Ceel Barde',
'Ceel Buur', 'Ceel Dheer', 'Ceel Waaq', 'Ceerigaabo', 'Dhuusamarreeb', 'Diinsoor', 'Doolow',
'Eyl', 'Gaalkacyo', 'Galdogob', 'Garbahaarey', 'Garoowe', 'Gebiley', 'Hargeysa', 'Hobyo', 'Iskushuban',
'Jalalaqsi', 'Jamaame', 'Jariiban', 'Jilib', 'Jowhar', 'Kismaayo', 'Kurtunwaarey', 'Laas Caanood', 'Laasqoray',
'Laasqoray/Badhan', 'Badhan', 'Lughaye', 'Luuq', 'Marka', 'Owdweyne', 'Qandala', 'Qansax Dheere', 'Qardho', 'Qoryooley',
'Rab Dhuure', 'Saakow', 'Saakow/Salagle', 'Sablaale', 'Sheikh', 'Taleex', 'Tayeeglow', 'Waajid', 'Wanla Weyn',
'Xarardheere', 'Xudun', 'Xudur', 'Zeylac']


def levenshteinDistanceDP(token1, token2):
    """
    This function implements the levenshtein text similarity measure &
    returns a numeric value representing the distance between two words
    :param token1: first word
    :param token2: second word
    :return: distance between tokens
    """
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def update_districts(df):
    """
    This function checks whether the district name is the standard.
    If it is not, then the word is corrected by known value or by levenshtein distance.
    It creates a list with all the standard districts and sets that list as district column.
    Uncomment print statements in last loop to see what districts changed by levenshtein algo.
    :param df: df on which we apply the checks and corrections
    :return: corrected df
    """

    new_series = []
    for token1 in df['district']:

        #If district is standard append to list
        if token1 in districts:
            new_series.append(token1)

        #If district is not standard and a known variant
        elif token1 == 'Mogadishu':
            correct_district = 'Banadir'
            new_series.append(correct_district)
        elif token1 == 'Baydhaba':
            correct_district = 'Baidoa'
            new_series.append(correct_district)
        elif token1 == 'Belethawa':
            correct_district = 'Belet Xaawo'
            new_series.append(correct_district)
        elif token1 == 'Abudwak':
            correct_district = 'Cabudwaaq'
            new_series.append(correct_district)
        elif token1 == 'Adado':
            correct_district = 'Cadaado'
            new_series.append(correct_district)
        elif token1 == 'El berde':
            correct_district = 'Ceel Barde'
            new_series.append(correct_district)

        #If district is not a known variant, apply levenshtein algo
        elif token1 not in districts:
            #print('old: %s' % token1)
            distances = []
            for token2 in districts:
                distances.append(levenshteinDistanceDP(token1, token2))
            min_value = min(distances)
            correct_district = districts[distances.index(min_value)]
            #print('new: %s' % correct_district)
            new_series.append(correct_district)
    df['district'] = new_series
    return df

def district_name_cleaning (in_path, out_path):
    """
    This function calls the necessary functions to get from teh path of an uncleaned csv to a clean df
    :param in_path: the path to read the csv
    :param out_path: the path for the output csv
    :return: store the new csv in out_path
    """

    df = pd.read_csv(in_path)
    if in_path[-7:] == 'ipc.csv':
        df.rename({'area': 'district'}, axis=1, inplace=True)
    if in_path[-13:] == 'locations.csv':
        df = df[df.district != 'Grand Total']
    df = update_districts(df)
    df.to_csv(out_path)

def make_clean_district_data (initial_data, data_path):
    """
    This function loads the necessary csvs and calls the necessary functions in order to clean the district names and
    then store th enew csvs in the data folder
    :param initial_data: the path to the folder to read the csv
    :param data_path: the path for the output folder
    :return: stores the df in the output paths
    """
    start_time = time.time()
    print("Cleaning district names ...")

    district_name_cleaning(initial_data + 'admissions.csv', data_path + 'admissions.csv')
    district_name_cleaning(initial_data + 'FSNAU_riskfactors.csv', data_path + 'FSNAU_riskfactors.csv')
    district_name_cleaning(initial_data + 'ipc2.csv', data_path + 'ipc2.csv')
    district_name_cleaning(initial_data + 'prevalence_estimates.csv', data_path + 'prevalence_estimates.csv')
    district_name_cleaning(initial_data + 'conflict.csv', data_path + 'conflict.csv')
    district_name_cleaning(initial_data + 'locations.csv', data_path + 'locations.csv')
    district_name_cleaning(initial_data + 'ipc.csv', data_path + 'ipc.csv')
    district_name_cleaning(initial_data + 'production.csv', data_path + 'production.csv')

    print(f"District names were cleaned and the new CSVs are stored in the data folder. ",
          f"({round((time.time() - start_time),2)}s)\n")