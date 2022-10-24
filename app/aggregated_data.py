import pandas as pd
import numpy as np
import time


def make_district_df_semiyearly(datapath, district_name):
    """
    Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder
    district_name : string
        Name of the district

    Returns
    -------
    df : pandas dataframe
    """

    # Read all relevant datasets
    df_admissions = pd.read_csv(datapath + 'admissions.csv', parse_dates=['date'])
    df_FSNAU = pd.read_csv(datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    df_locations = pd.read_csv(datapath + 'locations.csv', parse_dates=['date'])
    df_ipc = pd.read_csv(datapath + 'ipc.csv', parse_dates=['date'])
    df_ipc2 = pd.read_csv(datapath + 'ipc2.csv', parse_dates=['date'])
    df_prevalence = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    df_conflict = pd.read_csv(datapath + 'conflict.csv', parse_dates=['date'])

    # Select data for specific district
    df_admissions = df_admissions[df_admissions['district'] == district_name]
    df_FSNAU = df_FSNAU[df_FSNAU['district'] == district_name]
    df_locations = df_locations[df_locations['district'] == district_name]
    df_ipc = df_ipc[df_ipc['district'] == district_name]
    df_ipc2 = df_ipc2[df_ipc2['district'] == district_name]
    df_prevalence = df_prevalence[df_prevalence['district'] == district_name]
    df_conflict = df_conflict[df_conflict['district'] == district_name]

    df_locations = df_locations.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_locations = df_locations.reset_index()
    df_locations['date'] = df_locations['date'].apply(lambda x : x.replace(day=1))

    df_FSNAU = df_FSNAU.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_FSNAU = df_FSNAU.reset_index()
    df_FSNAU['date'] = df_FSNAU['date'].apply(lambda x : x.replace(day=1))

    df_admissions = df_admissions.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_admissions = df_admissions.reset_index()
    df_admissions['date'] = df_admissions['date'].apply(lambda x : x.replace(day=1))

    df_conflict = df_conflict.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_conflict = df_conflict.reset_index()
    df_conflict['date'] = df_conflict['date'].apply(lambda x : x.replace(day=1))

    # Sort dataframes on date
    df_admissions.sort_values('date', inplace=True)
    df_FSNAU.sort_values('date', inplace=True)
    df_locations.sort_values('date', inplace=True)
    df_ipc.sort_values('date', inplace=True)
    df_ipc2.sort_values('date', inplace=True)
    df_prevalence.sort_values('date', inplace=True)
    df_conflict.sort_values('date', inplace=True)

    # Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(left=df_prevalence, right=df_ipc2, direction='backward', on='date', suffixes=('', '_drop'))
    df = pd.merge_asof(left=df, right=df_ipc, direction='backward', on='date', suffixes=('', '_drop'))
    df = pd.merge_asof(left=df, right=df_locations, direction='backward', on='date', suffixes=('', '_drop'))
    df = pd.merge_asof(left=df, right=df_FSNAU, direction='backward', on='date', suffixes=('', '_drop'))
    df = pd.merge_asof(left=df, right=df_admissions, direction='backward', on='date', suffixes=('', '_drop'))
    df = pd.merge_asof(left=df, right=df_conflict, direction='backward', on='date', suffixes=('', '_drop'))

    # Calculate prevalence 6lag
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(1)
    df['next_prevalence'] = df['GAM Prevalence'].shift(-1)

    # Select needed columns
    columns = ['date', 'district', 'total population', 'Under-Five Population', 'GAM',
               'MAM', 'SAM', 'GAM Prevalence', 'SAM Prevalence', 'phase3plus_perc',
               'rainfall', 'ndvi_score', 'Price of water', 'Total alarms',
               'n_conflict_total', 'Average of centy', 'Average of centx',
               'prevalence_6lag', 'next_prevalence']
    df = df[columns]

    # Add month column
    df['month'] = df['date'].dt.month

    # Add target variable: increase for next month prevalence (boolean)
    increase = [False if x[1 ] <x[0] else True for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase.append(False)
    df['increase'] = increase
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan  # No info on next month

    # Add target variable: increase for next month prevalence (boolean)
    increase_numeric = [x[1] - x[0] for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase_numeric.append(0)
    df['increase_numeric'] = increase_numeric
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan  # No info on next month

    return df


def make_combined_df_semiyearly(datapath):
    """
    Function that creates a pandas dataframe for all districts with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder

    Returns
    -------
    df : pandas dataframe
    """

    prevdf = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    districts = prevdf['district'].unique()

    df_list = []
    for district in districts:
        district_df = make_district_df_semiyearly(datapath, district)
        district_df['district'] = district
        df_list.append(district_df)

    df = pd.concat(df_list, ignore_index=True)
    df['district_encoded'] = df['district'].astype('category').cat.codes

    return df


def make_aggrgated_csv(datapath, start_time):
    """
    This function takes a datpath and creates the aggregated csv with chosen column from that path
    :param datapath: the data folder path
    :return: doen't return but stores csv at path
    """
    print("Aggregating datasets ...")

    # Create the semiyearly dataframe for all districts
    df = make_combined_df_semiyearly(datapath)

    # Sort dataframe on date and reset the index
    df.sort_values('date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    drop_districts = ['Saakow/Salagle', 'Badhan', 'Laasqoray/Badhan', 'Belet Weyne (Mataban)', 'Baydhaba/Bardaale']
    df = df[df.district.isin(drop_districts) == False].copy()

    df.to_csv(datapath + 'semiyearly_chosen_columns.csv')

    print(f"Datasets were aggregated and the new CSV was saved as 'semiyearly_chosen_columns.csv'. ",
          f"({round((time.time() - start_time),2)}s)\n")