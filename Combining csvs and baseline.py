#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

your_datapath = 'data/'


# In[2]:


# Cleaning of the misspelings of the districs

districts = ['Adan Yabaal', 'Afgooye', 'Afmadow', 'Baardheere', 'Badhaadhe', 'Baidoa', 
 'Baydhaba/Bardaale', 'Baki', 'Balcad', 'Banadir', 'Bandarbeyla', 'Baraawe', 'Belet Weyne', 'Belet Weyne (Mataban)','Belet Xaawo', 
'Berbera', 'Borama', 'Bossaso', 'Bu\'aale', 'Bulo Burto', 'Burco', 'Burtinle', 'Buuhoodle',
'Buur Hakaba', 'Cabudwaaq', 'Cadaado', 'Cadale', 'Caluula', 'Caynabo', 'Ceel Afweyn', 'Ceel Barde',
'Ceel Buur', 'Ceel Dheer', 'Ceel Waaq', 'Ceerigaabo', 'Dhuusamarreeb', 'Diinsoor', 'Doolow',
'Eyl', 'Gaalkacyo', 'Galdogob', 'Garbahaarey', 'Garoowe', 'Gebiley', 'Hargeysa', 'Hobyo', 'Iskushuban',
'Jalalaqsi', 'Jamaame', 'Jariiban', 'Jilib', 'Jowhar', 'Kismaayo', 'Kurtunwaarey', 'Laas Caanood', 'Laasqoray', 
'Laasqoray/Badhan', 'Badhan', 'Lughaye', 'Luuq', 'Marka', 'Owdweyne', 'Qandala', 'Qansax Dheere', 'Qardho', 'Qoryooley', 
'Rab Dhuure', 'Saakow', 'Saakow/Salagle' 'Sablaale', 'Sheikh', 'Taleex', 'Tayeeglow', 'Waajid', 'Wanla Weyn',
'Xarardheere', 'Xudun', 'Xudur', 'Zeylac']

def levenshteinDistanceDP(token1, token2):
    '''
    This function implements the levenshtein text similarity measure 
    and returns a numeric value representing the distance between two words
    '''
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
            if (token1[t1-1] == token2[t2-1]):
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
    '''
    This function checks whether the district name is the standard. 
    If it is not, then the word is corrected by known value or by levenshtein distance.
    It creates a list with all the standard districts and sets that list as district column.
    Uncomment print statements in last loop to see what districts changed by levenshtein algo.
    '''

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


path_admissions = your_datapath + 'admissions.csv'
path_FSNAU = your_datapath + 'FSNAU_riskfactors.csv'
path_ipc = your_datapath + 'ipc.csv'
path_ipc2 = your_datapath + 'ipc2.csv'
path_locations = your_datapath + 'locations.csv'
path_prevalence = your_datapath + 'prevalence_estimates.csv'
# path_production = your_datapath + 'production.csv'


df_admissions = pd.read_csv(path_admissions)
df_FSNAU = pd.read_csv(path_FSNAU)
df_ipc = pd.read_csv(path_ipc)
df_ipc.rename({'area': 'district'}, axis=1, inplace=True)
df_ipc2 = pd.read_csv(path_ipc2)
df_locations = pd.read_csv(path_locations)
df_locations = df_locations[df_locations.district != 'Grand Total']
df_prevalence = pd.read_csv(path_prevalence)


df_admissions = update_districts(df_admissions)
df_FSNAU = update_districts(df_FSNAU)
df_ipc = update_districts(df_ipc)
df_ipc2 = update_districts(df_ipc2)
df_locations = update_districts(df_locations)
df_prevalence = update_districts(df_prevalence)


df_admissions.to_csv(path_admissions)
df_FSNAU.to_csv(path_FSNAU)
df_ipc.to_csv(path_ipc)
df_ipc2.to_csv(path_ipc2)
df_locations.to_csv(path_locations)
df_prevalence.to_csv(path_prevalence)


# In[3]:


# Making prevalence, ipc and ipc2 monthly

df_ipc = pd.read_csv(your_datapath + 'ipc.csv', parse_dates=['date'])
df_ipc2 = pd.read_csv(your_datapath + 'ipc2.csv', parse_dates=['date'])
df_prevalence = pd.read_csv(your_datapath + 'prevalence_estimates.csv', parse_dates=['date'])

mon_prevalence = []
for dstrict in set(df_prevalence['district']):
    df2 = df_prevalence[df_prevalence['district']==dstrict].copy()
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
    df2.sort_values('date', inplace=True)
    
    start= df2.reset_index().date.astype('datetime64')[0] - datetime.timedelta(days=1)
    df3 = (pd.date_range(start,periods= len(df2.date)*6, freq='M') + timedelta(days=1)).to_frame()
    
    df = pd.merge_asof(left=df3.rename(columns={0: 'date'}), right=df2, direction='backward', on='date')
    mon_prevalence.append(df)
mon_prevalence = pd.concat(mon_prevalence, ignore_index=True)


mon_ipc = []
for dstrict in set(df_ipc['district']):
    df2 = df_prevalence[df_prevalence['district']==dstrict].copy()
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
    df2.sort_values('date', inplace=True)
    
    start= df2.reset_index().date.astype('datetime64')[0] - datetime.timedelta(days=1)
    df3 = (pd.date_range(start,periods= len(df2.date)*6, freq='M') + timedelta(days=1)).to_frame()
    
    df = pd.merge_asof(left=df3.rename(columns={0: 'date'}), right=df2, direction='backward', on='date')
    mon_ipc.append(df)
mon_ipc = pd.concat(mon_ipc, ignore_index=True)


mon_ipc2 = []
for dstrict in set(df_ipc2['district']):
    df2 = df_prevalence[df_prevalence['district']==dstrict].copy()
    df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
    df2.sort_values('date', inplace=True)
    
    start= df2.reset_index().date.astype('datetime64')[0] - datetime.timedelta(days=1)
    df3 = (pd.date_range(start,periods= len(df2.date)*6, freq='M') + timedelta(days=1)).to_frame()
    
    df = pd.merge_asof(left=df3.rename(columns={0: 'date'}), right=df2, direction='backward', on='date')
    mon_ipc2.append(df)
mon_ipc2 = pd.concat(mon_ipc2, ignore_index=True)


mon_prevalence.to_csv('data/mon_ipc2.csv')
mon_ipc.to_csv('data/mon_ipc.csv')
mon_ipc2.to_csv('data/mon_prevalence_estimates.csv')


# In[3]:





# In[4]:


# Reading all the csvs if you already have them and want to investigate them

df_admissions = pd.read_csv(your_datapath + 'admissions.csv', parse_dates=['date'])
df_FSNAU = pd.read_csv(your_datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
df_locations = pd.read_csv(your_datapath + 'locations.csv', parse_dates=['date'])
df_ipc = pd.read_csv(your_datapath + 'ipc.csv', parse_dates=['date'])
df_ipc2 = pd.read_csv(your_datapath + 'ipc2.csv', parse_dates=['date'])
df_prevalence = pd.read_csv(your_datapath + 'prevalence_estimates.csv', parse_dates=['date'])
df_mon_ipc = pd.read_csv(your_datapath + 'mon_ipc.csv', parse_dates=['date'])
df_mon_ipc2 = pd.read_csv(your_datapath + 'mon_ipc2.csv', parse_dates=['date'])
df_mon_prevalence = pd.read_csv(your_datapath + 'mon_prevalence_estimates.csv', parse_dates=['date'])


# In[4]:





# In[5]:


# Functions for making the combined monthly data frame

def make_district_df_monthly(datapath, district_name):
    """
    Function that creates a pandas dataframe for a single district with columns for the baseline model with monthly entries

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

	#Read all relevant datasets
    df_admissions = pd.read_csv(your_datapath + 'admissions.csv', parse_dates=['date'])
    df_FSNAU = pd.read_csv(your_datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    df_locations = pd.read_csv(your_datapath + 'locations.csv', parse_dates=['date'])
    df_mon_ipc = pd.read_csv(your_datapath + 'mon_ipc.csv', parse_dates=['date'])
    df_mon_ipc2 = pd.read_csv(your_datapath + 'mon_ipc2.csv', parse_dates=['date'])
    df_mon_prevalence = pd.read_csv(your_datapath + 'mon_prevalence_estimates.csv', parse_dates=['date'])
    
    #Select data for specific district
    df_admissions = df_admissions[df_admissions['district']==district_name]
    df_FSNAU = df_FSNAU[df_FSNAU['district']==district_name]
    df_locations = df_locations[df_locations['district']==district_name]
    df_mon_ipc = df_mon_ipc[df_mon_ipc['district']==district_name]
    df_mon_ipc2 = df_mon_ipc2[df_mon_ipc2['district']==district_name]
    df_mon_prevalence = df_mon_prevalence[df_mon_prevalence['district']==district_name]
    
    #Sort dataframes on date
    df_admissions.sort_values('date', inplace=True)
    df_FSNAU.sort_values('date', inplace=True)
    df_locations.sort_values('date', inplace=True)
    df_mon_ipc.sort_values('date', inplace=True)
    df_mon_ipc2.sort_values('date', inplace=True)
    df_mon_prevalence.sort_values('date', inplace=True)

    #Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(left=df_mon_prevalence, right=df_mon_ipc, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_mon_ipc2, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_locations, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_FSNAU, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_admissions, direction='backward', on='date')
    
    #Calculate prevalence 1lag
    df['prevalence_1lag'] = df['GAM Prevalence'].shift(1)
    df['next_prevalence'] = df['GAM Prevalence'].shift(-1)
    
    #Calculate prevalence 6lag
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(6)
    df['6lag_next_prevalence'] = df['GAM Prevalence'].shift(-6)
    
    #Select needed columns
#     df = df[['date', 'district', 'GAM Prevalence', 'next_prevalence', 'prevalence_6lag', 'ndvi_score', 'total population']]
#     df.columns = ['date', 'district', 'prevalence', 'next_prevalence', 'prevalence_6lag', 'covid', 'ndvi', 'ipc', 'population']
    
    #Add month column
    df['month'] = df['date'].dt.month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase = [False if x[1]<x[0] else True for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase.append(False)
    df['increase'] = increase
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan #No info on next month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase_numeric = [x[1] - x[0] for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase_numeric.append(0)
    df['increase_numeric'] = increase_numeric
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan #No info on next month
        
    return df


def make_combined_df_monthly(datapath):
    """
    Function that creates a pandas dataframe for all districts with columns for the baseline model with monthly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder

    Returns
    -------
    df : pandas dataframe
    """

    prevdf = pd.read_csv(datapath + 'mon_prevalence_estimates.csv', parse_dates=['date'])
    districts = prevdf['district'].unique()
    
    df_list = []
    for district in districts:
        district_df = make_district_df_monthly(datapath, district)
        district_df['district'] = district
        df_list.append(district_df)
        
    df = pd.concat(df_list, ignore_index=True)
    df['district_encoded'] = df['district'].astype('category').cat.codes

    return df


# In[6]:


# Functions for making the combined semiyearly data frame

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

	#Read all relevant datasets
    df_admissions = pd.read_csv(your_datapath + 'admissions.csv', parse_dates=['date'])
    df_FSNAU = pd.read_csv(your_datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    df_locations = pd.read_csv(your_datapath + 'locations.csv', parse_dates=['date'])
    df_ipc = pd.read_csv(your_datapath + 'ipc.csv', parse_dates=['date'])
    df_ipc2 = pd.read_csv(your_datapath + 'ipc2.csv', parse_dates=['date'])
    df_prevalence = pd.read_csv(your_datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    
    #Select data for specific district
    df_admissions = df_admissions[df_admissions['district']==district_name]
    df_FSNAU = df_FSNAU[df_FSNAU['district']==district_name]
    df_locations = df_locations[df_locations['district']==district_name]
    df_ipc = df_ipc[df_ipc['district']==district_name]
    df_ipc2 = df_ipc2[df_ipc2['district']==district_name]
    df_prevalence = df_prevalence[df_prevalence['district']==district_name]

    df_locations = df_locations.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_locations = df_locations.reset_index()
    df_locations['date'] = df_locations['date'].apply(lambda x : x.replace(day=1))
    
    df_FSNAU = df_FSNAU.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_FSNAU = df_FSNAU.reset_index()
    df_FSNAU['date'] = df_FSNAU['date'].apply(lambda x : x.replace(day=1))
    
    df_admissions = df_admissions.groupby(pd.Grouper(key='date', freq='6M')).mean()
    df_admissions = df_admissions.reset_index()
    df_admissions['date'] = df_admissions['date'].apply(lambda x : x.replace(day=1))
        
    #Sort dataframes on date
    df_admissions.sort_values('date', inplace=True)
    df_FSNAU.sort_values('date', inplace=True)
    df_locations.sort_values('date', inplace=True)
    df_ipc.sort_values('date', inplace=True)
    df_ipc2.sort_values('date', inplace=True)
    df_prevalence.sort_values('date', inplace=True)

    #Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(left=df_prevalence, right=df_ipc, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_ipc2, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_locations, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_FSNAU, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=df_admissions, direction='backward', on='date')
    
    #Calculate prevalence 6lag
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(1)
    df['next_prevalence'] = df['GAM Prevalence'].shift(-1)
    
    #Select needed columns
#     df = df[['date', 'district', 'GAM Prevalence', 'next_prevalence', 'prevalence_6lag', 'new_cases', 'ndvi_score', 'phase3plus_perc', 'cropdiv', 'total population']]
#     df.columns = ['date', 'district', 'prevalence', 'next_prevalence', 'prevalence_6lag', 'covid', 'ndvi', 'ipc', 'cropdiv', 'population']
    
    #Add month column
    df['month'] = df['date'].dt.month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase = [False if x[1]<x[0] else True for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase.append(False)
    df['increase'] = increase
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan #No info on next month
    
    #Add target variable: increase for next month prevalence (boolean)
    increase_numeric = [x[1] - x[0] for x in list(zip(df['GAM Prevalence'], df['GAM Prevalence'][1:]))]
    increase_numeric.append(0)
    df['increase_numeric'] = increase_numeric
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan #No info on next month
        
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


# In[7]:


# Create the monthly dataframe for all districts
df_mon = make_combined_df_monthly(your_datapath)

#Drop every row with missing values
# df_mon.dropna(inplace=True)

#Sort dataframe on date and reset the index
df_mon.sort_values('date', inplace=True)
df_mon.reset_index(inplace=True, drop=True)


# In[8]:


# Create the semiyearly dataframe for all districts
df_sy = make_combined_df_semiyearly(your_datapath)

#Drop every row with missing values
# df_sy.dropna(inplace=True)

#Sort dataframe on date and reset the index
df_sy.sort_values('date', inplace=True)
df_sy.reset_index(inplace=True, drop=True)


# In[13]:


df_sy.to_csv("semiyearly_data.csv")


# In[14]:


df_mon.to_csv("monthly_data.csv")


# In[12]:


# In[ ]:




