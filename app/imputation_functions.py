from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import pandas as pd


def update_MAM(df):
    """
    This function updates MAM values
    :param df: the dataframe where we wan to update the MAM values
    :return: the updated dataframe
    """

    df.MAM = df.GAM - df.SAM
    return df

def update_total_population(df):
    """
    This function updates total_population values
    :param df: the dataframe where we wan to update the total_population values
    :return: the updated dataframe
    """
    df.date = pd.to_datetime(df.date)
    for i in range(len(df.district.unique())):
        # retrieve district name
        district = df.district.unique()[i]
        # retrieve index of the district for 2021-07-01
        index = df[(df.date == "2021-07-01") & (df.district == district)].index
        try:
            # set its value to 2021-01-01
            value = df[(df.date == "2021-01-01") & (df.district == district)]['total population'].values[0]
        except:
            # if that value does not exist, just use its previous recorded population value
            value = df[df.district == district]['total population'].values[-2]
        # change nans to the previous population value which is now carried forward
        df.loc[index, 'total population'] = value
    return df

def update_n_conflict_total_spline(df):
    """
    This function updates n_conflict_total values
    :param df: the dataframe where we wan to update the n_conflict_total values
    :return: the updated dataframe
    """
    df['n_conflict_total_spline'] = df['n_conflict_total']
    for i in range(len(df.district.unique())):
        # retrieve district name
        name = df.district.unique()[i]
        # retrieve conflict data
        data = df[df.district==name]['n_conflict_total']
        # retrieve indices of the missing values
        index = data[data.isna()].index.tolist()
        # interpolate and fill any missing values with a reasonable estimate
        data_spline = data.interpolate("spline", order=1).bfill()
        # retrieve interpolated values at the required indices
        conflict = data_spline[index].values.tolist()
        df.loc[index,'n_conflict_total_spline'] = conflict
    return df

def update_price_of_water_MICE(df):
    """
    This function updates n_conflict_total values
    :param df: the dataframe where we wan to update the n_conflict_total values
    :return: the updated dataframe
    """
    X = df.select_dtypes(exclude=["object", "category"]).iloc[:, 1:]
    mice_imputer = IterativeImputer(n_nearest_features=5, max_iter=100).fit_transform(X)
    df_imputed = pd.DataFrame(mice_imputer, columns=X.columns)
    df["Price of water MICE"] = df_imputed["Price of water"]
    return df

def update_knn(df):
    """
    This function updates n_conflict_total, ndvi_score, phase3plus_perc and values
    :param df: the dataframe where we wan to update the values
    :return: the updated dataframe
    """
    # defining the columns where we wnat to remone some data
    df_imputation = df.copy()
    df_imputation = df_imputation.drop(['date', 'district', 'Average of centy', 'Average of centx'], axis=1)
    # we need to normalize the data to make sure the imputer isn't bisaed
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputation), columns=df_imputation.columns)
    imputer = KNNImputer(n_neighbors=5)
    df_scaled_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns)
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_scaled_imputed), columns=df_scaled_imputed.columns)
    # imputing values to initial dataset
    df.ndvi_score = df_imputed.ndvi_score
    df.phase3plus_perc = df_imputed.phase3plus_perc
    df["n_conflict_total_knn"] = df_imputed.n_conflict_total
    df["Price of water KNN"] = df_imputed["Price of water"]
    return df