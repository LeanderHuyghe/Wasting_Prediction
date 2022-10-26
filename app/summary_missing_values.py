import pandas as pd
import time


def writing_summary(data_path, output):
    """
    This function takes as input the path to the csv and writes the summary into a text file at the output path
    :param data_path:
    :param output:
    """
    start_time = time.time()
    print("Writing summary missing values ...")

    df = pd.read_csv(data_path, parse_dates=['date']).drop('Unnamed: 0', axis=1)

    summary = open(output, 'w')

    summary.write(f'Number of columns: {len(df.columns)} \nList of columns:')
    for i in range(len(df.columns)):
        if i == len(df.columns) - 1:
            summary.write(f' {df.columns[i]}.\n\n')
        elif (i+1) % 5 == 0:
            summary.write(f' {df.columns[i]},\n')
        else:
            summary.write(f' {df.columns[i]},')
    summary.write(f'Number of districts: {len(df.district.unique())} \n')

    missing = len(df) - df.count()
    summary.write(f'\nMissing values per column: \n{missing.sort_values(ascending=False)}'[:-13] + '\n')
    stri = ''
    missing_col = missing[missing > 0].index
    for i in range(len(missing_col)):
        if i == len(missing_col) - 1:
            stri += str(missing_col[i]) + '.\n\n'
        elif (i+1) % 5 == 0:
            stri += str(missing_col[i]) + ',\n'
        else:
            stri += str(missing_col[i]) + ','
    summary.write(f'\nColumns with missing values: \n{stri}')

    df2 = df[['date', 'district','total population']]
    df2 = df2[df2.isnull().any(axis=1)]
    summary.write(f'\nNumber of dates missing for total popuation: {len(df2.date.unique())}\n')
    summary.write(f'Missing date: {df2.date.unique()[0]}\n')

    df2 = df[['date', 'district','ndvi_score']]
    df2 = df2[df2.isnull().any(axis=1)]
    summary.write(f'\nNumber of dates missing for ndvi_score: {len(df2.date.unique())}\n')
    summary.write(f'Number of districts for ndvi_score: {len(df2.district.unique())}\n')
    summary.write(f'Missing district: {df2.district.unique()[0]}\n')

    df2 = df[['date', 'district','phase3plus_perc']]
    df2 = df2[df2.isnull().any(axis=1)]
    summary.write(f'\nNumber of dates missing for phase3plus_perc: {len(df2.date.unique())}\n')
    summary.write(f'Number of districts for phase3plus_perc: {len(df2.district.unique())}\n')
    summary.write(f'Missing district: {df2.district.unique()[0]}\n')

    df2 = df[['date', 'district','next_prevalence']]
    df2 = df2[df2.isnull().any(axis=1)]
    di_dist = {}
    di_count = {}
    for i in df2.index:
        if df2.date[i] in di_dist:
            di_dist[df2.date[i]] += [df2.district[i]]
            di_count[df2.date[i]] += 1
        else:
            di_dist[df2.date[i]] = [df2.district[i]]
            di_count[df2.date[i]] = 1
    summary.write(f'\nNumber of dates missing for next_prevalence: {len(df2.date.unique())}\n')
    summary.write(f'Number of districts for next_prevalence: {len(df2.district.unique())}\n')
    summary.write(f'Number of missing districts for each date: \n{di_count}\n')
    # summary.write(f'Missing districts for each date: \n{di_dist}\n')

    df2 = df[['date', 'district','n_conflict_total']]
    df2 = df2[df2.isnull().any(axis=1)]
    di_count = {}
    di_datesc = {}
    for i in df2.index:
        if df2.date[i] in di_count:
            di_count[df2.date[i]] += 1
        else:
            di_count[df2.date[i]] = 1
        if df2.district[i] in di_datesc:
            di_datesc[df2.district[i]] += 1
        else:
            di_datesc[df2.district[i]] = 1
    summary.write(f'\nNumber of dates missing for n_conflict_total: {len(df2.date.unique())}\n')
    summary.write(f'Number of districts for n_conflict_total: {len(df2.district.unique())}\n')
    summary.write(f'Number of missing districts for each date: \n{di_count}\n')
    summary.write(f'Number of missing dates for each district: \n{di_datesc}\n')

    df2 = df[['date', 'district','Price of water']]
    df2 = df2[df2.isnull().any(axis=1)]
    di_count = {}
    di_datesc = {}
    for i in df2.index:
        if df2.date[i] in di_count:
            di_count[df2.date[i]] += 1
        else:
            di_count[df2.date[i]] = 1
        if df2.district[i] in di_datesc:
            di_datesc[df2.district[i]] += 1
        else:
            di_datesc[df2.district[i]] = 1
    summary.write(f'\nNumber of dates missing for Price of water: {len(df2.date.unique())}\n')
    summary.write(f'Number of districts for Price of water: {len(df2.district.unique())}\n')
    summary.write(f'Number of missing districts for each date: \n{di_count}\n')
    summary.write(f'Number of missing dates for each district: \n{di_datesc}\n')

    summary.close()

    print(f"The missing values summary was written in a text file and saved in the output folder. ",
          f"({round((time.time() - start_time),2)}s)\n")
