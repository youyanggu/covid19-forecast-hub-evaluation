import datetime

import pandas as pd

for i in range(5,21):
    df = pd.read_csv(f'evaluation/summary/summary_{i}_weeks_ahead_states.csv', index_col=0) 
    df_filt = df[[c for c in df.columns if 'sq_abs' not in c]]
    df_filt = df_filt.drop('Baseline')
    col_start_dates = [pd.to_datetime(c.split('_')[-2]).date() for c in df_filt.columns]
    col_end_dates = [pd.to_datetime(c.split('_')[-1]).date() for c in df_filt.columns]
    col_mask = []
    for start_date, end_date in zip(col_start_dates, col_end_dates):
        if start_date >= datetime.date(2020,5,11) and end_date <= datetime.date(2020,12,19):
            col_mask.append(True)
        else:
            col_mask.append(False)
    df_filt = df_filt[df_filt.columns[col_mask]]

    df_filt = df_filt.loc[df_filt[((~pd.isnull(df_filt)).sum(axis=1) > 0)].index] # filter rows/models
    df_filt = df_filt.loc[:, ((~pd.isnull(df_filt)).sum(axis=0) > 1).values] # filter columns / weeks with >1 models

    df_num_forecasts = (~pd.isnull(df_filt)).sum(axis=1).sort_values(ascending=False)
    df_is_first = (df_filt.rank() == df_filt.rank().min()).sum(axis=1).sort_values(ascending=False)
    df_is_last = (df_filt.rank() == df_filt.rank().max()).sum(axis=1).sort_values(ascending=False)

    df_tot = pd.concat([df_num_forecasts, df_is_first, df_is_last], axis=1)
    df_tot.columns = ['weeks_with_forecasts', 'weeks_lowest_mae', 'weeks_highest_mae']
    df_tot['frac_forecasts_is_lowest'] = df_tot['weeks_lowest_mae'] / df_tot['weeks_with_forecasts']
    df_tot['weeks_lowest_minus_highest'] = df_tot['weeks_lowest_mae'] - df_tot['weeks_highest_mae']
    df_tot = df_tot.sort_values(['weeks_lowest_minus_highest', 'weeks_lowest_mae', 'weeks_with_forecasts'], ascending=[False, False, True])

    print('-------------------')
    print(f'{i} weeks ahead')
    print(df_tot)
