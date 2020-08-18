import argparse
import datetime
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def str_to_date(date_str, fmt='%Y-%m-%d'):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()


def get_dates_from_fname(fname):
    """Returns the projection and eval date given a file name."""
    basename = os.path.basename(fname).replace('.csv', '')
    proj_date = str_to_date(basename.split('_')[1])
    eval_date = str_to_date(basename.split('_')[2])
    assert eval_date > proj_date

    return proj_date, eval_date


def filter_fnames_by_weeks_ahead(fnames, weeks_ahead):
    """Return evaluation files that match the provided weeks_ahead."""
    include_fnames = []
    for fname in fnames:
        proj_date, eval_date = get_dates_from_fname(fname)

        days_ahead = (eval_date - proj_date).days
        max_days_tolerance = 3
        if abs(days_ahead - 7*weeks_ahead) <= max_days_tolerance:
            # this is a weeks_ahead forecast
            include_fnames.append(fname)

    return include_fnames


def main(eval_date, weeks_ahead, evaluations_dir, out_dir):
    print('Evaluation date:', eval_date)
    print('Weeks ahead:', weeks_ahead)
    print('Evaluations dir:', evaluations_dir)
    print('Output dir:', out_dir)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    assert eval_date or weeks_ahead, \
        'must provide either an --eval_date or --weeks_ahead'
    assert not (eval_date and weeks_ahead), \
        'must provide only one of --eval_date or --weeks_ahead'

    print('==============================')
    print('US evaluations')
    print('==============================')
    if eval_date:
        global_fnames = sorted(glob.glob(
            f'{evaluations_dir}/{eval_date}/projections_*_{eval_date}.csv'))
    else:
        global_fnames = sorted(glob.glob(
            f'{evaluations_dir}/*/projections_*.csv'))
        global_fnames = filter_fnames_by_weeks_ahead(global_fnames, weeks_ahead)

    assert len(global_fnames) > 0, 'Need global evaluation files'

    team_names = ['Baseline', 'YYG', 'IHME']
    row_ordering = ['num_countries',
        'num_countries_beat_baseline-YYG', 'num_countries_beat_baseline-IHME',
        'perc_beat_baseline-YYG', 'perc_beat_baseline-IHME',
        'mean_abs_error-Baseline', 'mean_abs_error-YYG', 'mean_abs_error-IHME',
    ]
    col_to_data = {}
    for global_fname in global_fnames:
        proj_date_, eval_date_ = get_dates_from_fname(global_fname)
        df_global = pd.read_csv(global_fname, index_col=0)

        num_countries = len(df_global)
        col_data = {
            'num_countries' : num_countries,
        }
        df_global_sum = df_global.abs().sum()
        for team_name in team_names:
            num_countries_with_proj = df_global[f'error-{team_name}'].count()
            if df_global_sum.loc[f'error-{team_name}'] == 0:
                # No projections
                col_data[f'num_countries_beat_baseline-{team_name}'] = np.nan
                col_data[f'perc_beat_baseline-{team_name}'] = np.nan
                col_data[f'mean_abs_error-{team_name}'] = np.nan
                continue
            if team_name != 'Baseline':
                num_beat_baseline = df_global_sum.loc[f'beat_baseline-{team_name}']
                col_data[f'num_countries_beat_baseline-{team_name}'] = int(num_beat_baseline)
                col_data[f'perc_beat_baseline-{team_name}'] = num_beat_baseline / num_countries_with_proj
            col_data[f'mean_abs_error-{team_name}'] = df_global_sum.loc[f'error-{team_name}'] / num_countries_with_proj

        col_to_data[f'{proj_date_}_{eval_date_}'] = col_data

    df_all = pd.DataFrame(col_to_data).T
    df_all = df_all[row_ordering].T

    if out_dir:
        if eval_date:
            out_fname = f'{out_dir}/baseline_comparison_global_{eval_date}.csv'
        else:
            out_fname = f'{out_dir}/baseline_comparison_{weeks_ahead}_weeks_ahead_global.csv'
        df_all.to_csv(out_fname, float_format='%.10g')
        print('Saved global summary to:', out_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Given an evaluation date or weeks ahead (not both),'
            'summarize all the historical projections that fit the criteria.'))
    parser.add_argument('--eval_date',
        help='Evaluate all projections based on eval_date')
    parser.add_argument('--weeks_ahead', type=int,
        help='Evaluate all projections based on number of weeks ahead.')
    parser.add_argument('--evaluations_dir',
        help='Directory containing the raw evaluations.')
    parser.add_argument('--out_dir',
        help='Directory to save output data.')

    args = parser.parse_args()
    eval_date = args.eval_date
    weeks_ahead = args.weeks_ahead
    evaluations_dir = args.evaluations_dir
    out_dir = args.out_dir

    if eval_date:
        eval_date = str_to_date(args.eval_date)
    if not evaluations_dir:
        evaluations_dir = Path(__file__).parent

    main(eval_date, weeks_ahead, evaluations_dir, out_dir)
    print('Done', datetime.datetime.now())

