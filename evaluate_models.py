"""Evaluate models from the COVID-19 Forecast Hub.

COVID-19 Forecast Hub: https://github.com/reichlab/covid19-forecast-hub
Learn more at: https://github.com/youyanggu/covid19-forecast-hub-evaluation

To see list of command line options: `python evaluate_models.py --help`
"""

import argparse
import datetime
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def update_pandas_settings():
    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = '{:.1f}'.format


def str_to_date(date_str, fmt='%Y-%m-%d'):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()


def find_last_projections(fnames, proj_date):
    """Find the appropriate projections file.

    The projection date is always a Monday. Following CDC submission guidelines
        on https://github.com/reichlab/covid19-forecast-hub, we use the latest
        projection that is within 3 days of the projection date. If no file
        exists, then we do not consider that model for that date.

    Params
    ------
    fnames - list of possible file names (file format: YYYY-MM-DD-team-model.csv)
    proj_date - the date of the projection (must be a Monday)
    """
    last_valid_fname = None
    last_valid_date = None
    for fname in fnames:
        file_basename = os.path.basename(fname)
        try:
            file_date = str_to_date(file_basename[:10])
        except ValueError:
            continue
        # Consider latest projections from within last 3 days for Monday's projections (Fri-Mon)
        # Note: from 2020-05-18, only projections from Sun or Mon are used for ensembles
        days_tolerance = 3
        if file_date <= proj_date and (proj_date - file_date).days <= days_tolerance:
            if last_valid_date is None or file_date > last_valid_date:
                last_valid_fname = fname
                last_valid_date = file_date
    return last_valid_fname, last_valid_date


def validate_projections(df_model):
    """Verify all columns are in dataframe and convert dates to datetime."""
    for col in ['forecast_date', 'target', 'target_end_date', 'location', 'type', 'quantile', 'value']:
        assert col in df_model.columns, col

    df_model['forecast_date'] = pd.to_datetime(df_model['forecast_date']).dt.date
    df_model['target_end_date'] = pd.to_datetime(df_model['target_end_date']).dt.date


def main(forecast_hub_dir, proj_date, eval_date, out_dir,
        use_point=True, print_additional_stats=False, merge_models=True):
    """For full description of methods, refer to:

    https://github.com/youyanggu/covid19-forecast-hub-evaluation
    """
    print('Forecast hub dir:', forecast_hub_dir)
    print('proj_date:', proj_date)
    print('eval_date:', eval_date)
    print('out_dir   :', out_dir)
    print('use_point:', use_point)

    assert os.path.isdir(forecast_hub_dir), \
        (f'Could not find COVID-19 Forecast Hub repo at: {forecast_hub_dir}.'
            ' You can provide the location via the --forecast_hub_dir flag')
    assert eval_date > proj_date, 'evaluation date must be greater than the projection date'
    assert proj_date.weekday() == 0, 'proj_date must be a Monday'
    assert eval_date.weekday() == 5, 'eval date must be a Saturday'

    model_ran_date = proj_date - datetime.timedelta(days=1)
    days_ahead = (eval_date - proj_date).days
    print('Days ahead:', days_ahead)

    update_pandas_settings()

    df_states = pd.read_csv(f'{forecast_hub_dir}/data-locations/locations.csv', dtype=str)
    abbr_to_location_name =  df_states.set_index('abbreviation')['location_name'].to_dict()
    abbr_to_fips = df_states.set_index('abbreviation')['location'].to_dict()
    US_TERRITORIES = ['AS', 'GU', 'MP', 'PR', 'VI', 'UM'] # we do not evaluate US territories

    fips_to_us_state = {v : abbr_to_location_name[k] for k,v in abbr_to_fips.items()}
    regions_to_evaluate = ['US'] + [s for s in abbr_to_location_name if s not in US_TERRITORIES]
    fpis_to_evaluate = ['US' if x == 'US' else abbr_to_fips[x] for x in regions_to_evaluate]

    print('=================================================')
    print('Loading data from COVID-19 Forecast Hub')
    print('=================================================')
    model_to_projections = {}
    all_models_dirs = sorted(glob.glob(f'{forecast_hub_dir}/data-processed/*'))
    for model_dir in all_models_dirs:
        model_name = os.path.basename(model_dir)
        if not os.path.isdir(model_dir):
            continue
        model_fnames = sorted(glob.glob(f'{model_dir}/*.csv'))
        last_valid_fname, last_valid_date = find_last_projections(model_fnames, proj_date)
        if last_valid_fname is None:
            model_basenames = [os.path.basename(fname) for fname in model_fnames]
            print(f'{model_name} - No files within range: {model_basenames}')
            continue
        print(f'{model_name} - Found file: {os.path.basename(last_valid_fname)}')

        model_to_projections[model_name] = {
            'last_valid_fname' : last_valid_fname,
            'last_valid_date' : last_valid_date,
        }

    truth_file_name = forecast_hub_dir / 'data-truth' / 'truth-Cumulative Deaths.csv'
    df_truth_raw = pd.read_csv(truth_file_name)
    df_truth_raw['date'] = pd.to_datetime(df_truth_raw['date']).dt.date
    df_truth_raw = df_truth_raw.rename(columns={'value' : 'total_deaths'})
    df_truth_raw = df_truth_raw[['date', 'location', 'total_deaths']]

    df_truth = df_truth_raw[df_truth_raw['date'] == eval_date]
    df_truth = df_truth.set_index('location')['total_deaths']
    df_truth_filt = df_truth[df_truth.index.isin(fpis_to_evaluate)]
    us_truth = df_truth_filt['US']

    proj_date_total_deaths = \
        df_truth_raw[df_truth_raw['date'] == proj_date].set_index('location')['total_deaths']['US']
    additional_deaths = us_truth - proj_date_total_deaths

    print('Incident US deaths:', additional_deaths)

    df_truth_past = df_truth_raw[df_truth_raw['date'] == model_ran_date]
    df_truth_past = df_truth_past.set_index('location')['total_deaths']
    df_truth_past_minus_7_days = df_truth_raw[df_truth_raw['date'] == model_ran_date-datetime.timedelta(days=7)]
    df_truth_past_minus_7_days = df_truth_past_minus_7_days.set_index('location')['total_deaths']
    df_truth_per_day = (df_truth_past - df_truth_past_minus_7_days) / 7

    # Baseline #1 uses the avg daily deaths from previous week to make all future projections
    baseline_daily_decay = 1
    weighted_days = sum([baseline_daily_decay**i for i in range(days_ahead+1)])
    df_baseline = df_truth_past + df_truth_per_day * weighted_days
    df_baseline_filt = df_baseline[(df_baseline.index.isin(fpis_to_evaluate)) & (~pd.isnull(df_baseline))]

    # Baseline #2 uses the avg daily deaths from previous week and make a 2% daily decrease
    baseline2_daily_decay = 0.98
    weighted_days2 = sum([baseline2_daily_decay**i for i in range(days_ahead+1)])
    df_baseline2 = df_truth_past + df_truth_per_day * weighted_days2
    df_baseline2_filt = df_baseline2[(df_baseline2.index.isin(fpis_to_evaluate)) & (~pd.isnull(df_baseline2))]

    model_to_num_locations = {}
    model_to_errors = {}
    model_to_df = {}
    model_to_us_projection = {}

    # add baseline models
    model_to_num_locations['Baseline'] = len(df_baseline_filt)
    df_model_diffs = df_baseline_filt - df_truth_filt
    model_to_errors['Baseline'] = df_model_diffs.to_dict()
    model_to_df['Baseline'] = df_baseline_filt
    model_to_us_projection['Baseline'] = df_baseline_filt['US']

    baseline_name = f'Baseline_{baseline2_daily_decay}'
    model_to_num_locations[baseline_name] = len(df_baseline2_filt)
    df_model_diffs2 = df_baseline2_filt - df_truth_filt
    model_to_errors[baseline_name] = df_model_diffs2.to_dict()
    model_to_df[baseline_name] = df_baseline2_filt
    model_to_us_projection[baseline_name] = df_baseline2_filt['US']

    date_to_cu_select = {
        datetime.date(2020,4,13) : 'CU-70contact',
        datetime.date(2020,4,20) : 'CU-70contact',
        datetime.date(2020,4,27) : 'CU-70contact',
        datetime.date(2020,5,4) : 'CU-80contact1x10p',
    }

    for model_name in model_to_projections:
        # Load projections from each model
        print('-----------------------------')
        print(model_name)

        projections_dict = model_to_projections[model_name]
        if model_name.startswith('CU-'):
            # handle multiple models by CU
            if proj_date in date_to_cu_select:
                if model_name == date_to_cu_select[proj_date]:
                    model_name = 'CU-select'
                else:
                    continue
            elif proj_date >= datetime.date(2020,5,11) and model_name != 'CU-select':
                continue
            else:
                assert model_name == 'CU-select'

        df_model_raw = pd.read_csv(projections_dict['last_valid_fname'],
            dtype={'location' : str})
        validate_projections(df_model_raw)
        print('Max projection date: {} - {:.1f} weeks ahead'.format(
            df_model_raw['target_end_date'].max(),
            (df_model_raw['target_end_date'].max() - proj_date).days / 7))

        df_model = df_model_raw[df_model_raw['target_end_date'] == eval_date]
        assert df_model['location'].apply(lambda x: isinstance(x, str)).all(), \
            'All FIPS locations must be a string'
        model_to_df[model_name] = df_model

        if df_model['target'].str.contains('wk ahead cum death').sum() > 0:
            target_str = 'wk ahead cum death'
        else:
            target_str = 'day ahead cum death'

        has_point = (df_model['type'] == 'point').sum() > 0
        has_median = (df_model['quantile'] == 0.5).sum() > 0

        if not has_point:
            print('* No point data')
        if not has_median:
            print('* No median data')
        if has_point and (use_point or not has_median):
            df_model_filt = df_model[
                (df_model['target'].str.contains(target_str)) & \
                (df_model['type'] == 'point') & \
                (df_model['location'].isin(fpis_to_evaluate))]
        else:
            df_model_filt = df_model[
                (df_model['target'].str.contains(target_str)) & \
                (df_model['quantile'] == 0.5) & \
                (df_model['location'].isin(fpis_to_evaluate))]

        print('Num unique locations (pre-filt) :', len(df_model['location'].unique()))
        num_locations = len(df_model_filt['location'].unique())
        print('Num unique locations (post-filt):', num_locations)
        assert num_locations <= len(fips_to_us_state), num_locations
        if len(df_model_filt) == 0:
            print('No rows after filt, skipping...')
            continue

        model_to_num_locations[model_name] = num_locations

        df_model_filt_values = df_model_filt.set_index('location')['value']

        df_model_diffs = df_model_filt_values - df_truth_filt
        diffs_dict = df_model_diffs.to_dict()
        model_to_errors[model_name] = diffs_dict
        model_to_us_projection[model_name] = df_model_filt_values.get('US', np.nan)

    print('=================================================')
    print('Begin Evaluation')
    print('=================================================')

    df_errors = pd.DataFrame(model_to_errors).T
    assert model_to_num_locations == df_errors.notna().sum(axis=1).to_dict(), \
        'Certain locations not parsed'

    df_errors = df_errors.rename(columns=fips_to_us_state).sort_index()
    print('Number of locations with projections')
    print(df_errors.notna().sum(axis=1))

    df_errors_us = pd.DataFrame({
        f'total_deaths_{proj_date}' : proj_date_total_deaths,
        'predicted_deaths' : model_to_us_projection,
        'actual_deaths' : us_truth,
        'additional_deaths' : additional_deaths,
    })
    df_errors_us['error'] = df_errors_us['predicted_deaths'] - df_errors_us['actual_deaths']
    assert ((df_errors_us['error'] == df_errors['US']) | \
        (np.isnan(df_errors_us['error']) & np.isnan(df_errors['US']))).all()

    if merge_models:
        # For fairness, we average the projections if there are multiple submissions
        for merge_models_prefix in ['Imperial']:
            print('-----------\nCombining:', merge_models_prefix)
            model_states_mask = df_errors.index.str.contains(merge_models_prefix)
            model_us_mask = df_errors_us.index.str.contains(merge_models_prefix)

            print('Num rows, us rows:', model_states_mask.sum(), model_us_mask.sum())
            if model_states_mask.sum() > 0:
                df_avg = df_errors.loc[model_states_mask].mean(axis=0)
                df_errors = df_errors.loc[~df_errors.index.str.contains(merge_models_prefix)]
                df_errors.loc[f'{merge_models_prefix}-combined'] = df_avg

            if model_us_mask.sum() > 0:
                df_us_avg = df_errors_us.loc[model_us_mask].mean(axis=0)
                df_errors_us = df_errors_us.loc[~df_errors_us.index.str.contains(merge_models_prefix)]
                df_errors_us.loc[f'{merge_models_prefix}-combined'] = df_us_avg

    df_errors_us['error'] = df_errors_us['predicted_deaths'] - df_errors_us['actual_deaths']
    df_errors_us['perc_error'] = (df_errors_us['error'] / df_errors_us['additional_deaths']).apply(
        lambda x: '' if pd.isnull(x) else f'{x:.1%}')
    df_errors_us[f'total_deaths_{proj_date}'] = df_errors_us[f'total_deaths_{proj_date}'].astype(int)
    df_errors_us['actual_deaths'] = df_errors_us['actual_deaths'].astype(int)

    print('=================================================')
    print('US Evaluation:')
    print('=================================================')
    df_errs_us_summary = df_errors_us.reindex(df_errors_us['error'].abs().sort_values().index)
    df_errs_us_summary.name = 'US Projected - True'
    print(df_errs_us_summary)

    if out_dir:
        us_errs_fname = f'{out_dir}/{proj_date}_{eval_date}_us_errs.csv'
        df_errs_us_summary.to_csv(us_errs_fname, float_format='%.1f')
        print('Saved to:', us_errs_fname)

    print('=================================================')
    print('State-by-state Evaluation:')
    print('=================================================')
    df_errors_states = df_errors.drop(columns=['US'])
    # filter out models without most state projections
    df_errors_states = df_errors_states.loc[df_errors_states.notna().sum(axis=1) > 40]
    print('Number of states with valid projections:')
    print(df_errors_states.notna().sum(axis=1))

    # we fill na with avg abs error for that state
    df_errors_states = df_errors_states.fillna(df_errors_states.abs().mean())
    print(df_errors_states)

    df_sq_errs_states = df_errors_states**2
    print('----------------------\nStates - mean squared errors:')
    df_sq_errs_states_summary = df_sq_errs_states.T.describe().T.sort_values('mean')
    df_sq_errs_states_summary = df_sq_errs_states_summary.rename(columns={'50%' : 'median'})
    cols = ['count', 'mean', 'median'] + \
        [c for c in df_sq_errs_states_summary.columns if c not in ['count', 'mean', 'median']]
    df_sq_errs_states_summary = df_sq_errs_states_summary[cols]
    print(df_sq_errs_states_summary)
    if out_dir:
        sq_errs_fname = f'{out_dir}/{proj_date}_{eval_date}_sq_errs.csv'
        df_sq_errs_states_summary.to_csv(sq_errs_fname, float_format='%.1f')
        print('Saved to:', sq_errs_fname)

    df_abs_errs_states = df_errors_states.abs()
    print('----------------------\nStates - mean absolute errors:')
    df_abs_errs_states_summary = df_abs_errs_states.T.describe().T.sort_values('mean')
    df_abs_errs_states_summary = df_abs_errs_states_summary.rename(columns={'50%' : 'median'})
    cols = ['count', 'mean', 'median'] + \
        [c for c in df_abs_errs_states_summary.columns if c not in ['count', 'mean', 'median']]
    df_abs_errs_states_summary = df_abs_errs_states_summary[cols]
    print(df_abs_errs_states_summary)
    if out_dir:
        abs_errs_fname = f'{out_dir}/{proj_date}_{eval_date}_abs_errs.csv'
        df_abs_errs_states_summary.to_csv(abs_errs_fname, float_format='%.1f')
        print('Saved to:', abs_errs_fname)

    # Print the average rank for state-by-state projections (1=most accurate)
    # We ignore baseline for ranks
    df_ranks = df_errors_states[~df_errors_states.index.str.startswith('Baseline')]
    df_ranks = df_ranks.abs().rank()
    print('----------------------\nMean/median ranks:')
    df_ranks_summary = df_ranks.mean(axis=1).sort_values()
    df_ranks_summary.name = 'mean_rank'
    print(df_ranks_summary)
    if out_dir:
        mean_ranks_fname = f'{out_dir}/{proj_date}_{eval_date}_mean_ranks.csv'
        df_ranks_summary.to_csv(mean_ranks_fname, float_format='%.1f')
        print('Saved to:', mean_ranks_fname)

    if print_additional_stats:
        print('=================================================')
        print('R^2 Correlation of errors:')
        print('=================================================')
        with pd.option_context('display.float_format', '{:.3f}'.format):
            print((df_errors_states.T.corr()**2))

        df_errs = (df_errors_states / df_errors_states.abs().mean(axis=0)).T
        df_errs['actual_deaths'] = df_truth_filt.rename(index=fips_to_us_state)
        print('=================================================')
        print('Error / Mean Error (of all models):')
        print('=================================================')
        print(df_errs)
        print('-------------------------------------------------')
        print('Correlation between error and total deaths:')
        print(df_errs.corr()['actual_deaths'].sort_values().to_string(float_format='{:.3f}'.format))
        print('-------------------------------------------------')
        print('Correlation between abs error and total deaths:')
        print(df_errs.abs().corr()['actual_deaths'].sort_values().to_string(float_format='{:.3f}'.format))

    pd.reset_option('display.float_format') # reset way we print floats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Script to evaluate models from the COVID-19 Forecast Hub. For more info: '
            'https://github.com/youyanggu/covid19-forecast-hub-evaluation'))
    parser.add_argument('proj_date', help='Date of projection. Must be a Monday')
    parser.add_argument('eval_date', help='Date of evaluation. Must be a Saturday')
    parser.add_argument('--forecast_hub_dir', help=('Local location of the covid19-forecast-hub repo:'
        'https://github.com/reichlab/covid19-forecast-hub. By default, check in the parent directory.'))
    parser.add_argument('--out_dir', help='Directory to save outputs (if provided)')
    parser.add_argument('--use_median', action='store_true',
        help='Use median estimate instead of point estimate')
    parser.add_argument('--print_additional_stats', action='store_true',
        help='Print additional statistics, like mean rank and residual analysis')
    args = parser.parse_args()

    proj_date = str_to_date(args.proj_date)
    eval_date = str_to_date(args.eval_date)

    if args.forecast_hub_dir:
        forecast_hub_dir = Path(args.forecast_hub_dir)
    else:
        forecast_hub_dir = Path(os.path.abspath(__file__)).parent.parent / 'covid19-forecast-hub'

    main(forecast_hub_dir, proj_date, eval_date, args.out_dir,
        use_point=(not args.use_median), print_additional_stats=args.print_additional_stats)

    print('=================================================')
    print('Done', datetime.datetime.now())

