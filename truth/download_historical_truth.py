"""
Downloads all historical truth data from the Reich Lab:
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Cumulative%20Deaths.csv

This script is a reformatted version of the one originally developed by the Reich Lab:

https://github.com/reichlab/covidData/blob/master/code/data-processing/download-historical-jhu.py

Note: GitHub API only allows 60 calls per hour from an unauthorized IP
"""

import datetime
import json
import os
from pathlib import Path

import pandas as pd
import requests


def str_to_date(date_str, fmt='%Y-%m-%d'):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()


def first_sunday_before(date):
    weekday = date.weekday()
    return date - datetime.timedelta(days=(weekday + 1) % 7)


def main():
    overwrite = False
    # if below is True, only download the truth uploaded on Sundays, the day after eval_date
    eval_dates_only = True

    base_dir = Path(os.path.abspath(__file__)).parent
    base_file_to_out_fname = {
        'truth-Cumulative Deaths.csv' : 'truth-cumulative-deaths',
        #'truth-Incident Cases.csv'  : 'truth-incident-cases', # not needed since we just directly copy from reich repo
    }

    for base_file, out_fname in base_file_to_out_fname.items():
        # retrieve information about all commits that modified the file we want
        all_commits = []

        page = 0
        while True:
            page += 1
            print(f'{base_file} - Fetching page {page}...')
            r = requests.get(
                'https://api.github.com/repos/reichlab/covid19-forecast-hub/commits',
                params = {
                    'path': f'data-truth/{base_file}',
                    'page': str(page)
                }
            )

            if (not r.ok) or (r.text == '[]'):
                break

            all_commits += json.loads(r.text or r.content)

        # date of each commit, in reverse chronological order
        commit_dates = [
            str_to_date(commit['commit']['author']['date'][0:10]) for commit in all_commits
        ]

        # sha for the last commit made each day
        commit_date_to_sha_and_fname = {}
        for i, commit_date in enumerate(commit_dates):
            result_path =  f'{base_dir}/{out_fname}-{commit_date}.csv'
            if commit_date not in commit_date_to_sha_and_fname:
                commit_date_to_sha_and_fname[commit_date] = (all_commits[i]['sha'], result_path)

        assert list(sorted(commit_date_to_sha_and_fname.keys(), reverse=True)) == \
            list(commit_date_to_sha_and_fname.keys()), 'dict is not sorted'

        # download and save the csvs in chronological order
        sundays_covered = {}
        for commit_date, sha_and_fname in reversed(commit_date_to_sha_and_fname.items()):
            commit_sha, result_path = sha_and_fname

            continue_loop = False
            if os.path.isfile(result_path) and not overwrite:
                continue_loop = True
            elif commit_date <= datetime.date(2020,11,1) and eval_dates_only and commit_date.weekday() != 0:
                continue_loop = True
            elif commit_date == datetime.date(2020,10,12):
                continue_loop = True # bad file

            if continue_loop:
                last_date = commit_date - datetime.timedelta(days=1)
                sunday_before = first_sunday_before(last_date)
                sundays_covered[sunday_before] = True
                continue

            url = requests.utils.requote_uri(
                'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/'
                f'{commit_sha}/data-truth/{base_file}')
            print(f'Downloading for {result_path}: {url}')
            df = pd.read_csv(url, dtype={'location' : str})
            df['date'] = pd.to_datetime(df['date']).dt.date

            last_date = df['date'].max()
            sunday_before = first_sunday_before(last_date)
            if sunday_before in sundays_covered:
                continue

            sundays_covered[sunday_before] = True
            print('Saving to:', result_path)
            df.to_csv(result_path, index=False)


if __name__ == '__main__':
    main()
