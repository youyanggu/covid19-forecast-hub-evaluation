"""
Downloads all historical truth data from the Reich Lab:
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Cumulative%20Deaths.csv

This script is a reformatted version of the one originally developed by the Reich Lab:

https://github.com/reichlab/covidData/blob/master/code/data-processing/download-historical-jhu.py
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


def main():
    overwrite = False
    # if below is True, only download the truth uploaded on Sundays, the day after eval_date
    eval_dates_only = True

    base_dir = Path(os.path.abspath(__file__)).parent
    base_files = [
        'truth-Cumulative Deaths.csv',
    ]

    for base_file in base_files:
        # retrieve information about all commits that modified the file we want
        all_commits = []

        page = 0
        while True:
            page += 1
            print(f'Fetching page {page}...')
            r = requests.get(
                'https://api.github.com/repos/reichlab/covid19-forecast-hub/commits',
                params = {
                    'path': 'data-truth/truth-Cumulative Deaths.csv',
                    'page': str(page)
                }
            )

            if (not r.ok) or (r.text == '[]'):
                break

            all_commits += json.loads(r.text or r.content)

        # date of each commit
        commit_dates = [
            str_to_date(commit['commit']['author']['date'][0:10]) for commit in all_commits
        ]

        # sha for the last commit made each day
        commit_date_to_sha_and_fname = {}
        for i, commit_date in enumerate(commit_dates):
            result_path =  f'{base_dir}/truth-cumulative-deaths-{commit_date}.csv'
            if commit_date not in commit_date_to_sha_and_fname:
                commit_date_to_sha_and_fname[commit_date] = (all_commits[i]['sha'], result_path)

        # download and save the csvs
        for commit_date, sha_and_fname in commit_date_to_sha_and_fname.items():
            commit_sha, result_path = sha_and_fname
            if os.path.isfile(result_path) and not overwrite:
                continue
            if eval_dates_only and commit_date.weekday() != 0:
                continue
            url = requests.utils.requote_uri(
                'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/'
                f'{commit_sha}/data-truth/{base_file}')
            df = pd.read_csv(url, dtype={'location' : str})

            print('Saving to:', result_path)
            df.to_csv(result_path, index=False)


if __name__ == '__main__':
    main()
