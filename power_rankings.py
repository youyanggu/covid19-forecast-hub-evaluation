from collections import defaultdict
import glob

import numpy as np
import pandas as pd


"""
Compute a "Power Rankings" based on the mean 'N weeks ahead' percentiles,
    where N is 1-6 weeks ahead. This includes both US and state-by-state rankings.

There are a total of 12 summaries, US and state-by-state for each of 1-6 weeks ahead.
    If a model is the best in every summary, it will have a mean percentile of 0.
    If a model is the median rank in every summary, it will have a mean percentile of 0.5.
    The lower the mean percentile, the better the model.

If a model does not appear in a summary (e.g. it does not make 6 week ahead forecasts),
    it does not get included in the mean percentile. But a model must be included in at least
    6 summaries to appear in the Power Rankings.

"""

print('========================================')
print('Power Rankings')
print('========================================')
model_to_percentiles = defaultdict(list)
for fname in glob.glob('summary/*weeks_ahead*.csv'):
    df = pd.read_csv(fname, index_col=0)
    # Only count models with 3 or more entries in summary
    df_filt = df[[c for c in df.columns if 'mean_sq_abs_error' not in c]]
    df_filt = df_filt[(~pd.isnull(df_filt)).sum(axis=1) >= 3]

    n = len(df_filt) - 1
    for rank, model_name in enumerate(df_filt.index):
        model_to_percentiles[model_name].append(rank / n)

model_to_mean_percentile = {}
for model, percentiles in model_to_percentiles.items():
    if len(percentiles) < 6:
        continue # only include models in 6 or more summaries
    model_to_mean_percentile[model] = (np.mean(percentiles), len(percentiles))

df_ranks = pd.DataFrame(model_to_mean_percentile,
    index=['mean_percentile', 'num_summaries']).T.sort_values('mean_percentile')
df_ranks['num_summaries'] = df_ranks['num_summaries'].astype(int)
print(df_ranks)
out_fname = 'summary/power_rankings.csv'
df_ranks.to_csv(out_fname)
print('Saved rankings to:', out_fname)
