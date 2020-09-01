## Evaluation of US Incident Case Forecasts

We present evaluations for US weekly incident case forecasts on a national, state-by-state, and county-by-county level.

The methodology for computing errors for incident case is straightforward: we simply compute the incident weekly cases and compare them to the [latest truth data](https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Incident%20Cases.csv).

We are only evaluating models with forecasts for at least 40 states or 1000 counties. For models with missing state or county projections, we substitute the missing error with the mean absolute error for that state/county (among all the models with available forecasts).

As usual, all the results here are reproducable. See the [Usage](https://github.com/youyanggu/covid19-forecast-hub-evaluation#usage) page for more details.

#### Sample Usage for 4-week-ahead evaluations (ending on August 29)
```
python evaluate_models_cases.py 2020-08-03 2020-08-29 --out_dir evaluations
python ../summarize_evaluations.py --weeks_ahead 4 --evaluations_dir evaluations --out_dir summary --summarize_counties
```

All formats and columns in the [`evaluations`](/cases/evaluations) directory follow the same pattern as the forecast [evaluations for deaths](/evaluations).

The results are summarized in the [`summary`](/cases/summary) directory.
