## Evaluation of Incident Case Forecasts

We present evaluations for weekly incident case forecasts for the US, state-by-state, and county-by-county.

The methodology for computing errors for incident case is straightforward: we simply compute the incident weekly cases and compare them to the [latest truth data](https://github.com/reichlab/covid19-forecast-hub/blob/master/data-truth/truth-Incident%20Cases.csv).

We are only evaluating models with forecasts for at least 40 states or 1000 counties. For models with missing state or county projections, we substitute the missing error with the mean absolute error for that state/county (among all the models with available forecasts).

All formats and columns in the [`evaluations`](/cases/evaluations) directory follow the same pattern as the forecast [evaluations for deaths](/evaluations).

The results are summarized in the [`summary`](/cases/summary) directory.
