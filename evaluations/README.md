## Historical Evaluations

Here is where we save all the outputs of our evaluations. You can see a full list of team/model names [here](https://github.com/reichlab/covid19-forecast-hub#teams-and-models). The files are located in the directory based on the evaluation date (e.g. all projections evaluated on Saturday, June 13 are located in the `2020-06-13` directory). The file format is the following:

```{proj-date}_{eval-date}_{eval-type}.csv```

where

* `proj-date` is the date the projections were generated (Mondays)
* `eval-date` is the date the projections were evaluated (Saturdays)
* `eval-type` is the type of evaluation:
  * `us_errs`: US errors
  * `abs_errs` : State-by-state mean absolute errors
  * `sq_errs` : State-by-state mean squared errors
  * `mean_ranks` : State-by-state mean rankings. For each state, we rank all the models and take their mean ranks (1=highest ranking).

For the `projections-{proj-date}_{eval-date}.csv` files, you will find the following columns:

* `actual_deaths`: The actual number of cumulative deaths on the evaluation date
* `Baseline`: The number of deaths if we use the [baseline metric](https://github.com/youyanggu/covid19-forecast-hub-evaluation#baseline-model) of using the previous week's average daily deaths to make all future forecasts (this is a costant number)
* `{MODEL_NAME}`: Each model's cumulative deaths projections for the evaluation date
* `error`: The error from each team's projection subtracted by the actual deaths
* `beat_baseline`: Whether the team's projection is closer to the actual deaths than the baseline metric (`abs(error-TEAM) < abs(error-Baseline)`)

These evaluations are subsequently summarized in the [summary](/summary) directory.
