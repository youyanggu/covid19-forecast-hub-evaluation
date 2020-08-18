## Global Evaluation of Country-by-country Forecasts

In this section, we present evaluations of country-by-country forecasts by [covid19-projections.com](https://covid19-projections.com) and [IHME](https://covid19.healthdata.org/). Because global forecasts are not standardized, each team's forecasts must be downloaded and processed separately. We compare the models with the [baseline metric](https://github.com/youyanggu/covid19-forecast-hub-evaluation#baseline-model). More teams will be added to the evaluation in the future.

*Note: In the global evaluation, the baseline metric has look-ahead bias in the sense that it uses the latest data to compute the baseline. This data may not have been available at the time that the forecasts were generated. The model forecasts do not have this look-ahead bias.*

The files are located in the directory based on the evaluation date (e.g. all projections evaluated on Saturday, June 13 cumulative deaths are located in the `2020-06-13` directory). The file format is the following:

```projections_{proj-date}_{eval-date}.csv```

where

* `proj-date` is the date the projections were generated (Mondays)
* `eval-date` is the date the projections were evaluated (Saturdays)

In each projections file, you will find the following columns:

* `actual_deaths`: The actual number of cumulative deaths on the evaluation date
* `Baseline`: The number of deaths if we use the [baseline metric](https://github.com/youyanggu/covid19-forecast-hub-evaluation#baseline-model) of using the previous week's average daily deaths to make all future forecasts (this is a costant number)
* `{MODEL_NAME}`: Each model's cumulative deaths projections for the evaluation date
* `error`: The error from each team's projection subtracted by the actual deaths
* `beat_baseline`: Whether the team's projection is closer to the actual deaths than the baseline metric (`abs(error-TEAM) < abs(error-Baseline)`)

The results are summarized in the [`summary`](summary) directory.
