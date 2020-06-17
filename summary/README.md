# Summary of Evaluations

Here is where we summarize the outputs from the [evaluations](/evaluations) directory. Here, we present summaries for two types of evaluations:

- N weeks ahead: we evaluate historical forecasts based on their forecasts N weeks ahead
- Evaluation date: we evaluate all historical forecasts based on their forecasts for the specified evaluation date

The individual column names in each file are in the format:

`eval-type_proj-date_eval-date`

where

* `proj-date` is the date the projections were generated (Mondays)
* `eval-date` is the date the projections were evaluated (Saturdays)
* `eval-type` is the type of evaluation:
  * `perc_error`: percent error based on additional deaths since the projection date
  * `mean_abs_errs` : State-by-state mean absolute errors
  * `mean_sq_errs` : State-by-state mean squared errors
