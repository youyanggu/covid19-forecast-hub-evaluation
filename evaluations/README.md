# Historical Evaluations

Here is where we save all the outputs of our evaluations. The files are located in the directory based on the evaluation date (e.g. all projections evaluated on Saturday, June 13 are located in the `2020-06-13` directory). The file format is the following:

```proj-date_eval-date_eval-type.csv```

where

* `proj-date` is the date the projections were generated (Mondays)
* `eval-date` is the date the projections were evaluated (Saturdays)
* `eval-type` is the type of evaluation:
  * `us_errs`: US errors
  * `abs_errs` : State-by-state mean absolute errors
  * `sq_errs` : State-by-state mean squared errors
  * `mean_ranks` : State-by-state mean rankings. For each state, we rank all the models and take their mean ranks (1=highest ranking).
