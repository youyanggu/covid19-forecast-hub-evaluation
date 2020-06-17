# Evaluation of COVID-19 Models

Here we present an evaluation of models from the [COVID-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub). These models are submitted weekly to the [CDC COVID-19 Forecasting page](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html) to help inform public health decision-making.

While a model's future projections can be useful, it is also important to take into account the model's historical performance in a transparent, rigorous, and non-biased manner. This is the goal of this project.

You can view the outputs of the evaluations in the [evaluations](/evaluations) directory.

## Dependencies

### Python

You need Python 3.7+ with the [NumPy](https://numpy.org/install/) and [pandas](https://pandas.pydata.org/getting_started.html) packages.

Once you have Python 3, you can simply install the NumPy and pandas packages by running the following command:
```
pip install numpy pandas
```

### Data

You also need to download the [COVID-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub) data. We recommend cloning the repository:
```
git clone https://github.com/reichlab/covid19-forecast-hub.git
```

## Getting Started
1. Make sure the dependencies (NumPy/pandas) are installed: `pip install numpy pandas`
2. Clone this repository: `git clone https://github.com/youyanggu/covid19-forecast-hub-evaluation.git`
3. Clone the [COVID-19 Forecast Hub](https://github.com/reichlab/covid19-forecast-hub) repository: `git clone https://github.com/reichlab/covid19-forecast-hub.git`
4. (Optional) Make sure that the COVID-19 Forecast Hub and this repository share the same parent. Otherwise, you would need to pass in the Forecast Hub location via the `--forecast_hub_dir` flag.
5. Pick a Monday to choose the projections, and pick a Saturday to evaluate those Monday projections (e.g. Mon May 4 and Sat June 13). We explain the reasoning for this [below](#methods).
6. Run the evaluation: `python evaluate_models.py 2020-05-04 2020-06-13`

## Usage

### Evaluate Mon May 4 projections on Sat June 13 data:
```
python evaluate_models.py 2020-05-04 2020-06-13
```

### Custom COVID-19 Forecast Hub directory
```
python evaluate_models.py 2020-05-04 2020-06-13 --forecast_hub_dir /PATH/TO/covid19-forecast-hub
```

### Save evaluation results to a directory:
```
python evaluate_models.py 2020-05-04 2020-06-13 --out_dir evaluations/
```

### Use median projections rather than point projections
(For most models, this makes no difference)
```
python evaluate_models.py 2020-05-04 2020-06-13 --use_median
```

### Print additional statistics such as mean rank and residual analysis
```
python evaluate_models.py 2020-05-04 2020-06-13 --print_additional_stats
```

## Methods

### Overview

Models in the COVID-19 Forecast Hub submit their forecasts every Monday to be sent to the CDC. To be declared valid, a model must have 1-4 week ahead forecasts. For these week-ahead forecasts, all models use the specification of [epidemiological weeks (epiweek)](https://wwwn.cdc.gov/nndss/document/MMWR_Week_overview.pdf) defined by the CDC. Because each epiweek ends on Saturday, all models are also evaluated on Saturdays. For example, if a model submits a forecast on Monday, June 15, a one week ahead forecast corresponds to the forecast ending on Saturday, June 20. A two week ahead forecast corresponds to Saturday, June 27, and so forth. This is explained in more detail in the [COVID-19 Forecast Hub README](https://github.com/reichlab/covid19-forecast-hub/#covid-19-forecast-hub).

To summarize, due to the reason above, we standardize all projection dates to be on Mondays and all evaluation dates to be on Saturdays.

### Truth Data

As described in the [COVID-19 Forecast Hub README](https://github.com/reichlab/covid19-forecast-hub/tree/master/data-processed#ground-truth-data), all forecasts are compared to the [Johns Hopkins University CSSE Time Series Summary](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series) as the gold standard reference data for deaths in the US. This truth data can be found in the [Forecast Hub data-truth directory](https://github.com/reichlab/covid19-forecast-hub/tree/master/data-truth).

### US Evaluation

For US country-wide forecasts, we compute the error for each model's point forecasts using the following formula:
```
additional_deaths = true_deaths_eval_date - true_deaths_proj_date
error_us = projected_deaths_eval_date - true_deaths_eval_date
perc_error_us = error_us / additional_deaths
```

So for example, if our projection date is Mon June 1 and our evaluation date is Sat June 13, below is a sample of how we compute the error.

| Example | |
| --- | --- |
| June 1 US true deaths | 105,430 |
| June 13 US true deaths | 115,436 |
| Additional deaths | 10,006 (115,436 - 105,430) |
| Sample model forecast for June 13 | 115,581 |
| Error | 145 (115,581 - 115,436) |
| % Error | 1.4% (145 / 10,006) |

### State-by-state Evaluation

In addition to US country-wide forecasts, we also evaluate state-by-state forecasts for all 50 states plus DC. In this section, we only evaluate models that have 1-4 week ahead forecasts for more than 40 states.

Using the point forecast for every model, we compute the error for each state by the following:

```error_state = projected_deaths_eval_date - true_deaths_eval_date```

For example, if a state reports 2,462 deaths on June 13 and the model's forecasted total for June 13 is 2,420, then our error for that state is -42.

We then compute the mean absolute error and mean squared error for all states for every model. For models with missing state projections, we substitute the error with the mean absolute error for that state (among all the models).

### Baseline Model

In any evaluation, it is important to include a baseline as a control, similar to how scientfic trials include a placebo. We define a simple baseline model that takes the mean of the previous week's daily deaths to make all future forecasts. For example, for Monday, May 25 projections, we use the average daily deaths from May 18 to May 24 to make forecasts. For US country-wide projections, this would amount to a constant 1,164 deaths per day for each forecast day.

We also include another baseline model that takes mean of the previous week's daily deaths and decrease that by 2% each day for future projections. This is in general a much more accurate model.

### COVIDHub Ensemble Model

The [COVIDhub ensemble model](https://github.com/reichlab/covid19-forecast-hub/#ensemble-model) is created by taking a combination of all eligible models that submit projections. You can see which models are included and their corresponding weights [here](https://github.com/reichlab/covid19-forecast-hub/tree/master/ensemble-metadata).

## Questions? Bugs? Feature Request?

Lastly, we encourage open collaboration. Please open [an issue request](https://github.com/youyanggu/covid19-forecast-hub-evaluation/issues) if you have any questions/suggestions/bug reports.
