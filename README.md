# German Day-Ahead Electricity Price Forecasting

This project explores the forecasting of German day-ahead electricity prices using machine learning.

The goal is to predict electricity prices from historical market and fundamental data, combining information on energy production, consumption, cross-border electricity flows, generation costs, and weather conditions.

## Project Overview

The model is built using `XGBoost`, with hyperparameter optimization performed using `Optuna`.

The workflow is:

1. Load and clean the different datasets.
2. Align all variables on a common hourly time index.
3. Create a modelling dataset containing electricity prices and explanatory variables.
4. Split the data into training and testing periods.
5. Tune the XGBoost hyperparameters.
6. Train the final model.
7. Evaluate its forecasting performance using RMSE and MAE.
8. Inspect feature importance and compare predictions with observed electricity prices.

## Data

The project uses several datasets related to the German electricity market.

### Day-ahead electricity prices

Historical German day-ahead electricity prices are used as the prediction target.

### Electricity production

Hourly electricity production data from different generation sources are included as explanatory variables.

These variables provide information about the composition and availability of electricity supply.

### Electricity consumption

Hourly electricity consumption is used to represent demand conditions in the German electricity market.

### Net imports

Cross-border electricity flows between Germany and neighbouring countries are included as net imports:

```text
net imports = imports - exports
```

Positive values therefore indicate that Germany is a net importer of electricity, while negative values indicate net exports.

### Generation costs

The dataset also includes:

* Clean Spark Spread
* Clean Dark Spread

These variables provide information about the relative economics of gas-fired and coal-fired electricity generation.

### Weather

Average weather conditions are included through variables such as:

* temperature
* wind speed

Weather can affect both electricity demand and renewable electricity production.

## Data Preparation

The individual datasets cover slightly different time periods and therefore need to be aligned before modelling.

Dates and timestamps are converted into a consistent format, and the datasets are merged into a single dataframe.

The final dataset contains the electricity price as the target variable together with the available market, production, consumption, cost, import, and weather variables.

The data is kept in chronological order throughout the analysis because this is a time-series forecasting problem.

## Model

The forecasting model used in this project is:

```python
XGBRegressor
```

from the `XGBoost` library.

Gradient-boosted decision trees are useful for this problem because they can model nonlinear relationships and interactions between market variables without requiring an explicit functional form.

For example, the effect of electricity demand on price may depend simultaneously on available generation, fuel costs, imports, and weather conditions.

## Hyperparameter Optimization

The model hyperparameters are tuned using `Optuna`.

The search includes parameters such as:

* number of estimators
* learning rate
* maximum tree depth
* subsampling ratio
* feature subsampling ratio
* gamma

The objective of the optimization is to minimize the Root Mean Squared Error (RMSE).

## Model Evaluation

The main evaluation metrics are:

### Root Mean Squared Error

RMSE measures the typical size of prediction errors while giving greater weight to large errors.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

This is particularly relevant for electricity prices because occasional large price movements can be important.

### Mean Absolute Error

MAE measures the average absolute difference between predicted and observed prices:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Using both metrics gives a better picture of model performance.

## Feature Importance

After training, XGBoost feature importance is inspected to understand which variables contribute most strongly to the model predictions.

This should not be interpreted as a direct measure of causality. Instead, it provides an indication of which features the model relies on most when constructing its predictions.

## Technologies

The project is implemented in Python using:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `optuna`

## Project Structure

Current project structure is:

```text
.
├── data/
│   ├── price.csv
│   ├── production.csv
│   ├── consumption.csv
│   ├── net_imports.csv
│   ├── costs.csv
│   └── weather.csv
│
├── prediction.py
├── requirements.txt
└── README.md
```

The raw datasets are loaded from the `data/` directory and processed inside the main analysis script.

## Current Scope

This project is intended as an exploration of electricity-price forecasting using fundamental market data and gradient-boosted trees.

The current implementation focuses on:

* combining multiple electricity-market datasets;
* constructing a consistent time-series dataset;
* training an XGBoost regression model;
* optimizing model hyperparameters;
* evaluating forecasting errors;
* examining model feature importance.

Further improvements could include additional time-series features, alternative forecasting models, and a more detailed investigation of model performance during different electricity-market conditions.
