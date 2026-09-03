# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import cross_val_score, TimeSeriesSplit,  KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%

# Loading data
price_df = pd.read_csv('data/price.csv') # Day-ahead prices for German market
productions_df = pd.read_csv('data/production.csv') # Hourly actual energy productions for different areas
consumption_df = pd.read_csv('data/consumption.csv') # Actual hourly consumption 
net_import_df = pd.read_csv('data/net_imports.csv') # Net physical electricity flows from neighbouring countries to Germany (import - export)
costs_df = pd.read_csv('data/costs.csv') # Clean spark spread and clean dark spread
weather_df = pd.read_csv('data/weather.csv') # Average country temperature and wind speed

# Rename columns for consistency
net_import_df.rename(columns={'Date': 'datetime'}, inplace=True)
costs_df.rename(columns={'date' : 'datetime'}, inplace=True)

# Convert 'datetime' columns to datetime format
price_df['datetime'] = pd.to_datetime(price_df['datetime'])
productions_df['datetime'] = pd.to_datetime(productions_df['datetime'])
consumption_df['datetime'] = pd.to_datetime(consumption_df['datetime'])
net_import_df['datetime'] = pd.to_datetime(net_import_df['datetime'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
costs_df['datetime'] = pd.to_datetime(costs_df['datetime'])

# Check the date ranges
print("Price data range:", price_df['datetime'].min(), "to", price_df['datetime'].max())
print("Production data range:", productions_df['datetime'].min(), "to", productions_df['datetime'].max())
print("Consumption data range:", consumption_df['datetime'].min(), "to", consumption_df['datetime'].max())
print("Net import data range:", net_import_df['datetime'].min(), "to", net_import_df['datetime'].max())
print("Weather data range:", weather_df['datetime'].min(), "to", weather_df['datetime'].max())
print("Costs data range:", costs_df['datetime'].min(), "to", costs_df['datetime'].max())

# Merge all dataframes on 'datetime'
merged_df = price_df.copy()
merged_df = (merged_df
             .merge(productions_df, on='datetime', how='outer')
             .merge(consumption_df, on='datetime', how='outer')
             .merge(net_import_df, on='datetime', how='outer')
             .merge(weather_df, on='datetime', how='outer')
             .merge(costs_df, on='datetime', how='outer'))

# Production-demand imbalance
merged_df['production_demand_imbalance'] = (
    merged_df['Solar'] +
    merged_df['WindOnshore'] +
    merged_df['WindOffshore'] +
    merged_df['NaturalGas'] +
    merged_df['HardCoal'] +
    merged_df['Ror'] +
    merged_df['Dam'] +
    merged_df['Nuclear']
) - merged_df['consumption']


plt.figure(figsize=(15, 7))
plt.plot(costs_df['datetime'], costs_df['Clean Spark Spread'], label='Clean Spark Spread')
plt.plot(costs_df['datetime'], costs_df['Clean Dark Spread'], label='Clean Dark Spread')
plt.xlabel('Datetime')
plt.ylabel('Costs')
plt.title('Imbalance')
plt.legend()
plt.show()

costs_df.head()

# Define the training and testing periods
train_start_date = '2019-01-01 00:00:00'
train_end_date = '2022-03-25 23:00:00'

test_start_date = '2022-03-26 00:00:00'
test_end_date = '2022-04-30 23:00:00'

req_start_date = '2022-05-01 00:00:00'
req_end_date = '2022-06-30 23:00:00'

# Split the data
train_df = merged_df[(merged_df['datetime'] >= train_start_date) & (merged_df['datetime'] <= train_end_date)].copy()
test_df = merged_df[(merged_df['datetime'] >= test_start_date) & (merged_df['datetime'] <= test_end_date)].copy()
pred_df = merged_df[(merged_df['datetime'] >= req_start_date) & (merged_df['datetime'] <= req_end_date)].copy()

# Drop rows with NaNs in the training set
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Features and target for training
X_train = train_df.drop(columns=['price', 'datetime'])
y_train = train_df['price']

# Features and target for testing
X_test = test_df.drop(columns=['price', 'datetime'])
y_test = test_df['price']  # Use this only if actual prices are available

# Hyperparameter Optimization with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.9),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 50),
        'random_state': 42
    }
    
    xgb_model = XGBRegressor(**params)
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=8)
    cv_scores = cross_val_score(
        xgb_model, 
        X_train, 
        y_train, 
        cv=tscv, 
        scoring='neg_root_mean_squared_error'
    )
    
    return -cv_scores.mean()

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

# Best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train the model with the best hyperparameters
best_xgb_model = XGBRegressor(**study.best_params, random_state=42)
best_xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best = best_xgb_model.predict(X_test)

# Predict on the required dates 2022-05-01 to 2022-06-30
X_req = pred_df.drop(columns=['price', 'datetime'])
y_pred_req = best_xgb_model.predict(X_req)

# Evaluate the model if y_test is available
if not y_test.isnull().all():
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    mae_best = mean_absolute_error(y_test, y_pred_best)
    print(f'Optimized XGBoost RMSE: {rmse_best}')
    print(f'Optimized XGBoost MAE: {mae_best}')
else:
    print("y_test contains all NaNs. Evaluation metrics cannot be computed.")

# Plot feature importance
plot_importance(best_xgb_model, max_num_features=25)
plt.title('Feature Importance')
plt.show()

# Cross-validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=8)
cv_scores = cross_val_score(
    best_xgb_model, 
    X_train, 
    y_train, 
    scoring='neg_root_mean_squared_error', 
    cv=tscv
)
print(f'Cross-validated RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Add predictions to the test dataframe and plot
test_df['predicted_price'] = y_pred_best
pred_df['predicted_price'] = y_pred_req

# Save to .csv
pred_df[['datetime','predicted_price']].to_csv('predicted_prices.csv', index=False)

# Predicted prices
plt.figure(figsize=(15, 7))
plt.plot(pred_df['datetime'], pred_df['predicted_price'], label='Predicted prices')
plt.xlabel('Datetime')
plt.ylabel('Energy Price')
plt.title('Predicted Energy Prices (May - June 2022)')
plt.legend()
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(15, 7))
plt.plot(test_df['datetime'], test_df['price'], label='Actual Price', alpha=0.7)
plt.plot(test_df['datetime'], test_df['predicted_price'], label='Predicted Price', alpha=0.7)
plt.xlabel('Datetime')
plt.ylabel('Energy Price')
plt.title('Actual vs Test Energy Prices')
plt.legend()
plt.show()

# Prices over time with predicted values
plt.figure(figsize=(10, 6))
plt.plot(merged_df['datetime'], merged_df['price'], label='Actual prices')
plt.plot(pred_df['datetime'], pred_df['predicted_price'], label='Predicted prices')
plt.title('Price over time including predicted data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

