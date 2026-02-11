import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')

#adds neighborhood column
housing["neighborhood"] = housing["id"].astype(str).str[:6]

#adds columns for year, month, day, day of the week
dt = pd.to_datetime(housing["date"], errors="coerce")
housing["year"] = dt.dt.year
housing["month"] = dt.dt.month
housing["day"] = dt.dt.day
housing["dayofweek"] = dt.dt.dayofweek

#adds total_sqft column
housing["total_sqft"] = housing["sqft_above"] + housing["sqft_basement"].fillna(0)


#adds house_age, renovated_flag, reno_age columns
housing["house_age"] = housing["year"] - housing["yr_built"]
housing["renovated_flag"] = (housing["yr_renovated"].fillna(0) > 0).astype(int)
housing["reno_age"] = np.where(
    housing["renovated_flag"] == 1,
    housing["year"] - housing["yr_renovated"],
    0
)

#adds bathrooms_per_bedroom column
housing["bathrooms_per_bedroom"] = housing["bathrooms"] / housing["bedrooms"].replace(0, np.nan)
housing["bathrooms_per_bedroom"] = housing["bathrooms_per_bedroom"].fillna(0)

# drops extreme square footage values to reduce outlier impact
for col in ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "total_sqft"]:
    if col in housing.columns:
        hi = housing[col].quantile(0.995)
        housing = housing[housing[col] <= hi]

# drops extreme bedroom and bathroom counts
for col, hi in [("bedrooms", 7), ("bathrooms", 4)]:
    if col in housing.columns:
        housing = housing[housing[col] <= hi]

housing.head()

target_col = "price"
drop_cols = [target_col, "id", "date"]
X = housing.drop(columns=[c for c in drop_cols if c in housing.columns])
y = np.log1p(housing[target_col])

# One-hot encode the 'neighborhood' column
X = pd.get_dummies(X, columns=['neighborhood'], drop_first=True)

# Split into train (80%) and temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split temp into dev (10%) and test (10%)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model = XGBRegressor(
    n_estimators=5000,        # total number of trees (more trees = more learning capacity, but slower)
    learning_rate=0.03,       # how much each tree contributes (smaller = more gradual learning, usually better generalization)
    max_depth= 8,             # max depth of each tree (controls complexity; deeper = can fit more patterns but risk overfitting)
    subsample=0.8,            # percent of rows used per tree (adds randomness to reduce overfitting)
    colsample_bytree=0.8,     # percent of features used per tree (adds randomness + helps prevent reliance on a few features)
    gamma=0.1,                # minimum loss reduction needed to split a node (higher = fewer splits = less overfitting)
    min_child_weight=10,      # minimum "weight" in a leaf (higher = more conservative splits, helps prevent overfitting)
    reg_alpha=0.0,            # L1 regularization (pushes some feature effects toward 0, can help if too many features)
    reg_lambda=1.0,           # L2 regularization (stabilizes weights, reduces overfitting; default is usually 1)
    random_state=42,          # makes results reproducible (same split/training behavior each run)
    n_jobs=-1                 # uses all CPU cores to train faster
)


# Fit on TRAIN only
model.fit(X_train, y_train)

# Predict on DEV
dev_pred_log = model.predict(X_dev)

# Convert back to dollars
dev_pred_price = np.expm1(dev_pred_log)
y_dev_price = np.expm1(y_dev)

# Metrics on DEV
dev_rmse = root_mean_squared_error(y_dev_price, dev_pred_price)
dev_r2 = r2_score(y_dev_price, dev_pred_price)

print("DEV RMSE (dollars):", dev_rmse)
print("DEV R^2 (dollars):", dev_r2)


