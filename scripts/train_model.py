import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from sklearn.datasets import make_regression
from sklearn import set_config

# Define path and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw", "simulated_data.csv"))
SEED = 1982

# Load data
df = pd.read_csv(DATA_PATH)
df = shuffle(df, random_state=SEED)

# Experiment tracking and logging
mlflow.set_experiment("Burst_Pressure_Modeling")
mlflow.sklearn.autolog()
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Feature splitting
x = df.drop(columns=["burst_pressure_psi"])
y = df["burst_pressure_psi"]
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED)

# Model training
with mlflow.start_run() as run:
    model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"MLflow run ID: {run.info.run_id}")
print(f"Model trained and logged: R² = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")