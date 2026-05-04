import pandas as pd
import numpy as np
import json
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Starting training...")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("data/training_data.csv")
X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate(model):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    return mae, rmse, r2, mape

# -------------------------
# MLflow Setup
# -------------------------
mlflow.set_experiment("freshbasket-delivery-time-min")

results = []

# -------------------------
# SVR MODEL
# -------------------------
with mlflow.start_run(run_name="SVR"):
    svr = SVR()
    svr.fit(X_train, y_train)

    mae, rmse, r2, mape = evaluate(svr)

    mlflow.log_param("model", "SVR")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.set_tag("domain", "grocery_delivery")

    # 🔥 THIS WAS MISSING BEFORE
    mlflow.sklearn.log_model(svr, "model")

    results.append({
        "name": "SVR",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    })

    joblib.dump(svr, "models/svr_model.pkl")

# -------------------------
# RANDOM FOREST MODEL
# -------------------------
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    mae, rmse, r2, mape = evaluate(rf)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.set_tag("domain", "grocery_delivery")

    # 🔥 THIS WAS MISSING BEFORE
    mlflow.sklearn.log_model(rf, "model")

    results.append({
        "name": "RandomForest",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    })

    joblib.dump(rf, "models/rf_model.pkl")

# -------------------------
# BEST MODEL SELECTION
# -------------------------
best_model = min(results, key=lambda x: x["rmse"])

if best_model["name"] == "SVR":
    joblib.dump(svr, "models/best_model.pkl")
else:
    joblib.dump(rf, "models/best_model.pkl")

# -------------------------
# SAVE JSON OUTPUT
# -------------------------
output = {
    "experiment_name": "freshbasket-delivery-time-min",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best_model["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed ✅")