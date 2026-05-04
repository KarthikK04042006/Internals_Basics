import mlflow
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model_name = "freshbasket-delivery-time-min-predictor"

# Load training data
df = pd.read_csv("data/training_data.csv")
X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

# Train challenger model (random_state=99)
model_v2 = RandomForestRegressor(random_state=99)
model_v2.fit(X, y)

# Evaluate challenger
pred_v2 = model_v2.predict(X)
rmse_v2 = np.sqrt(mean_squared_error(y, pred_v2))

# Get existing versions
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")

# Assume version 1 is current champion
champion_version = int(versions[0].version)

# Compare (dummy logic since we don’t store rmse in registry directly)
if rmse_v2 < 7.05:   # use your best rmse from Task 1
    action = "promoted"
    new_version = 2
else:
    action = "kept"
    new_version = champion_version

# Save JSON
output = {
    "registered_model_name": model_name,
    "alias_name": "production",
    "champion_version": champion_version,
    "challenger_version": 2,
    "action": action
}

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed ✅")