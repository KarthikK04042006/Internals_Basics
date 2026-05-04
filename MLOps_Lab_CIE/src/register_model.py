import mlflow
import json

model_name = "freshbasket-delivery-time-min-predictor"

# Get experiment
experiment = mlflow.get_experiment_by_name("freshbasket-delivery-time-min")

# Get runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Take best run (first row)
run_id = runs.iloc[0]["run_id"]

# Register model
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    model_name
)

version = result.version

# FIXED LINE HERE 👇
rmse_value = runs.iloc[0]["metrics.rmse"]

# Save JSON
output = {
    "registered_model_name": model_name,
    "version": int(version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": float(rmse_value)
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed ✅")