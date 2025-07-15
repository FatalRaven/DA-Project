import os
import sys
import json
from datetime import datetime
import traceback

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set path for storing prediction run JSON outputs.
EXPERIMENT_NAME = "Burst_Pressure_Modeling"

PREDRUN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predrun"))

# Set dir for storing prediction outputs.
PRED_LOG_DIR = os.path.abspath(os.path.join("predrun"))
os.makedirs(PRED_LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_file = os.path.join(PRED_LOG_DIR, f"prediction_{timestamp}.json")

# Load latest model with error handling and run model.
def load_latest_model(experiment_name: str):
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model, run.info.run_id


def main():
    input_data = pd.DataFrame([{
        "balloon_dia_mm": 12,
        "material_thickness_mm": 0.2,
        "stretch_ratio": 1.3,
        "polymer_modulus_mpa": 2.21,
        "bond_temp_C": 120,
        "cooling_rate_Cps": 3.5
    }])

    try:
        model, run_id = load_latest_model(EXPERIMENT_NAME)
        prediction = model.predict(input_data)[0]
        print(f"Predicted burst pressure (Run ID {run_id}): {prediction:.2f} psi")

        payload = {
            "input": input_data.to_dict(orient="records")[0],
            "prediction": float(prediction)
        }

        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Prediction saved to {output_file}")

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
