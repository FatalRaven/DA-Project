import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Define constants
EXPERIMENT_NAME = 'Burst_Pressure_Modeling'


# Get all mlruns from the experiment in mlflow
def get_all_runs(experiment_name: str):
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    # Search for the first 10 runs and order by start date in descending order
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
        order_by=['start_time DESC']
    )
    return runs


# Initialize empty list to collect runs then extract data from mlflow runs
def summarize_runs(runs):
    records = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'start_time': run.info.start_time,
            'rmse': run.data.metrics.get('root_mean_squared_error', None),
            'mae': run.data.metrics.get('mean_absolute_error'),
            'r2': run.data.metrics.get('r2_score'),
        }

        # Filter hyperparameters relevant to RFG
        run_data.update({
            k: v for k, v in run.data.params.items()
            if k in ['n_estimators', 'max_depth', 'random_state']
        })

        records.append(run_data)

    # Create a dataframe from all summarized runs and sorts
    df = pd.DataFrame(records)
    df.sort_values(by='rmse', ascending=True, inplace=True, na_position='last')
    return df


# Gets all runs and summarizes them in the dataframe
def main():
    runs = get_all_runs(EXPERIMENT_NAME)
    summary_df = summarize_runs(runs)

    print("\nModel Evaluation Summary:")
    print(summary_df.to_string(index=False))

    output_path = os.path.join('reports', 'evaluation_summary.csv')
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\n Summary saved to {output_path}")

if __name__ == '__main__':
    main()