import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()

    # Uncomment and customize the following lines to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect experiment runs within the mlflow UI.\n"
        "Find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()
