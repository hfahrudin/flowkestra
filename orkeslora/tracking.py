import mlflow
import os

class MLflowTracking:
    def __init__(self, tracking_uri=None, experiment_name="Default"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        mlflow.log_artifact(local_path, artifact_path)

    def set_tag(self, key, value):
        mlflow.set_tag(key, value)
