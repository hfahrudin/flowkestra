import yaml
from flowkestra.remote import RemoteTrainer
from flowkestra.local import LocalTrainer
import requests

class TrainingManager:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mlflow_uri = self.config.get("mlflow_uri")
        self.check_mlflow_server()
        self.instances = self.config.get("instances", [])
        if not self.instances:
            raise ValueError("No instances defined in YAML under 'instances:'")

        self.trainers = []

        for instance_conf in self.instances:
            mode = instance_conf.get("mode", "local")
            venv_name = instance_conf.get("venv_name", "flowkestra_env")
            requirements = instance_conf.get("requirements", "requirements.txt")

            if mode == "local":
                trainer = LocalTrainer(
                    workdir=instance_conf.get("workdir", "./training"),
                    venv_name=venv_name,
                    requirements=requirements,
                    mlflow_uri=instance_conf.get("mlflow_uri", self.mlflow_uri)
                )
            elif mode == "remote":
                remote_conf = instance_conf["remote"]
                trainer = RemoteTrainer(
                    hostname=remote_conf["hostname"],
                    username=remote_conf["username"],
                    password=remote_conf.get("password"),
                    key_filename=remote_conf.get("key_filename"),
                    port=remote_conf.get("port", 22),
                    remote_workdir=remote_conf.get("remote_workdir", "/home/user/training"),
                    venv_name=venv_name,
                    requirements=requirements,
                    mlflow_uri=instance_conf.get("mlflow_uri", self.mlflow_uri)
                )
            else:
                raise ValueError(f"Invalid mode '{mode}' for instance {instance_conf.get('name')}")

            self.trainers.append((instance_conf, trainer))

    def setup_all(self):
        """Set up environment for all instances."""
        for conf, trainer in self.trainers:
            print(f"\n🧩 Setting up instance: {conf.get('name')}")
            trainer.setup_environment()

    def run_all(self, script_path):
        """Run ETL and training on each instance."""
        for conf, trainer in self.trainers:
            print(f"\n🚀 Running instance: {conf.get('name')}")
            etl_script = conf.get("etl_script")
            train_script = conf.get("training_script")
            dataset_path = conf.get("dataset_path")
            env_vars = conf.get("env_vars", {})

            if etl_script:
                trainer.deploy_script(etl_script)
                trainer.run_training(etl_script, env_vars)
            if train_script:
                trainer.deploy_script(train_script)
                trainer.run_training(train_script, env_vars)
            if dataset_path:
                trainer.deploy_dataset(dataset_path)
    def close_all(self):
        for _, trainer in self.trainers:
            trainer.close()

    def check_mlflow_server(self, timeout=5):
        """Check if MLflow server root URL is reachable."""
        if not self.mlflow_uri:
            return  # nothing to check

        uri_check = self.mlflow_uri+"/#/experiments"
        try:
            response = requests.get(uri_check, timeout=timeout)
            if response.status_code == 200:
                print(f"✓ MLflow server at {self.mlflow_uri} is reachable.")
            else:
                raise ConnectionError(
                    f"MLflow server responded with status code {response.status_code}"
                )
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to MLflow at {self.mlflow_uri}: {e}")