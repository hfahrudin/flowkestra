import yaml
from orkeslora.remote import RemoteTrainer
from orkeslora.local import LocalTrainer

class TrainingManager:
    """
    Unified trainer based on YAML configuration.
    """
    def __init__(self, config_path):
        """
        Load configuration from YAML.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config.get("mode", "local")  # "local" or "remote"
        self.mlflow_uri = self.config.get("mlflow_uri", None)

        if self.mode == "local":
            workdir = self.config.get("workdir", "./training")
            self.trainer = LocalTrainer(workdir=workdir, mlflow_uri=self.mlflow_uri)
        elif self.mode == "remote":
            remote_conf = self.config["remote"]
            self.trainer = RemoteTrainer(
                hostname=remote_conf["hostname"],
                username=remote_conf["username"],
                password=remote_conf.get("password"),
                key_filename=remote_conf.get("key_filename"),
                port=remote_conf.get("port", 22),
                remote_workdir=remote_conf.get("remote_workdir", "/home/user/training"),
                mlflow_uri=self.mlflow_uri
            )
        else:
            raise ValueError("Invalid mode in YAML. Use 'local' or 'remote'.")

    def setup(self):
        """Setup working environment if needed (remote folder, etc.)"""
        if self.mode == "remote":
            self.trainer.setup_remote_dir()

    def deploy_script(self, script_path):
        return self.trainer.deploy_script(script_path)

    def run_training(self, script_path, additional_env=None):
        self.trainer.run_training(script_path, additional_env)

    def download_artifact(self, source_path, target_path):
        self.trainer.download_artifact(source_path, target_path)

    def close(self):
        if self.mode == "remote":
            self.trainer.close()