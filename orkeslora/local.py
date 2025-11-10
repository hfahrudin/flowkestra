import os
import subprocess
import shutil

class LocalTrainer:
    """
    Class to run ML training scripts locally with MLflow tracking.
    """
    def __init__(self, workdir="./training", mlflow_uri=None):
        """
        Args:
            workdir (str): Local working directory for training scripts and artifacts
            mlflow_uri (str, optional): MLflow tracking URI (local or remote)
        """
        self.workdir = workdir
        self.mlflow_uri = mlflow_uri
        os.makedirs(self.workdir, exist_ok=True)

    def deploy_script(self, local_script_path):
        """
        Copy training script to working directory.
        """
        remote_path = os.path.join(self.workdir, os.path.basename(local_script_path))
        shutil.copy(local_script_path, remote_path)
        print(f"Script deployed to {remote_path}")
        return remote_path

    def run_training(self, script_path, additional_env=None):
        """
        Execute training script locally.
        
        Args:
            script_path (str): Path to the training script
            additional_env (dict, optional): Additional environment variables
        """
        env = os.environ.copy()
        if self.mlflow_uri:
            env["MLFLOW_TRACKING_URI"] = self.mlflow_uri
        if additional_env:
            env.update(additional_env)

        print(f"Running training script: {script_path}")
        process = subprocess.Popen(
            ["python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout, stderr = process.communicate()

        print("=== Training Output ===")
        print(stdout.decode())
        if stderr:
            print("=== Training Errors ===")
            print(stderr.decode())

    def download_artifact(self, source_path, target_path):
        """
        Copy trained model or artifact to a specific location.
        """
        shutil.copy(source_path, target_path)
        print(f"Artifact copied to {target_path}")
