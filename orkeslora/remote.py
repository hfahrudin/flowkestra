import os
from orkeslora.utils import SSHClient  # your SSH wrapper

class RemoteTrainer:
    """
    Class to deploy and run ML training scripts on a remote server with optional MLflow tracking.
    """
    def __init__(self, hostname, username, password=None, key_filename=None, port=22, 
                 remote_workdir="/home/user/training", mlflow_uri=None):
        """
        Args:
            hostname (str): Remote server IP or domain
            username (str): SSH username
            password (str, optional): SSH password
            key_filename (str, optional): Private key file path
            port (int, optional): SSH port
            remote_workdir (str): Remote folder to store training scripts
            mlflow_uri (str, optional): MLflow tracking URI
        """
        self.ssh = SSHClient(hostname, username, password, key_filename, port)
        self.remote_workdir = remote_workdir
        self.mlflow_uri = mlflow_uri

    def setup_remote_dir(self):
        """Create remote working directory."""
        self.ssh.connect()
        self.ssh.execute(f"mkdir -p {self.remote_workdir}")

    def deploy_script(self, local_script_path):
        """Upload local training script to remote server."""
        remote_path = os.path.join(self.remote_workdir, os.path.basename(local_script_path))
        self.ssh.upload(local_script_path, remote_path)
        print(f"Script deployed to {remote_path}")
        return remote_path

    def run_training(self, remote_script_path, additional_env=None):
        """
        Execute training script on remote server.
        
        Args:
            remote_script_path (str): Path of the training script on remote server
            additional_env (dict, optional): Additional environment variables
        """
        env_vars = ""
        if self.mlflow_uri:
            env_vars += f"MLFLOW_TRACKING_URI={self.mlflow_uri} "
        if additional_env:
            env_vars += " ".join(f"{k}={v}" for k, v in additional_env.items())
        
        command = f"{env_vars} python3 {remote_script_path}"
        output, error = self.ssh.execute(command)
        print("=== Training Output ===")
        print(output)
        if error:
            print("=== Training Errors ===")
            print(error)

    def download_artifact(self, remote_path, local_path):
        """Download trained model or artifacts from remote server."""
        self.ssh.download(remote_path, local_path)
        print(f"Artifact downloaded: {local_path}")

    def close(self):
        """Close SSH connection."""
        self.ssh.close()
