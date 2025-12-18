import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import requests
import os
from flowkestra.supervisor import Supervisor
import random
import secrets
import time
import warnings

def validate_mlflow_results(experiment_name):
    base_url = "http://localhost:5000/api/2.0/mlflow"
    
    # 1. Get Experiment Details
    try:
        response = requests.get(
            f"{base_url}/experiments/get-by-name",
            params={"experiment_name": experiment_name}
        )
        response.raise_for_status()
        experiment_data = response.json()
        experiment_id = experiment_data['experiment']['experiment_id']
        print(f"Found Experiment ID: {experiment_id}")

        # 2. Search for Runs in this experiment
        # This tells you how many times the pipeline actually executed successfully
        runs_response = requests.post(
            f"{base_url}/runs/search",
            json={"experiment_ids": [experiment_id]}
        )
        runs_response.raise_for_status()
        runs_data = runs_response.json()
        
        # 'runs' key might be missing if 0 runs exist
        runs = runs_data.get('runs', [])
        run_count = len(runs)
        
        print(f"Total runs found for '{experiment_name}': {run_count}")
        return run_count

    except requests.exceptions.RequestException as e:
        print(f"MLflow Connection Error: {e}")
        return 0

@pytest.fixture
def project_structure():
    # 1. Randomize experiment name
    timestamp = int(time.time())
    unique_id = secrets.token_hex(4) 
    experiment_name = f"test_{timestamp}_{unique_id}"
    
    base_dir = Path(tempfile.mkdtemp())
    origin_dir = base_dir / "origin"
    origin_dir.mkdir()

    # 2. Randomize number of instances (1 to 4)
    num_instances = random.randint(1, 4)
    
    instances = []
    success_files = []

    # Create common files in origin
    (origin_dir / "requirements.txt").write_text("mlflow")

    # 3. Build randomized instances
    for i in range(num_instances):
        # Each instance needs its own dedicated target directory
        instance_workdir = base_dir / f"local_run_inst_{i}"
        instance_workdir.mkdir()
        
        success_file = instance_workdir / "success_local.txt"
        success_files.append(success_file)

        # Create the script for this instance
        # Using absolute path for success_file so we know exactly where it goes
        run_name = f"local_instance_{i}"
        script_content = f"""
import mlflow
from pathlib import Path
import sys

try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("{experiment_name}")
    with mlflow.start_run(run_name="{run_name}"):
        mlflow.log_metric("instance_id", {i})
    
    Path(r'{success_file}').touch()
except Exception as e:
    print(f"Instance {i} failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        script_name = f"local_script_{i}.py"
        (origin_dir / script_name).write_text(script_content)

        instances.append({
            'mode': 'local',
            'workdir': str(origin_dir),
            'target_workdir': str(instance_workdir),
            'requirements': "requirements.txt",
            'pipelines': {'train': {'script': script_name}}
        })

    config = {
        'mlflow_uri': "http://localhost:5000",
        'experiment_name': experiment_name,
        'visualize_progress': False,
        'instances': instances
    }
    
    config_path = base_dir / "test_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    yield {
        "config_path": config_path,
        "success_files": success_files,
        "experiment_name": experiment_name,
        "num_expected": num_instances
    }
    # shutil.rmtree(base_dir) # Comment this out if you want to inspect files after failure

def is_mlflow_offline():
    """Real check against the local server."""
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        return response.status_code != 200
    except:
        return True

def test_supervisor_randomized_instances(project_structure, capfd):

    if is_mlflow_offline():
        warnings.warn(
            "ENVIRONMENT ISSUE: MLflow server is offline. Skipping E2E validation.", 
            UserWarning
        )
        pytest.skip("MLflow server not found at localhost:5000")

    supervisor = Supervisor(config_path=str(project_structure["config_path"]))
    supervisor.run_all()

    out, err = capfd.readouterr()
    
    assert not err, f"Subprocess errors detected:\n{err}"

    time.sleep(1) # Wait for MLflow DB consistency
    
    resp = requests.get(
        "http://localhost:5000/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": project_structure["experiment_name"]}
    ).json()
    
    exp_id = resp['experiment']['experiment_id']
    
    runs_resp = requests.post(
        "http://localhost:5000/api/2.0/mlflow/runs/search",
        json={"experiment_ids": [exp_id]}
    ).json()
    
    actual_runs = len(runs_resp.get('runs', []))
    expected_runs = project_structure["num_expected"]
    
    print(f"Randomized instances: {expected_runs}, Actual MLflow runs: {actual_runs}")
    assert actual_runs == expected_runs, f"Expected {expected_runs} runs, but found {actual_runs}"


