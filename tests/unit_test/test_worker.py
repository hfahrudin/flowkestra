
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from flowkestra.worker import Worker
from flowkestra.schema import SSHConfig
from flowkestra.runner import Runner

@pytest.fixture
def worker_params():
    return {
        "worker_id": "worker_1",
        "workdir": Path("/tmp/test_workdir"),
        "origin_dir": Path("/tmp/origin"),
        "main_states": {"worker_1": {}},
        "requirements": "requirements.txt",
        "pipelines": {"step1": {"script": "script1.py"}}
    }

@patch('flowkestra.worker.Runner')
@patch('flowkestra.worker.shutil')
@patch('flowkestra.worker.Path')
def test_init_local(MockPath, mock_shutil, MockRunner, worker_params, monkeypatch):
    """Test Worker initialization for local execution."""
    mock_runner_instance = MockRunner.return_value
    
    # Mock methods that are called during __init__
    monkeypatch.setattr(Worker, '_clean_workdir', MagicMock())
    monkeypatch.setattr(Worker, '_sync_workdir', MagicMock())

    worker = Worker(**worker_params)

    assert worker_params["main_states"][worker_params["worker_id"]]['status'] == 'ready'
    mock_runner_instance.setup_environment.assert_called_once()
    worker._clean_workdir.assert_called()
    worker._sync_workdir.assert_called()


@patch('flowkestra.worker.Runner')
@patch('flowkestra.worker.SSHClient')
def test_init_remote(MockSSHClient, MockRunner, worker_params, monkeypatch):
    """Test Worker initialization for remote execution."""
    mock_ssh_config = MagicMock(spec=SSHConfig)
    mock_runner_instance = MockRunner.return_value
    mock_ssh_client_instance = MockSSHClient.return_value
    mock_runner_instance.ssh_client = mock_ssh_client_instance

    # Mock methods that are called during __init__
    monkeypatch.setattr(Worker, '_clean_workdir', MagicMock())
    monkeypatch.setattr(Worker, '_sync_workdir', MagicMock())

    worker = Worker(**worker_params, ssh_config=mock_ssh_config)

    assert worker_params["main_states"][worker_params["worker_id"]]['status'] == 'ready'
    mock_runner_instance.setup_environment.assert_called_once()
    worker._clean_workdir.assert_called()
    worker._sync_workdir.assert_called()

@patch('flowkestra.worker.Runner')
def test_run(MockRunner, worker_params, monkeypatch):
    """Test the run method."""
    # Mock methods that are called during __init__
    monkeypatch.setattr(Worker, '_clean_workdir', MagicMock())
    monkeypatch.setattr(Worker, '_sync_workdir', MagicMock())

    mock_runner_instance = MockRunner.return_value
    worker = Worker(**worker_params)
    
    # Reset mocks from init
    mock_runner_instance.reset_mock()
    worker._clean_workdir.reset_mock()

    worker.run()
    
    assert worker_params["main_states"][worker_params["worker_id"]]['status'] == 'completed'
    mock_runner_instance.run_script.assert_called_once()
    worker._clean_workdir.assert_called()
