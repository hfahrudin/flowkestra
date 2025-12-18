
import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
import yaml

from flowkestra.supervisor import Supervisor
from flowkestra.schema import ConfigSchema

@pytest.fixture
def mock_supervisor(monkeypatch):
    # Mock ConfigSchema to return a valid dictionary
    mock_config_instance = MagicMock()
    mock_config_instance.model_dump.return_value = {
        'mlflow_uri': 'http://test-mlflow:5000',
        'instances': [{'mode': 'local', 'workdir': '/tmp', 'target_workdir': '/tmp/target', 'requirements': 'req.txt', 'pipelines': {'p1': {'script': 's.py'}}}]
    }
    monkeypatch.setattr('flowkestra.supervisor.ConfigSchema', lambda **kwargs: mock_config_instance)

    # Mock requests.get to simulate a live MLflow server
    mock_get = MagicMock()
    mock_get.return_value.status_code = 200
    monkeypatch.setattr('flowkestra.supervisor.requests.get', mock_get)

    # Mock multiprocessing Manager
    mock_manager_instance = MagicMock()
    mock_worker_state = MagicMock()
    mock_manager_instance.dict.return_value = mock_worker_state
    monkeypatch.setattr('flowkestra.supervisor.multiprocessing.Manager', lambda: mock_manager_instance)

    # Mock uuid to have predictable worker IDs
    monkeypatch.setattr('flowkestra.supervisor.uuid.uuid4', lambda: "test_uuid")
    
    # Mock worker
    monkeypatch.setattr('flowkestra.supervisor.Worker', MagicMock())

    # Mock file open
    m = mock_open(read_data=yaml.dump({
        'mlflow_uri': 'http://test-mlflow:5000',
        'instances': [{'mode': 'local', 'workdir': '/tmp', 'target_workdir': '/tmp/target', 'requirements': 'req.txt'}]
    }))
    monkeypatch.setattr('builtins.open', m)

    # Mock ThreadPoolExecutor to run sequentially
    class SequentialExecutor:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def submit(self, fn, *args, **kwargs):
            future = MagicMock()
            future.done.return_value = True
            future.result.return_value = fn(*args, **kwargs)
            return future
            
    monkeypatch.setattr('flowkestra.supervisor.ThreadPoolExecutor', SequentialExecutor)
    
    supervisor = Supervisor(config_path="dummy/path.yml", visualize_progress=False)
    supervisor.worker_state = mock_worker_state
    return supervisor

def test_init(mock_supervisor):
    """Test Supervisor initialization."""
    assert mock_supervisor.config is not None
    assert mock_supervisor.mlflow_uri == "http://test-mlflow:5000"
    mock_supervisor.worker_state.__setitem__.assert_called_with('test_uuid', ANY)

@patch('flowkestra.supervisor.multiprocessing.Process')
@patch('flowkestra.supervisor.threading.Thread')
def test_run_all(MockThread, MockProcess, mock_supervisor):
    """Test the run_all method."""
    # Setup a worker in the state
    mock_worker = MagicMock()
    mock_worker.ssh_client = None # For local worker
    mock_supervisor.worker_state.__getitem__.return_value = {'obj': mock_worker}
    mock_supervisor.worker_state.items.return_value = [('test_uuid', {'obj': mock_worker})]

    mock_supervisor.run_all()

    MockProcess.assert_called_once_with(target=mock_worker.run, name='test_uuid')
    MockProcess.return_value.start.assert_called_once()
    MockProcess.return_value.join.assert_called_once()
    
    # The monitor thread should also be started and joined
    assert MockThread.call_count == 1
    MockThread.return_value.start.assert_called_once()
    MockThread.return_value.join.assert_called_once()

def test_load_config(mock_supervisor, monkeypatch):
    """Test config loading and validation."""
    m = mock_open(read_data=yaml.dump({'instances': []}))
    monkeypatch.setattr('builtins.open', m)
    
    mock_schema_instance = MagicMock()
    mock_schema_instance.model_dump.return_value = {'instances': []}
    monkeypatch.setattr('flowkestra.supervisor.ConfigSchema', lambda **kwargs: mock_schema_instance)
    
    config = mock_supervisor._load_config('dummy.yml')
    assert 'instances' in config
