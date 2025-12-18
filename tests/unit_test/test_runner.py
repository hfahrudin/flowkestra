
import pytest
from unittest.mock import MagicMock, call
from pathlib import Path
import platform
import subprocess

from flowkestra.runner import Runner
from flowkestra.utils import SSHClient

@pytest.fixture
def workdir():
    return Path("/tmp/test_workdir")

@pytest.fixture
def venv_name():
    return "test_venv"

def test_init_local(workdir, venv_name):
    """Test Runner initialization for local execution."""
    runner = Runner(workdir, venv_name)
    assert runner.workdir == workdir.resolve()
    assert runner.venv_name == venv_name
    assert runner.ssh_client is None
    assert runner.remote_is_windows is None

def test_init_remote(workdir, venv_name, monkeypatch):
    """Test Runner initialization for remote execution."""
    mock_ssh = MagicMock(spec=SSHClient)
    mock_ssh.execute.return_value = ("", "") # for _detect_remote_os
    
    runner = Runner(workdir, venv_name, ssh_client=mock_ssh)
    assert runner.workdir == workdir
    assert runner.ssh_client is not None
    assert runner.remote_is_windows is False # default mock return is not windows

def test_detect_remote_os_windows(workdir, monkeypatch):
    """Test remote OS detection for Windows."""
    mock_ssh = MagicMock(spec=SSHClient)
    mock_ssh.execute.return_value = ("Microsoft Windows [Version 10.0.19042.1165]", "")
    runner = Runner(workdir, ssh_client=mock_ssh)
    assert runner.remote_is_windows is True
    mock_ssh.execute.assert_called_with("ver", suppress_output=True)

def test_detect_remote_os_linux(workdir, monkeypatch):
    """Test remote OS detection for Unix-like systems."""
    mock_ssh = MagicMock(spec=SSHClient)
    mock_ssh.execute.side_effect = Exception("command not found")
    runner = Runner(workdir, ssh_client=mock_ssh)
    assert runner.remote_is_windows is False

def test_get_venv_paths_local_linux(workdir, monkeypatch):
    """Test venv paths for local Linux."""
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    runner = Runner(workdir)
    venv_path = workdir.resolve() / runner.venv_name
    assert runner._get_venv_python() == venv_path / "bin" / "python"
    assert runner._get_pip() == venv_path / "bin" / "pip"

def test_get_venv_paths_local_windows(workdir, monkeypatch):
    """Test venv paths for local Windows."""
    monkeypatch.setattr(platform, 'system', lambda: 'Windows')
    runner = Runner(workdir)
    venv_path = workdir.resolve() / runner.venv_name
    assert runner._get_venv_python() == venv_path / "Scripts" / "python.exe"
    assert runner._get_pip() == venv_path / "Scripts" / "pip.exe"

def test_setup_environment_local(workdir, monkeypatch):
    """Test environment setup for local execution."""
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, 'run', mock_run)
    monkeypatch.setattr(Path, 'exists', lambda self: False)
    mock_mkdir = MagicMock()
    monkeypatch.setattr(Path, 'mkdir', mock_mkdir)
    
    runner = Runner(workdir)
    requirements_path = "/tmp/requirements.txt"
    runner.setup_environment(requirements_path)
    
    mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    assert mock_run.call_count > 2

def test_setup_environment_remote(workdir, monkeypatch):
    """Test environment setup for remote execution."""
    mock_ssh = MagicMock(spec=SSHClient)
    mock_ssh.execute.return_value = ("", "")
    runner = Runner(workdir, ssh_client=mock_ssh)
    requirements_path = "/remote/requirements.txt"

    runner.setup_environment(requirements_path)

    expected_cmds = [
        f"mkdir -p {runner.workdir}",
        f"python3 -m venv {runner.workdir / runner.venv_name}",
        f"{runner._get_pip()} install --upgrade pip",
        f"{runner._get_pip()} install -r {requirements_path}"
    ]
    calls = [call(cmd, suppress_output=runner.suppress_output) for cmd in expected_cmds]
    mock_ssh.execute.assert_has_calls(calls, any_order=False)

def test_run_script_local(workdir, monkeypatch):
    """Test script execution for local runner."""
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, 'run', mock_run)
    
    runner = Runner(workdir, suppress_output=False)
    script_path = "/tmp/script.py"
    args = ["arg1", "arg2"]
    
    runner.run_script(script_path, args)
    
    assert mock_run.called

def test_run_script_remote(workdir, monkeypatch):
    """Test script execution for remote runner."""
    mock_ssh = MagicMock(spec=SSHClient)
    mock_ssh.execute.return_value = ("output", "error")
    runner = Runner(workdir, ssh_client=mock_ssh, suppress_output=False)
    script_path = "/remote/script.py"
    args = ["arg1", "arg2"]

    out, err = runner.run_script(script_path, args)
    
    assert out == "output"
    assert err == "error"
    assert mock_ssh.execute.called
