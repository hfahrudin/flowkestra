
import pytest
import requests
import paramiko
import os
import tempfile
import shutil

from tests.mock_servers.mock_vm import MockVM

@pytest.fixture(scope="session")
def ssh_keys():
    """Pytest fixture to generate SSH keys for the test session."""
    key_dir = tempfile.mkdtemp()
    host_key_path = os.path.join(key_dir, 'mock_host_key')
    client_key_path = os.path.join(key_dir, 'mock_client_key')
    
    os.system(f"ssh-keygen -t ed25519 -f {host_key_path} -N '' > /dev/null 2>&1")
    os.system(f"ssh-keygen -t ed25519 -f {client_key_path} -N '' > /dev/null 2>&1")
    
    yield {
        "host_key": host_key_path,
        "client_key": client_key_path,
        "client_key_pub": f"{client_key_path}.pub"
    }
    
    shutil.rmtree(key_dir)

@pytest.fixture(scope="module")
def mock_vm(ssh_keys):
    """Pytest fixture to start and stop the MockVM."""
    vm = MockVM(
        host_key_path=ssh_keys["host_key"],
        client_pub_key_path=ssh_keys["client_key_pub"]
    )
    vm.start()
    yield vm
    vm.stop()

def test_http_server_status_ok(mock_vm):
    """Test that the HTTP server returns a 200 OK for the /status endpoint."""
    response = requests.get(f"{mock_vm.http_url}/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_http_server_not_found(mock_vm):
    """Test that the HTTP server returns a 404 Not Found for other endpoints."""
    response = requests.get(f"{mock_vm.http_url}/other")
    assert response.status_code == 404

def test_ssh_server_successful_connection(mock_vm, ssh_keys):
    """Test that we can successfully connect to the SSH server with the correct key."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            mock_vm.host,
            port=mock_vm.ssh_port,
            username='testuser',
            key_filename=ssh_keys["client_key"]
        )
        # If we get here, the connection was successful
        assert True
    except paramiko.AuthenticationException:
        pytest.fail("SSH connection failed with correct key.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during SSH connection: {e}")
    finally:
        client.close()

def test_ssh_server_failed_connection_wrong_key(mock_vm, ssh_keys):
    """Test that the SSH connection fails when using a wrong key."""
    key_dir = os.path.dirname(ssh_keys["client_key"])
    dummy_key_path = os.path.join(key_dir, 'dummy_key')
    os.system(f"ssh-keygen -t ed25519 -f {dummy_key_path} -N '' > /dev/null 2>&1")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    with pytest.raises(paramiko.AuthenticationException):
        client.connect(
            mock_vm.host,
            port=mock_vm.ssh_port,
            username='testuser',
            key_filename=dummy_key_path
        )
    client.close()

    # Clean up the dummy key
    os.remove(dummy_key_path)
    os.remove(f"{dummy_key_path}.pub")
