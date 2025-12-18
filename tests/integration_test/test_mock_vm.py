
import pytest
import requests
import paramiko
import os
from tests.mock_servers.mock_vm import MockVM

KEY_DIR = os.path.join(os.path.dirname(__file__), '..', 'mock_servers', 'keys')
CLIENT_KEY = os.path.join(KEY_DIR, 'mock_client_key')

@pytest.fixture(scope="module")
def mock_vm():
    """Pytest fixture to start and stop the MockVM."""
    vm = MockVM()
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

def test_ssh_server_successful_connection(mock_vm):
    """Test that we can successfully connect to the SSH server with the correct key."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            mock_vm.host,
            port=mock_vm.ssh_port,
            username='testuser',
            key_filename=CLIENT_KEY
        )
        # If we get here, the connection was successful
        assert True
    except paramiko.AuthenticationException:
        pytest.fail("SSH connection failed with correct key.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during SSH connection: {e}")
    finally:
        client.close()

def test_ssh_server_failed_connection_wrong_key(mock_vm):
    """Test that the SSH connection fails when using a wrong key."""
    # Generate a new dummy key for this test
    dummy_key_path = os.path.join(KEY_DIR, 'dummy_key')
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
