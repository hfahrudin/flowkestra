from __future__ import annotations
import logging
import paramiko
from flowkestra.schema import SSHConfig
from pathlib import Path

class SSHClient:
    def __init__(self, config: SSHConfig):
        """
        SSH client wrapper using SSHConfig schema.
        """
        self.config = config
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None
        self.connect()


    # ---------- internal helpers ----------
    def _log(self, msg: str):
        if self.config.debug:
            print(msg)

    # ---------- connection ----------
    def connect(self):
        """Establish SSH connection."""
        try:
            self._log(f"[SSH] Connecting to {self.config.hostname}:{self.config.port}...")
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.client.connect(
                hostname=self.config.hostname,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                key_filename=self.config.key_filename
            )
            self._log("[SSH] Connected successfully!")
        except Exception as e:
            raise RuntimeError(f"SSH connection failed: {e}")

    # ---------- command execution ----------
    def execute(self, command: str):
        """Execute a command remotely and return (stdout, stderr)."""
        if not self.client:
            raise RuntimeError("SSH client not connected. Call connect() first.")

        self._log(f"[SSH] Executing command: {command}")
        stdin, stdout, stderr = self.client.exec_command(command)




        if self.config.debug:
            full_output = []
            for line in iter(stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    self._log(f"[{self.config.hostname}][STDOUT] {clean_line}")
                    full_output.append(clean_line)
            output = "\n".join(full_output)
        else:
            output = stdout.read().decode().strip()

        error = stderr.read().decode().strip()
        if error:
            self._log(f"[SSH] Error: {error}")

        return output, error

    # ---------- file operations ----------
    def _ensure_sftp(self):
        if not self.client:
            raise RuntimeError("SSH client not connected.")
        if not self.sftp:
            self.sftp = self.client.open_sftp()

    def upload(self, local_path: str, remote_path: str):
        """Upload file to remote server."""
        self._ensure_sftp()
        remote_path_posix = Path(remote_path).as_posix()
        self._log(f"[SFTP] Uploading {local_path} → {remote_path_posix}")
        self.sftp.put(local_path, remote_path_posix)

    def download(self, remote_path: str, local_path: str):
        """Download file from remote server."""
        self._ensure_sftp()
        remote_path_posix = Path(remote_path).as_posix()
        self._log(f"[SFTP] Downloading {remote_path_posix} → {local_path}")
        self.sftp.get(remote_path_posix, local_path)

    # ---------- cleanup ----------
    def close(self):
        """Close SSH and SFTP sessions."""
        self._log("[SSH] Closing connection...")
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        self._log("[SSH] Connection closed.")
