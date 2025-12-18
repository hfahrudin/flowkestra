
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import paramiko
import socket
import os

class MockVMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

class MockSSHServer(paramiko.ServerInterface):
    def __init__(self, client_pub_key_path):
        self.event = threading.Event()
        self.client_pub_key_path = client_pub_key_path

    def check_auth_publickey(self, username, key):
        with open(self.client_pub_key_path, 'r') as f:
            pubkey_line = f.read().strip()
        key_type, key_string, comment = pubkey_line.split()

        if key.get_name() == key_type and key.get_base64() == key_string:
            return paramiko.AUTH_SUCCESSFUL

        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        if kind == "session":
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def get_allowed_auths(self, username):
        return "publickey"

class MockVM:
    def __init__(self, host='localhost', http_port=8000, ssh_port=2200, host_key_path=None, client_pub_key_path=None):
        self.host = host
        self.http_port = http_port
        self.ssh_port = ssh_port
        self.host_key_path = host_key_path
        self.client_pub_key_path = client_pub_key_path

        # HTTP Server
        self.http_server = HTTPServer((self.host, self.http_port), MockVMHandler)
        self.http_thread = threading.Thread(target=self.http_server.serve_forever)
        self.http_thread.daemon = True

        # SSH Server
        self.ssh_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ssh_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ssh_server_sock.bind((self.host, self.ssh_port))
        self.ssh_server_sock.listen(100)
        self.ssh_thread = threading.Thread(target=self.run_ssh_server)
        self.ssh_thread.daemon = True
        self.ssh_running = False

    def run_ssh_server(self):
        self.ssh_running = True
        while self.ssh_running:
            try:
                client_sock, client_addr = self.ssh_server_sock.accept()
                t = paramiko.Transport(client_sock)
                t.add_server_key(paramiko.Ed25519Key(filename=self.host_key_path))
                server = MockSSHServer(self.client_pub_key_path)
                t.start_server(server=server)
                chan = t.accept(20)
                if chan is not None:
                    chan.close()
                t.close()

            except Exception as e:
                # This will happen when we shut down the socket
                pass


    def start(self):
        self.http_thread.start()
        print(f"Mock VM HTTP server started at http://{self.host}:{self.http_port}")
        self.ssh_thread.start()
        print(f"Mock VM SSH server started at ssh://{self.host}:{self.ssh_port}")

    def stop(self):
        # Stop HTTP server
        self.http_server.shutdown()
        self.http_server.server_close()
        self.http_thread.join()
        print("Mock VM HTTP server stopped.")

        # Stop SSH server
        self.ssh_running = False
        # To unblock the accept call
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.ssh_port))
        self.ssh_server_sock.close()
        self.ssh_thread.join()
        print("Mock VM SSH server stopped.")

    @property
    def http_url(self):
        return f"http://{self.host}:{self.http_port}"

    @property
    def ssh_uri(self):
        return f"ssh://{self.host}:{self.ssh_port}"
