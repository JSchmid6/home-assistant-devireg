import logging
import socket
import ssl
from .grid import GridConnection

_LOGGER = logging.getLogger(__name__)

# The addresses of the Danfoss cloud servers
PEER_HOSTS = ["77.66.11.90", "77.66.11.92", "5.179.92.180", "5.179.92.182"]
PEER_PORT = 443


class PeerConnection:
    """
    Handles the communication with a paired device.
    """

    def __init__(self):
        self._socket = None

    def connect_to_remote(
        self, grid_connection: GridConnection, peer_id: str, protocol: str
    ):
        """
        Connects to the remote peer.
        """
        for host in PEER_HOSTS:
            try:
                self._socket = ssl.create_default_context().wrap_socket(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                    server_hostname=host,
                )
                self._socket.connect((host, PEER_PORT))

                # This is a simplified representation of the connection process.
                # A real implementation would involve a more complex handshake.
                client_hello = f"CONNECT {peer_id} {protocol}\n\n".encode()
                self._socket.sendall(client_hello)

                response = self._socket.recv(1024)
                # In a real implementation, we would parse the response
                # to confirm the connection was successful.

                return  # Success
            except (socket.gaierror, ConnectionRefusedError, OSError) as e:
                _LOGGER.warning("Failed to connect to peer %s: %s", host, e)
            finally:
                if self._socket:
                    self._socket.close()
                    self._socket = None

        _LOGGER.error("Failed to connect to any peer server.")

    def send_data(self, data: bytes):
        """
        Sends data to the remote peer.
        """
        if self._socket:
            self._socket.sendall(data)

    def receive_data(self) -> bytes:
        """
        Receives data from the remote peer.
        """
        if self._socket:
            return self._socket.recv(1024)
        return None

    def close(self):
        """
        Closes the connection to the remote peer.
        """
        if self._socket:
            self._socket.close()
