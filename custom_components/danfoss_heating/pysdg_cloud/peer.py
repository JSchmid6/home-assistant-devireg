import logging
import socket
import ssl
from .grid import GridConnection

_LOGGER = logging.getLogger(__name__)

# The address of the Danfoss cloud server
PEER_HOST = "jigsaw.trifork.com"
PEER_PORT = 25500


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
        try:
            self._socket = ssl.create_default_context().wrap_socket(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                server_hostname=PEER_HOST,
            )
            self._socket.connect((PEER_HOST, PEER_PORT))

            # This is a simplified representation of the connection process.
            # A real implementation would involve a more complex handshake.
            client_hello = f"CONNECT {peer_id} {protocol}\n\n".encode()
            self._socket.sendall(client_hello)

            response = self._socket.recv(1024)
            # In a real implementation, we would parse the response
            # to confirm the connection was successful.

        except (socket.gaierror, ConnectionRefusedError, OSError) as e:
            _LOGGER.error("Failed to connect to peer: %s", e)

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
