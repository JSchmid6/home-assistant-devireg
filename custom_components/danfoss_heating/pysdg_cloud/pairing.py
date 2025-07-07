import logging
import socket
import ssl
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from .grid import GridConnection

_LOGGER = logging.getLogger(__name__)

# The address of the Danfoss cloud server
PAIRING_HOST = "jigsaw.trifork.com"
PAIRING_PORT = 25500


class PairingConnection:
    """
    Handles the pairing process with the Danfoss cloud.
    """

    def __init__(self):
        self._peer_id = None
        self._socket = None

    def pair_with_remote(self, grid_connection: GridConnection, otp: str):
        """
        Pairs with the remote device using the provided OTP.
        """
        try:
            self._socket = ssl.create_default_context().wrap_socket(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                server_hostname=PAIRING_HOST,
            )
            self._socket.connect((PAIRING_HOST, PAIRING_PORT))

            # This is a simplified representation of the pairing process.
            # A real implementation would involve a more complex handshake.
            client_hello = b"\x01\x00"  # Example client hello
            self._socket.sendall(client_hello)

            server_hello = self._socket.recv(1024)
            # In a real implementation, we would parse the server hello
            # and perform a key exchange.

            # For now, we'll assume the pairing is successful and
            # the peer ID is returned in the response.
            self._peer_id = "00:11:22:33:44:55"  # Example peer ID

        except (socket.gaierror, ConnectionRefusedError, OSError) as e:
            _LOGGER.error("Failed to connect to pairing server: %s", e)
        finally:
            if self._socket:
                self._socket.close()

    def get_peer_id(self) -> str:
        """
        Returns the peer ID of the paired device.
        """
        return self._peer_id

    def close(self):
        """
        Closes the pairing connection.
        """
        if self._socket:
            self._socket.close()
