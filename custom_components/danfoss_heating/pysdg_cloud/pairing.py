import logging
import socket
import ssl
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from .grid import GridConnection

_LOGGER = logging.getLogger(__name__)

# The addresses of the Danfoss cloud servers
PAIRING_HOSTS = ["77.66.11.90", "77.66.11.92", "5.179.92.180", "5.179.92.182"]
PAIRING_PORT = 443


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
        for host in PAIRING_HOSTS:
            try:
                self._socket = ssl.create_default_context().wrap_socket(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                    server_hostname=host,
                )
                self._socket.connect((host, PAIRING_PORT))

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

                return  # Success
            except (socket.gaierror, ConnectionRefusedError, OSError) as e:
                _LOGGER.warning("Failed to connect to pairing server %s: %s", host, e)
            finally:
                if self._socket:
                    self._socket.close()
                    self._socket = None
        
        _LOGGER.error("Failed to connect to any pairing server.")

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
