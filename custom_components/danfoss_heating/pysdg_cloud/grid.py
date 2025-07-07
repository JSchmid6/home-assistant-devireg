import logging
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

_LOGGER = logging.getLogger(__name__)


class GridConnection:
    """
    Represents a connection to the Danfoss cloud.
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.private_key = self._load_or_create_private_key()
        self.public_key = self.private_key.public_key()

    def _load_or_create_private_key(self):
        key_path = os.path.join(self.storage_path, "private_key.pem")
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        else:
            private_key = ec.generate_private_key(ec.SECP256R1())
            with open(key_path, "wb") as f:
                f.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )
            return private_key

    def get_my_peer_id(self) -> str:
        """
        Returns the public key in the format required by the Danfoss cloud.
        """
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex()

class GridConnectionKeeper:
    """
    Manages the GridConnection instance.
    """
    _connection = None
    _user_count = 0

    @staticmethod
    def get_connection(storage_path: str):
        if GridConnectionKeeper._connection is None:
            GridConnectionKeeper._connection = GridConnection(storage_path)
        return GridConnectionKeeper._connection

    @staticmethod
    def add_user():
        GridConnectionKeeper._user_count += 1

    @staticmethod
    def remove_user():
        GridConnectionKeeper._user_count -= 1
        if GridConnectionKeeper._user_count == 0:
            # Here we would close the connection if it's no longer needed.
            GridConnectionKeeper._connection = None
