import logging

_LOGGER = logging.getLogger(__name__)

class GridConnection:
    """
    Represents a connection to the Danfoss cloud.
    """
    def __init__(self):
        # This will be expanded to handle the connection details.
        pass

    def get_my_peer_id(self):
        # This will return the peer ID of the Home Assistant instance.
        pass

class GridConnectionKeeper:
    """
    Manages the GridConnection instance.
    """
    _connection = None
    _user_count = 0

    @staticmethod
    def get_connection():
        if GridConnectionKeeper._connection is None:
            GridConnectionKeeper._connection = GridConnection()
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
