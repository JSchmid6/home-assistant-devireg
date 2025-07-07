import logging
from typing import Optional, Dict, Any
from .grid import GridConnectionKeeper
from .pairing import PairingConnection
from .peer import PeerConnection
from .dominion import DominionConfiguration

_LOGGER = logging.getLogger(__name__)


class DanfossCloudConnector:
    """
    Handles the cloud-based 'Share house' functionality for Danfoss devices.
    """

    def __init__(self, otp: str, user_name: str, storage_path: str):
        self.otp = otp
        self.user_name = user_name
        self.storage_path = storage_path

    async def get_devices(self) -> Optional[Dict[str, Any]]:
        """
        Connects to the Danfoss cloud, sends the one-time password,
        and retrieves the device configuration.
        """
        _LOGGER.debug("Attempting to get devices from cloud with OTP")

        GridConnectionKeeper.add_user()
        grid = GridConnectionKeeper.get_connection(self.storage_path)

        my_peer_id = grid.get_my_peer_id()
        if my_peer_id is None:
            _LOGGER.error("Peer ID is not set")
            GridConnectionKeeper.remove_user()
            return None

        pairing = PairingConnection()
        pairing.pair_with_remote(grid, self.otp)
        phone_id = pairing.get_peer_id()
        pairing.close()

        if phone_id is None:
            _LOGGER.error("Pairing failed")
            GridConnectionKeeper.remove_user()
            return None

        _LOGGER.debug("Pairing successful")

        cfg = PeerConnection()
        cfg.connect_to_remote(grid, phone_id, "dominion-configuration-1.0")

        request = DominionConfiguration.Request(self.user_name, my_peer_id)
        cfg.send_data(request.to_json().encode())

        data = cfg.receive_data()
        cfg.close()
        GridConnectionKeeper.remove_user()

        if data is None:
            _LOGGER.error("Failed to receive config")
            return None

        response = DominionConfiguration.Response(data)
        return response.__dict__
