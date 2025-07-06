import asyncio
import logging
import socket
import struct
from .dominion import Packet, PROTOCOL_NAME

_LOGGER = logging.getLogger(__name__)

class SDGPeerConnector:
    def __init__(self, peer_id, host, port=80):
        self.peer_id = peer_id
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.is_connected = False

    async def connect(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self.is_connected = True
            _LOGGER.info("Connected to %s:%s", self.host, self.port)
            
            # Perform handshake or initial communication if necessary
            # For example, sending a packet with the protocol name
            # and peer ID to initiate the connection.
            
        except (socket.gaierror, ConnectionRefusedError, OSError) as e:
            _LOGGER.error("Failed to connect to %s:%s: %s", self.host, self.port, e)
            self.is_connected = False

    async def send_packet(self, packet):
        if not self.is_connected:
            _LOGGER.warning("Not connected, cannot send packet.")
            return

        try:
            data = packet.serialize()
            _LOGGER.debug("Sending packet: %s", data.hex())
            self.writer.write(data)
            await self.writer.drain()
        except ConnectionResetError as e:
            _LOGGER.error("Connection lost: %s", e)
            self.is_connected = False

    async def receive_packet(self):
        if not self.is_connected:
            _LOGGER.warning("Not connected, cannot receive packet.")
            return None

        try:
            # The first byte is the prefix, followed by the header
            prefix = await self.reader.readexactly(1)
            header = await self.reader.readexactly(Packet.HEADER_SIZE)
            
            # Unpack the header to get the payload length
            _, _, payload_len = struct.unpack('<BHB', header)
            
            # Read the payload
            payload = await self.reader.readexactly(payload_len)
            
            # Deserialize the full packet
            full_packet = prefix + header + payload
            _LOGGER.debug("Received packet: %s", full_packet.hex())
            return Packet.deserialize(full_packet)
            
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            _LOGGER.error("Failed to receive packet: %s", e)
            self.is_connected = False
            return None

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.is_connected = False
        _LOGGER.info("Connection closed.")
