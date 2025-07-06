import logging
from .connector import SDGPeerConnector
from .dominion import Packet

_LOGGER = logging.getLogger(__name__)

class DeviReg:
    def __init__(self, connector: SDGPeerConnector):
        self.connector = connector
        self.current_temp = None
        self.target_temp = None
        self.floor_temp = None
        self.hvac_mode = None
        self.preset_mode = None
        self.window_open = None
        self.heating_on = None
        self.control_mode = None

    async def set_temperature(self, temp):
        # Example: setting comfort temperature
        packet = Packet.create_with_payload(
            msg_class=3,  # DOMINION_SCHEDULER
            msg_code=2,   # SCHEDULER_SETPOINT_COMFORT
            data=temp,
            data_type='decimal'
        )
        await self.connector.send_packet(packet)

    async def update(self):
        # Request room temperature
        packet = Packet(
            msg_class=2,  # DOMINION_HEATING
            msg_code=2    # HEATING_TEMPERATURE_ROOM
        )
        await self.connector.send_packet(packet)
        
        response = await self.connector.receive_packet()
        if response:
            self.current_temp = response.get_decimal()

        # Request control mode
        packet = Packet(
            msg_class=3,  # DOMINION_SCHEDULER
            msg_code=1    # SCHEDULER_CONTROL_INFO
        )
        await self.connector.send_packet(packet)

        response = await self.connector.receive_packet()
        if response:
            mode_map = {
                1: "Manual",
                2: "Schedule",
                3: "Schedule", # Away
                4: "Vacation",
                6: "Pause",
                7: "Off",
                8: "Override"
            }
            self.control_mode = mode_map.get(response.get_byte())

    def get_current_temperature(self):
        return self.current_temp

    def get_target_temperature(self):
        return self.target_temp

    def get_floor_temperature(self):
        return self.floor_temp

    def get_hvac_mode(self):
        return self.hvac_mode

    def get_preset_mode(self):
        return self.preset_mode

    def get_window_open(self):
        return self.window_open

    def get_heating_on(self):
        return self.heating_on
        
    def get_peer_id(self):
        return self.connector.peer_id

    def get_control_mode(self):
        return self.control_mode

    async def set_control_mode(self, mode):
        # This is a simplified implementation. A real implementation would
        # need to handle the different mode transitions correctly, as shown
        # in the OpenHAB binding's setMode method.
        
        mode_map = {
            "Manual": 1,
            "Schedule": 2,
            "Vacation": 4,
            "Pause": 6,
            "Off": 7,
            "Override": 8
        }
        
        if mode in mode_map:
            packet = Packet.create_with_payload(
                msg_class=3,  # DOMINION_SCHEDULER
                msg_code=1,   # SCHEDULER_CONTROL_MODE
                data=mode_map[mode],
                data_type='byte'
            )
            await self.connector.send_packet(packet)

class IconRoom:
    def __init__(self, connector: SDGPeerConnector, room_number: int):
        self.connector = connector
        self.room_number = room_number
        self.current_temp = None
        self.target_temp = None
        self.hvac_mode = None
        self.preset_mode = None

    async def set_temperature(self, temp):
        # Example: setting comfort temperature
        packet = Packet.create_with_payload(
            msg_class=128 + self.room_number,  # ROOM_FIRST + roomNumber
            msg_code=3,   # ROOM_SETPOINTATHOME
            data=temp,
            data_type='decimal'
        )
        await self.connector.send_packet(packet)

    async def update(self):
        # Example: requesting room temperature
        packet = Packet(
            msg_class=128 + self.room_number,  # ROOM_FIRST + roomNumber
            msg_code=2    # ROOM_ROOMTEMPERATURE
        )
        await self.connector.send_packet(packet)
        
        response = await self.connector.receive_packet()
        if response:
            self.current_temp = response.get_decimal()

    def get_current_temperature(self):
        return self.current_temp

    def get_target_temperature(self):
        return self.target_temp

    def get_hvac_mode(self):
        return self.hvac_mode

    def get_preset_mode(self):
        return self.preset_mode
        
    def get_peer_id(self):
        return f"{self.connector.peer_id}_{self.room_number}"
