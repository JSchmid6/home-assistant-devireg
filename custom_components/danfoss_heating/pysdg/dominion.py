import struct
from datetime import datetime, timezone

class Packet:
    HEADER_SIZE = 4

    def __init__(self, msg_class=None, msg_code=None, payload=b''):
        if msg_class is not None and msg_code is not None:
            # Constructing a packet to be sent
            self.msg_class = msg_class
            self.msg_code = msg_code
            self.payload = payload
            self.is_request = not payload
        else:
            # Parsing an incoming packet
            self.msg_class = None
            self.msg_code = None
            self.payload = None
            self.is_request = False

    def serialize(self):
        header = struct.pack('<BHB', self.msg_class, self.msg_code, len(self.payload))
        prefix = b'\x01' if self.is_request else b'\x00'
        return prefix + header + self.payload

    @classmethod
    def deserialize(cls, data):
        prefix = data[0]
        header = data[1:cls.HEADER_SIZE + 1]
        payload = data[cls.HEADER_SIZE + 1:]
        
        msg_class, msg_code, payload_len = struct.unpack('<BHB', header)
        
        packet = cls()
        packet.msg_class = msg_class
        packet.msg_code = msg_code
        packet.payload = payload
        packet.is_request = (prefix == 1)
        
        return packet

    def get_byte(self):
        return struct.unpack('<B', self.payload)[0]

    def get_short(self):
        return struct.unpack('<h', self.payload)[0]

    def get_int(self):
        return struct.unpack('<i', self.payload)[0]

    def get_boolean(self):
        return self.get_byte() != 0

    def get_decimal(self):
        fixed = self.get_short()
        return float('nan') if fixed == 0x8000 else fixed / 100.0

    def get_string(self):
        length = self.payload[0]
        return self.payload[1:1 + length].decode('utf-8')

    def get_date(self):
        sec, minute, hour, day_dow, month, year = struct.unpack('<BBBBBB', self.payload[:6])
        day = day_dow & 0x1F
        year += 2000
        return datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)

    @staticmethod
    def create_with_payload(msg_class, msg_code, data, data_type):
        if data_type == 'byte':
            payload = struct.pack('<B', data)
        elif data_type == 'short':
            payload = struct.pack('<h', data)
        elif data_type == 'int':
            payload = struct.pack('<i', data)
        elif data_type == 'boolean':
            payload = struct.pack('<B', 1 if data else 0)
        elif data_type == 'decimal':
            payload = struct.pack('<h', int(data * 100))
        elif data_type == 'string':
            payload = bytes([len(data)]) + data.encode('utf-8')
        else:
            payload = data
        
        return Packet(msg_class, msg_code, payload)

PROTOCOL_NAME = "dominion-1.0"
