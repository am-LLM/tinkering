"""Course 099: I2C/SPI Bus Frame Protocol Transceiver with CRC-8 Checksum"""
class I2CSPIFrameCodec:
    @staticmethod
    def crc8(data: bytes, polynomial: int = 0x07) -> int:
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ polynomial) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    @classmethod
    def encode_i2c_frame(cls, address_7bit: int, rw_bit: int, payload: bytes) -> bytes:
        addr_byte = ((address_7bit & 0x7F) << 1) | (rw_bit & 0x01)
        frame = bytes([addr_byte]) + payload
        checksum = cls.crc8(frame)
        return frame + bytes([checksum])
