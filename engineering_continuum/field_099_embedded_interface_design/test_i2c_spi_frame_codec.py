from i2c_spi_frame_codec import I2CSPIFrameCodec

def test_frame_encoding():
    frame = I2CSPIFrameCodec.encode_i2c_frame(0x48, 0, b"\x01\x02")
    assert len(frame) == 4
    assert I2CSPIFrameCodec.crc8(frame[:-1]) == frame[-1]
