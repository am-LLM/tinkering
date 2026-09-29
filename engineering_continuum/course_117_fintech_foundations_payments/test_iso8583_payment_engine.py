from iso8583_payment_engine import ISO8583PaymentEngine

def test_payment_engine():
    assert ISO8583PaymentEngine.luhn_checksum("49927398716") is True
    assert ISO8583PaymentEngine.luhn_checksum("49927398717") is False
    fields = ISO8583PaymentEngine.parse_bitmap("8000000000000001")
    assert fields == [1, 64]
