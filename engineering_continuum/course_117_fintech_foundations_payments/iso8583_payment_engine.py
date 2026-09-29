"""Course 117: ISO 8583 Payment Message Bitmap Parser & Luhn Validator"""
class ISO8583PaymentEngine:
    @staticmethod
    def luhn_checksum(card_number: str) -> bool:
        digits = [int(d) for d in card_number if d.isdigit()]
        if not digits:
            return False
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(int(x) for x in str(d * 2))
        return checksum % 10 == 0

    @staticmethod
    def parse_bitmap(bitmap_hex: str) -> list:
        bitmap_int = int(bitmap_hex, 16)
        active_fields = []
        for i in range(1, 65):
            if (bitmap_int >> (64 - i)) & 1:
                active_fields.append(i)
        return active_fields
