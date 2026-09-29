"""Course 139: WCAG 2.1 Luminance & Color Contrast Ratio Validator"""
class WCAGAccessibilityValidator:
    @staticmethod
    def relative_luminance(r: int, g: int, b: int) -> float:
        def adjust(c):
            c_norm = c / 255.0
            return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)

    @classmethod
    def contrast_ratio(cls, rgb1: tuple, rgb2: tuple) -> float:
        l1 = cls.relative_luminance(*rgb1)
        l2 = cls.relative_luminance(*rgb2)
        lum_max = max(l1, l2)
        lum_min = min(l1, l2)
        return float((lum_max + 0.05) / (lum_min + 0.05))
