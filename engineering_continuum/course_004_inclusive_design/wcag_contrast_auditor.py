"""Course 004: WCAG 2.1 Color Contrast & Luminance Calculator"""
class ContrastAuditor:
    @staticmethod
    def relative_luminance(rgb_255):
        srgb = [c / 255.0 for c in rgb_255]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    @classmethod
    def contrast_ratio(cls, rgb1, rgb2):
        l1, l2 = cls.relative_luminance(rgb1), cls.relative_luminance(rgb2)
        lighter, darker = max(l1, l2), min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)
