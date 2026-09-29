"""Course 090: Universal Design for Learning (UDL) Accessibility Audit Metric"""
import re

class UDLAccessibilityEvaluator:
    @staticmethod
    def calculate_flesch_kincaid_grade(text: str) -> float:
        words = re.findall(r'\w+', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        if not words or not sentences:
            return 0.0
        def count_syllables(w):
            return max(1, len(re.findall(r'[aeiouy]+', w.lower())))
        total_syllables = sum(count_syllables(w) for w in words)
        grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (total_syllables / len(words)) - 15.59
        return max(0.0, float(grade))

    @staticmethod
    def audit_alt_text(html_content: str) -> dict:
        total_imgs = len(re.findall(r'<img', html_content, re.I))
        imgs_with_alt = len(re.findall(r'<img[^>]+alt=["\'][^"\']+["\']', html_content, re.I))
        compliance = (imgs_with_alt / total_imgs) if total_imgs > 0 else 1.0
        return {"total_images": total_imgs, "compliant": imgs_with_alt, "ratio": float(compliance)}
