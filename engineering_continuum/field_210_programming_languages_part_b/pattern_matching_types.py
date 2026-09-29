"""Course 210: Algebraic Data Types (ADT) Tagged Union Pattern Matcher"""
class TaggedValue:
    def __init__(self, tag: str, value):
        self.tag = tag
        self.value = value

class ADTPatternMatcher:
    @staticmethod
    def match(tagged_val: TaggedValue, patterns: dict):
        if tagged_val.tag in patterns:
            return patterns[tagged_val.tag](tagged_val.value)
        elif "_" in patterns:
            return patterns["_"](tagged_val.value)
        raise ValueError("Pattern match exhaustiveness failure")
