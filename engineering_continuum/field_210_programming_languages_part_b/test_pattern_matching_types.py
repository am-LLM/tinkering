from pattern_matching_types import TaggedValue, ADTPatternMatcher

def test_adt_match():
    val = TaggedValue("Circle", 5.0)
    area = ADTPatternMatcher.match(val, {
        "Circle": lambda r: 3.14159 * r * r,
        "Square": lambda s: s * s
    })
    assert abs(area - 78.53975) < 1e-4
