from token_bucket_traffic_shaper import TokenBucketShaper

def test_token_bucket():
    tb = TokenBucketShaper(rate_tokens_per_sec=10.0, burst_capacity=20.0)
    assert tb.consume(15.0, 0.0) is True
    assert tb.consume(10.0, 0.0) is False # Only 5 tokens left
    assert tb.consume(10.0, 1.0) is True  # 5 + 10 = 15 tokens
