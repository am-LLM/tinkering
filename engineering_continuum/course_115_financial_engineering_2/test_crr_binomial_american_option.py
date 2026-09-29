from crr_binomial_american_option import CRRBinomialPricer
import pytest

def test_american_put():
    price = CRRBinomialPricer.price_american_put(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, steps=20)
    assert price > 0.0
