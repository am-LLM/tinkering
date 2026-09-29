import numpy as np
from cvar_risk_engine import CVaRRiskEngine

def test_cvar():
    returns = np.random.normal(0.001, 0.02, 1000)
    var, cvar = CVaRRiskEngine.calculate_var_cvar(returns, 0.95)
    assert cvar >= var
