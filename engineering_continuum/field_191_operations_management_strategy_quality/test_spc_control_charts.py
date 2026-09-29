import numpy as np
from spc_control_charts import SPCControlCharts

def test_spc():
    data = np.array([[10, 11, 10], [10, 10, 10], [9, 10, 11]])
    res = SPCControlCharts.calculate_xbar_limits(data)
    assert abs(res["x_double_bar"] - 10.111) < 0.1
