from inventory_eoq_calculator import InventoryEOQCalculator

def test_eoq():
    eoq = InventoryEOQCalculator.economic_order_quantity(1000, 10, 2)
    assert abs(eoq - 100.0) < 1e-4
