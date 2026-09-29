"""Course 162: Economic Order Quantity (EOQ) & Safety Stock Inventory Engine"""
import math

class InventoryEOQCalculator:
    @staticmethod
    def economic_order_quantity(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
        # EOQ = sqrt(2 * D * S / H)
        return float(math.sqrt(2.0 * annual_demand * order_cost / holding_cost_per_unit))

    @staticmethod
    def reorder_point(lead_time_days: float, daily_demand: float, safety_stock: float) -> float:
        return float(lead_time_days * daily_demand + safety_stock)
