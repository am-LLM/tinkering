"""Course 161: Dynamic Tiered Pricing & Tax Discount Matrix Engine"""
class ECommercePricingEngine:
    @staticmethod
    def calculate_total(cart_subtotal: float, discount_pct: float, tax_pct: float, shipping_flat: float) -> dict:
        discounted = cart_subtotal * (1.0 - discount_pct)
        tax = discounted * tax_pct
        total = discounted + tax + shipping_flat
        return {
            "subtotal": cart_subtotal,
            "discount_amount": cart_subtotal * discount_pct,
            "tax": tax,
            "total": total
        }
