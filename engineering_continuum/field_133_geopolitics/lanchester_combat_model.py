"""Course 133: Lanchester Linear and Square Law Combat Attrition Simulator"""
class LanchesterCombatModel:
    @staticmethod
    def square_law_step(blue_force: float, red_force: float, alpha: float = 0.05, beta: float = 0.03, dt: float = 0.1) -> tuple:
        d_blue = -beta * red_force * dt
        d_red = -alpha * blue_force * dt
        return max(0.0, blue_force + d_blue), max(0.0, red_force + d_red)
