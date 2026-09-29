"""Course 202: Synchronous Boost Converter Small-Signal State-Space Model"""
class SynchronousBoostModel:
    @staticmethod
    def small_signal_gvd(v_out: float, duty: float, r_load: float, l_h: float, c_f: float) -> dict:
        # Right Half-Plane Zero (RHPZ) frequency: w_z = (1-D)^2 * R / L
        w_rhpz = ((1.0 - duty) ** 2) * r_load / l_h
        dc_gain = v_out / (1.0 - duty)
        return {"dc_gain": float(dc_gain), "rhpz_rad_s": float(w_rhpz)}
