"""Course 170: Goldman-Hodgkin-Katz (GHK) Multi-Ion Membrane Voltage Equation"""
import math

class GHKVoltageModel:
    @staticmethod
    def calculate_vm(p_k: float, p_na: float, p_cl: float, k_in: float, k_out: float, na_in: float, na_out: float, cl_in: float, cl_out: float, temp_c: float = 37.0) -> float:
        # RT/F at temp
        rt_f = (8.314 * (temp_c + 273.15)) / 96485.0 * 1000.0 # in mV
        num = p_k * k_out + p_na * na_out + p_cl * cl_in
        den = p_k * k_in + p_na * na_in + p_cl * cl_out
        return float(rt_f * math.log(num / den))
