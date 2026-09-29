import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_51_dna_repair_llm_poison_filter import DNARepairAttentionPoisonFilter
from engine_52_relativistic_quantum_radar_insar import RelativisticQuantumInSARNav
from engine_53_replicator_phage_microfluidics import PhageEvolutionaryGameReactor
from engine_54_superconducting_sco2_datacenter_power import SuperconductingBraytonDataCenterPower
from engine_55_agentic_cbf_market_circuit_breaker import AgenticMarketCircuitBreaker
from engine_56_circadian_endocrine_microgrid import CircadianMicrogridManager
from engine_57_quantum_squeezed_das_geothermal import QuantumSqueezedGeothermalDAS
from engine_58_synthetic_myoglobin_uuv_oxygen import SyntheticMyoglobinUUVPower
from engine_59_active_inference_pain_interceptor import ActiveInferencePainInterceptor
from engine_60_astrocytic_calcium_lifelong_snn import AstrocyticLifelongSNN
from engine_61_superconducting_fluxon_space_debris import SuperconductingSpaceDebrisTrapper
from engine_62_slime_mold_evacuation_routing import PhysarumEvacuationRouter
from engine_63_stomatal_turgor_zero_power_hvac import StomatalHydrogelBuildingSkin
from engine_64_non_hermitian_ep_sepsis_detector import ExceptionalPointSepsisDetector
from engine_65_frechet_evt_anti_jam_trajectory import FrechetEWTrajectoryPlanner

class TestEngines51To65(unittest.TestCase):
    def test_engine_51_dna_repair(self):
        f = DNARepairAttentionPoisonFilter(embedding_dim=32)
        res = f.evaluate_token_excision_repair(np.random.randn(20, 32))
        self.assertIn("is_poisoned", res)

    def test_engine_52_quantum_insar(self):
        nav = RelativisticQuantumInSARNav()
        snr = nav.compute_two_mode_squeezed_quantum_advantage(1e-3, 1e-4)
        self.assertGreater(snr, 0.0)

    def test_engine_53_phage_replicator(self):
        g = PhageEvolutionaryGameReactor()
        p, b = g.step_replicator_dynamics(np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertAlmostEqual(np.sum(p), 1.0)
        self.assertAlmostEqual(np.sum(b), 1.0)

    def test_engine_54_superconducting_power(self):
        pwr = SuperconductingBraytonDataCenterPower()
        res = pwr.compute_brayton_cycle_work(350.0)
        self.assertGreater(res["net_power_mw"], 0.0)

    def test_engine_55_cbf_circuit_breaker(self):
        cb = AgenticMarketCircuitBreaker()
        safe_order, is_clamped, h = cb.evaluate_cbf_qp_safety(50.0, 0.8, 10.0)
        self.assertTrue(is_clamped)

    def test_engine_56_circadian_microgrid(self):
        m = CircadianMicrogridManager()
        res = m.step_circadian_shedding(10.0, 150.0, 200.0)
        self.assertGreater(res["battery_soc"], 0.0)

    def test_engine_57_quantum_squeezed_das(self):
        das = QuantumSqueezedGeothermalDAS()
        sens = das.compute_acoustic_strain_sensitivity(250.0)
        self.assertLess(sens, 1e-10)

    def test_engine_58_myoglobin_uuv(self):
        uuv = SyntheticMyoglobinUUVPower()
        res = uuv.calculate_oxygen_release_rate(50.0, 15.0)
        self.assertGreater(res["mission_endurance_hours"], 10.0)

    def test_engine_59_pain_interceptor(self):
        pi = ActiveInferencePainInterceptor()
        stim, fe = pi.step_free_energy_cancellation(np.array([1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 2.0]))
        self.assertEqual(len(stim), 8)

    def test_engine_60_astrocytic_snn(self):
        snn = AstrocyticLifelongSNN(num_nodes=8)
        gate = snn.propagate_glial_calcium_wave(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
        self.assertEqual(len(gate), 8)

    def test_engine_61_space_debris_trapper(self):
        t = SuperconductingSpaceDebrisTrapper()
        dv = t.compute_lorentz_eddy_braking(7500.0)
        self.assertGreater(dv, 0.0)

    def test_engine_62_slime_mold_router(self):
        sm = PhysarumEvacuationRouter(num_nodes=6)
        cond = sm.step_tubule_adaptation(np.ones((6, 6)) * 2.0)
        self.assertEqual(cond.shape, (6, 6))

    def test_engine_63_stomatal_hvac(self):
        hvac = StomatalHydrogelBuildingSkin()
        ap = hvac.step_turgor_ventilation(75.0, 800.0)
        self.assertGreater(ap, 0.0)

    def test_engine_64_ep_sepsis_detector(self):
        ep = ExceptionalPointSepsisDetector()
        sens = ep.compute_ep_eigenvalue_splitting(1e-4)
        self.assertGreater(sens, 10.0)

    def test_engine_65_frechet_anti_jam(self):
        f = FrechetEWTrajectoryPlanner()
        vec = f.compute_jamming_null_escape_vector(np.random.uniform(0.1, 5.0, (10, 10)))
        self.assertEqual(len(vec), 3)

if __name__ == '__main__':
    unittest.main()
