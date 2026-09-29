from flight_propulsion_dynamics import FlightPropulsionDynamics

def test_flight_propulsion():
    dv = FlightPropulsionDynamics.tsiolkovsky_delta_v(isp_sec=300.0, m_initial=1000.0, m_dry=200.0)
    assert dv > 4000.0
    eff = FlightPropulsionDynamics.brayton_thermal_efficiency(pressure_ratio=10.0)
    assert 0.4 < eff < 0.6
