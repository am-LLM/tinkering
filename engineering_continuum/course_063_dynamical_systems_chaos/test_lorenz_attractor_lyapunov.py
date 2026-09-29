from lorenz_attractor_lyapunov import LorenzChaosSimulator

def test_lorenz_divergence():
    sim = LorenzChaosSimulator()
    div = sim.compute_trajectory_divergence(steps=3000, dt=0.01, eps=1e-3)
    assert div > 5.0, f"Expected chaotic macro trajectory divergence, got {div}"
