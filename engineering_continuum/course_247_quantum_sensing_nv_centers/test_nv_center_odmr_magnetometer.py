from nv_center_odmr_magnetometer import NVCenterMagnetometer
import numpy as np

def test_zeeman_splitting():
    mag = NVCenterMagnetometer()
    fm, fp = mag.resonance_frequencies(b_z_tesla=1e-3) # 1 mT = 10 Gauss
    zeeman_shift = (fp - fm) / 2.0
    assert np.isclose(zeeman_shift, 28.0 * 1e-3)

def test_odmr_dip():
    mag = NVCenterMagnetometer()
    freqs = np.linspace(2.80, 2.94, 200)
    spec = mag.odmr_spectrum(freqs, b_z_tesla=1e-3)
    assert np.min(spec) < 1.0
    assert spec[0] > 0.98
