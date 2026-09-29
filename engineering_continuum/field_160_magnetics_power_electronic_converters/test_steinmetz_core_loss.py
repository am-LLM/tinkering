from steinmetz_core_loss import SteinmetzCoreLoss

def test_skin_depth():
    delta = SteinmetzCoreLoss.skin_depth_copper(100e3) # 100 kHz
    assert 0.0001 < delta < 0.001
