from emg_feature_classifier import EMGProstheticClassifier
import numpy as np

def test_emg_feature_extraction():
    clf = EMGProstheticClassifier()
    sig = np.sin(np.linspace(0, 10, 200))
    feats = clf.extract_mav_rms_wl(sig)
    assert len(feats) == 3
    assert feats[0] > 0 # MAV > 0
    assert feats[1] > 0 # RMS > 0

def test_emg_gesture_classification():
    clf = EMGProstheticClassifier()
    train_data = {
        "rest": [np.random.normal(0, 0.05, 200) for _ in range(5)],
        "power_grasp": [np.random.normal(0, 1.5, 200) for _ in range(5)]
    }
    clf.train_lda(train_data)
    pred_rest = clf.predict_gesture(np.random.normal(0, 0.05, 200))
    pred_grasp = clf.predict_gesture(np.random.normal(0, 1.5, 200))
    assert pred_rest == "rest"
    assert pred_grasp == "power_grasp"
