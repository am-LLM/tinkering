from rfm_customer_segmentation import RFMCustomerSegmentation

def test_rfm():
    scores = RFMCustomerSegmentation.score_rfm([5, 100], [10, 1], [500, 20])
    assert scores[0]["r_score"] == 5
    assert scores[0]["f_score"] == 5
