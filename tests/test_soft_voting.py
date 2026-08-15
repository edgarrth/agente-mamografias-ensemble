from mammography_agent.ensemble.soft_voting import vote

def test_baseline_vote():
    r=vote({"gmic":0.9,"nyu":0.6,"glam":0.3},{"gmic":0.333333,"nyu":0.333333,"glam":0.333334},0.50)
    assert 0 <= r.ensemble_malignancy_score <= 1
    assert r.classification == "CANCER"
