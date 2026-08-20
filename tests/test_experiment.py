import pandas as pd
from mammography_agent.ensemble.experiment import all_configurations, select_configuration

def test_grid_has_exactly_680_configurations():
    df=pd.DataFrame({"ground_truth":[0,1,0,1],"gmic_score":[.1,.9,.2,.8],"nyu_score":[.2,.8,.3,.7],"glam_score":[.1,.7,.4,.9]})
    r=all_configurations(df); assert len(r)==680; assert r.weight_id.nunique()==40; assert r.threshold_id.nunique()==17
    s=select_configuration(r); assert s.weight_id.startswith("W")
