import argparse
from mammography_agent.pipeline import freeze_experiment
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True); a=p.parse_args()
    print(freeze_experiment(a.experiment))
