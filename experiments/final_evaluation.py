import argparse
from mammography_agent.pipeline import final_evaluation
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True); a=p.parse_args()
    print(final_evaluation(a.experiment))
