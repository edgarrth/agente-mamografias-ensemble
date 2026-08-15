import argparse
from mammography_agent.pipeline import normal_test
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",required=True); p.add_argument("--samples",type=int)
    p.add_argument("--weights",nargs=3,type=float); p.add_argument("--threshold",type=float); p.add_argument("--config")
    p.add_argument("--max-runtime-minutes",type=float); a=p.parse_args()
    print(normal_test(a.datasets,a.samples,a.weights,a.threshold,a.config,a.max_runtime_minutes))
