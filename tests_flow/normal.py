import argparse
from mammography_agent.pipeline import normal_test, SAMPLING_STRATEGIES

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",nargs="+",required=True)
    p.add_argument("--samples",type=int)
    p.add_argument("--sampling",choices=SAMPLING_STRATEGIES,default="sequential",
                   help="sequential (legacy first-N), random, stratified (dataset-proportional), or balanced (equal classes)")
    p.add_argument("--seed",type=int,default=42,help="Deterministic seed for random/stratified/balanced sampling")
    p.add_argument("--weights",nargs=3,type=float)
    p.add_argument("--threshold",type=float)
    p.add_argument("--config")
    p.add_argument("--max-runtime-minutes",type=float)
    a=p.parse_args()
    print(normal_test(a.datasets,a.samples,a.weights,a.threshold,a.config,a.max_runtime_minutes,a.sampling,a.seed))
