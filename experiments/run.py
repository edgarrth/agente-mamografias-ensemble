import argparse
from mammography_agent.pipeline import experimental_test

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",nargs="+",required=True)
    p.add_argument("--samples",type=int)
    p.add_argument("--configuration-ratio",type=float,default=0.30)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--chunk-size",type=int,default=None,help="Formal orientation/inference chunk size; defaults to config/experiments.yaml")
    p.add_argument("--resume-experiment",default=None,help="Resume an interrupted Configuration experiment using its frozen manifests/checkpoints")
    a=p.parse_args()
    print(experimental_test(a.datasets,a.samples,a.configuration_ratio,a.seed,a.chunk_size,a.resume_experiment))
