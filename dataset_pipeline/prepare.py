import argparse,json
from mammography_agent.datasets.manager import prepare
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",required=True); a=p.parse_args()
    print(json.dumps(prepare(a.datasets),indent=2))
