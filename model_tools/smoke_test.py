import argparse,json
from mammography_agent.model_client import smoke_test
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--models",nargs="+",required=True); a=p.parse_args()
    print(json.dumps([smoke_test(m) for m in a.models],indent=2))
