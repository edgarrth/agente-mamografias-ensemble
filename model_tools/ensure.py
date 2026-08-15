import argparse,json
from mammography_agent.model_client import ensure_model
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--models",nargs="+",required=True); a=p.parse_args()
    print(json.dumps([ensure_model(m) for m in a.models],indent=2))
