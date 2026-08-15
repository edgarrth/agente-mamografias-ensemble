import argparse,json
from mammography_agent.datasets.manager import request_download
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--datasets",nargs="+",required=True); a=p.parse_args()
    print(json.dumps(request_download(a.datasets),indent=2))
