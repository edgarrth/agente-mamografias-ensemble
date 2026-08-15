import argparse, json
from mammography_agent.model_client import ensure_gpu_model

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the explicit GPU compatibility image for selected models")
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()
    print(json.dumps([ensure_gpu_model(m) for m in a.models], indent=2))
