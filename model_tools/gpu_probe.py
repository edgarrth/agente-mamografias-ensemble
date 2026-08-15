import argparse, json
from mammography_agent.model_client import gpu_probe

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run fail-safe CUDA allocation/kernel probes for selected model GPU runtimes")
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()
    print(json.dumps([gpu_probe(m) for m in a.models], indent=2))
