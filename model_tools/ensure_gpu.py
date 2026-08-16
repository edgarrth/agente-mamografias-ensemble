import argparse, json
from mammography_agent.model_client import ensure_gpu_model

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the explicit GPU compatibility image for selected models")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild selected GPU images even when their configured build_revision is already present")
    a = p.parse_args()
    print(json.dumps([ensure_gpu_model(m, force_rebuild=a.force_rebuild) for m in a.models], indent=2))
