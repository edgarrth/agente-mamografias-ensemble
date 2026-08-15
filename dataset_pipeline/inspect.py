import argparse, json
from mammography_agent.datasets.manager import inspect

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--force-dicom-index", action="store_true")
    a = p.parse_args()
    print(json.dumps(inspect(a.datasets, a.force_dicom_index), indent=2))
