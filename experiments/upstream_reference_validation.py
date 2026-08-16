from __future__ import annotations
import argparse
from mammography_agent.upstream_reference_validation import validate_upstream_sample

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Validate Blackwell runtimes against the official NYU metarepository 4-exam sample.")
    p.add_argument("--output", default=None)
    a=p.parse_args()
    print(validate_upstream_sample(a.output))
