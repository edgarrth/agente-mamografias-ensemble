from __future__ import annotations
import argparse
from mammography_agent.input_fidelity import audit_input_fidelity


def main():
    p = argparse.ArgumentParser(description="Audit CBIS-DDSM/model input fidelity without inference or mutations.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output")
    p.add_argument("--source-manifest")
    a = p.parse_args()
    print(audit_input_fidelity(a.run_dir, a.output, a.source_manifest))


if __name__ == "__main__":
    main()
