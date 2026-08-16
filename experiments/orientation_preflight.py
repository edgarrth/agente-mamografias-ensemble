import argparse
from mammography_agent.orientation_policy import audit_existing_run

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Run the label-independent v0.27 orientation preflight without classifier inference.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output")
    a=p.parse_args()
    print(audit_existing_run(a.run_dir,a.output))
