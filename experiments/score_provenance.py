import argparse
from mammography_agent.score_provenance import audit_score_provenance

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Audit native model score provenance without rerunning GPU inference")
    p.add_argument("--run-dir",required=True,help="Existing normal-test run directory containing model_batch native outputs")
    p.add_argument("--output",help="Optional output directory; default workspace/output/analyses/score-provenance-<timestamp>")
    a=p.parse_args()
    print(audit_score_provenance(a.run_dir,a.output))
