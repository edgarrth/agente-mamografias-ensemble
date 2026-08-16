import argparse
from mammography_agent.orientation_counterfactual import audit_orientation_counterfactual

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run a targeted horizontal_flip counterfactual on orientation-suspect studies.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output')
    p.add_argument('--min-nonzero-views', type=int, default=4)
    a = p.parse_args()
    print(audit_orientation_counterfactual(a.run_dir, a.output, a.min_nonzero_views))
