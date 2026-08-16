from __future__ import annotations
import argparse
from mammography_agent.breast_ensemble_analysis import analyze_breast_ensemble

def main():
    p=argparse.ArgumentParser(description='Compare current and breast-aware ensemble aggregation without inference.')
    p.add_argument('--breast-level-scores',required=True)
    p.add_argument('--output')
    p.add_argument('--config')
    a=p.parse_args()
    print(analyze_breast_ensemble(a.breast_level_scores,a.output,a.config))

if __name__=='__main__':
    main()
