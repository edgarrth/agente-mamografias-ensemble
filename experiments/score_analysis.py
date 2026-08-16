import argparse
from mammography_agent.score_analysis import analyze_score_file

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze cached model scores without running GPU inference")
    p.add_argument("--input", required=True, help="raw_model_predictions.csv or configuration_set_predictions.csv")
    p.add_argument("--output", help="Optional output directory; default is workspace/output/analyses/score-analysis-<timestamp>")
    a = p.parse_args()
    print(analyze_score_file(a.input, a.output))
