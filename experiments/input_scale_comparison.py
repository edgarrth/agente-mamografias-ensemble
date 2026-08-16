from __future__ import annotations
import argparse
from mammography_agent.input_scale_comparison import compare_input_scale

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare one selected dataset input intensity scale with the official upstream sample without classifier inference.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--skip-nyu-crop", action="store_true", help="Compare raw prepared PNGs only; do not run official crop/optimal-center preprocessing.")
    a = p.parse_args()
    print(compare_input_scale(a.run_dir, a.output, include_nyu_crop=not a.skip_nyu_crop))
