from __future__ import annotations
import argparse
from mammography_agent.glam_runtime_differential import run_glam_runtime_differential

p = argparse.ArgumentParser(description="Compare original GLAM PyTorch 1.1 CPU runtime with the Blackwell compatibility runtime on the official sample.")
p.add_argument("--output", default=None)
a = p.parse_args()
print(run_glam_runtime_differential(a.output))
