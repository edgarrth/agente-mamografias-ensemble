#!/usr/bin/env bash
set -euo pipefail
mkdir -p workspace/{input,datasets/{raw,processed,manifests,rejected},models,runtime,output/{analyses,normal_tests,experiments,final_evaluations,xai,reports},logs}
echo "Workspace ready: $(pwd)/workspace"
echo "Datasets are NOT downloaded by bootstrap. Use dataset_pipeline.download explicitly."
