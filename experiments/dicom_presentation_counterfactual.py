from __future__ import annotations

import argparse

from mammography_agent.dicom_presentation_counterfactual import run_dicom_presentation_counterfactual


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare current DICOM conversion with Modality/VOI presentation transforms without labels or classifier inference."
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--source-manifest", default=None)
    p.add_argument("--write-images", action="store_true", help="Persist diagnostic 16-bit PNG copies for all three branches.")
    a = p.parse_args()
    print(run_dicom_presentation_counterfactual(a.run_dir, a.output, a.source_manifest, a.write_images))


if __name__ == "__main__":
    main()
