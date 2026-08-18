# Web unit-inference + MinIO extension on v0.30.2

This extension intentionally keeps the validated RSNA batch methodology at **v0.30.2** unchanged.
It adds an isolated Web path for a single mammography exam.

## Web contract

- Input: DICOM only; no `train.csv`, no `cancer`, no ground truth.
- Required ensemble views: L-CC, R-CC, L-MLO and R-MLO.
- DICOM metadata is inspected before inference. Mammography `ViewCodeSequence (0054,0220)` and `ViewPosition` are prioritized; conservative descriptive fallbacks include `SeriesDescription`, `ProtocolName`, `RequestedProcedureDescription`, `StudyDescription`, `ImageType`, and procedure code sequences. Laterality uses `ImageLaterality`/`Laterality`.
- If CC/MLO cannot be resolved, Streamlit provides a presentation-only preview. When laterality is already known, the user selects only CC or MLO; a full L/R + projection assignment is requested only when laterality is also unavailable. No clinical label is requested.
- Preview images are generated exclusively for visual verification and are not model inputs. The inferential representation continues to use the common DICOM-to-16-bit-PNG converter.
- The common DICOM-to-16-bit-PNG converter, orientation policy, `_infer_three`, Model Runner and `ensemble.soft_voting.vote` are reused.
- Web output is written under `workspace/output/single_cases/<run_id>/` and never under batch experiment directories.

## Persistence

- PostgreSQL keeps a compact row in `web_inference_runs` with the model scores, ensemble score, threshold, classification and MinIO pointer.
- MinIO bucket `mammography-web` stores the original uploaded DICOMs, four canonical PNGs, compact audit artifacts, result JSON and an object manifest.
- MinIO is **not** an input to model inference. A MinIO failure is recorded as a non-blocking persistence failure and does not change a completed prediction.
- The large `model_batch/preprocessed` workspace is not mirrored to MinIO.

## Batch isolation

The following validated files/entrypoints are deliberately not modified by this extension:

- `mammography_agent/pipeline.py`
- `experiments/run.py`
- `experiments/final_evaluation.py`
- `tests_flow/normal.py`
- `config/experiments.yaml`
- `config/ensemble.yaml`

The Web path does not call `_infer_three_chunked`, formal split/resume, 80-configuration ranking, freeze or Final Test.
