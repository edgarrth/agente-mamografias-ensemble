# Implementation status — v0.14

## Validated on target workstation

- GMIC legacy CPU smoke test: PASS.
- GMIC Blackwell GPU probe and full GPU smoke test: PASS.
- DMV-CNN/NYU legacy CPU smoke test: PASS.
- DMV-CNN/NYU Blackwell GPU probe and full GPU smoke test: PASS.
- GLAM legacy CPU smoke test: PASS.
- GLAM Blackwell GPU probe and full GPU smoke test with XAI: PASS.
- `.env.example` now represents that validated three-GPU deployment state.

## Dataset layer

- CBIS-DDSM official-TCIA adapter: implemented.
- Missing-metadata preflight: implemented; returns `METADATA_REQUIRED` without DICOM indexing.
- Four official classification CSVs: required, recursively discovered, canonical and explicit TCIA filename aliases accepted.
- DICOM tree: expected from official TCIA/NBIA transfer.
- Full real CBIS-DDSM inspection: pending completion of local image + metadata download.
- Full CBIS-DDSM preparation: pending successful inspection.

No training/fine-tuning, model architecture changes or checkpoint changes are introduced in v0.14.
