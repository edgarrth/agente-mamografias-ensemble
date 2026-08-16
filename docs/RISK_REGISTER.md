# Risk register — research prototype

| Risk | Impact | Mitigation |
|---|---|---|
| Legacy CUDA/PyTorch incompatible with recent GPU | Model cannot run on GPU | CPU safe default; explicit GPU gate; real smoke test |
| GMIC/GLAM error correlation | Reduced ensemble diversity | Preserve individual scores; measure experimentally; document limitation |
| Dataset domain shift | Metrics vary across sources | Report per-dataset provenance and final combined metrics carefully |
| Uncalibrated scores | Misinterpretation as clinical probability | Use `malignancy_score`; no learned calibration |
| Incorrect label harmonization | Invalid research conclusion | Explicit `source_manifest.csv`; no BI-RADS-to-cancer inference |
| Dataset/license misuse | Ethical/legal problem | Explicit manual authorized acquisition when required |
| Docker socket exposure | Local privilege risk | Socket only in the single lightweight model-runner; FastAPI/UI and temporary model containers have no socket; local research only |
| False negative | Clinical risk if misused | Human-in-the-loop; research-only warning; prioritize Sensitivity/FN in selection |
| Historical CUDA base image unavailable | Upstream model image cannot be rebuilt | Auditable `FROM`-only compatibility patch to an available CUDA 10.1/Ubuntu 18.04 NVIDIA image; record hashes and reason |
| Upstream Dockerfile changes after compatibility rule was defined | Patch could target the wrong runtime definition | Exact first-line drift guard; fail explicitly and require a new reviewed project version |
| XAI unavailable | Reduced interpretability | Report missing artifact; never create synthetic heatmap |
| Dataset incompatible with four-view NYU input | CBIS-DDSM cases may not satisfy the current ensemble input contract | `dataset_pipeline.inspect` measures complete L-CC/R-CC/L-MLO/R-MLO studies first; incomplete cases are rejected and never synthesized |
| Small diagnostic sample (for example 5 benign / 5 malignant) | ROC-AUC point estimates change materially when only a few positive-negative pairs change order; apparent version-to-version gains may be noise | Report stratified-bootstrap 95% CI, pairwise AUC step, keep diagnostic runs ineligible for freeze, and defer conclusions to the preregistered Configuration/Final Test workflow |
