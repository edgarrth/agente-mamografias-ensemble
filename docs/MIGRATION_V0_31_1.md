# Migration v0.31.0 → v0.31.1

v0.31.1 is a storage-management patch for formal chunked experiments. It preserves all v0.31.0 scientific decisions and frozen Configuration artifacts.

## Preserved unchanged
- RSNA prepared data and formal manifests
- 30% Configuration / 70% Final split and seed 42
- Diagnostic exclusions
- GMIC / NYU / GLAM checkpoints and runtimes
- Orientation policy and resolved manifests
- Raw model predictions and prediction hashes
- Expanded 40 × 17 grid and 5-fold CV results
- Frozen ensemble weights and threshold
- Final-Test isolation

## New cleanup behavior
After a formal chunk is completed successfully, v0.31.1 prunes only regenerable heavyweight work products. Resume-critical evidence remains in place. Existing v0.31.0 Configuration chunks can be validated first with a dry-run and then pruned with `--apply`.
