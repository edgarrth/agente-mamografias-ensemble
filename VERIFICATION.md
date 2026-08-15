# Verification — v0.16

## Validaciones de empaquetado

- Python `compileall`.
- Pytest completo.
- YAML parse de `config/*.yaml`.
- Sintaxis `bash -n` de scripts shell.
- `.env.example` conserva el perfil de workstation validado: GMIC/NYU/GLAM en GPU, `ALLOW_GPU=true`, GPU 0.
- Dockerfiles exponen `VERSION` dentro de `/app/VERSION` y `/runner/VERSION`.
- Test del filtro de access logs: primer health, fallo y recuperación se registran; repeticiones del mismo estado se suprimen.
- Test del parche GMIC: Dockerfile contiene `torch.div(max_linear_idx, W_map, rounding_mode="floor")` y la configuración declara el cambio.
- Test de `build_revision`: GMIC=2, NYU/GLAM=1; el Runner invalida el probe cuando reconstruye por cambio de revisión.
- Integridad ZIP y checksums internos.

## Evidencia real previa en workstation

Todos los runtimes GPU pasaron `gpu_probe` y smoke test. CBIS-DDSM v0.15 pasó download/reuse, inspect y prepare. La primera prueba end-to-end de 5 estudios falló en GMIC con `top_k_prop_y <= 0.0`; ese fallo es el motivo del parche v0.16.

## Pendiente obligatorio

El paquete no afirma que el fix GMIC v0.16 esté validado hasta ejecutar en la workstation:

1. `model_tools.ensure_gpu --models gmic` (rebuild revision 2),
2. `model_tools.gpu_probe --models gmic`,
3. `tests_flow.normal --datasets cbis_ddsm --samples 5 --max-runtime-minutes 120`.

Docker Engine no está disponible en el entorno de empaquetado, por lo que `docker compose config/build` debe verificarse en la workstation objetivo.

## v0.17 validation additions

- `model_tools.validate_gpu --models all` expands to GMIC/NYU/GLAM.
- Subsets preserve caller order and remove duplicates.
- Validation phase order is all ensures → all probes → all smoke tests.
- `force_rebuild` is propagated to `/ensure-gpu` and the Runner rebuild decision.
- GPU-device guard prevents a CPU smoke test from being reported as GPU validation.
- Per-run JSON evidence is written to `workspace/output/model_validation/`.


## v0.18 validation additions

- GMIC Blackwell `build_revision=3` forces one auditable rebuild.
- Dockerfile patch requires the exact upstream `left_benign`/`right_benign` access anchors and fails build on upstream drift.
- Missing benign labels map to `NaN` output metadata; malignant labels remain required.
- Batch generator is verified not to invent independent benign ground truth.
- NYU/GLAM revisions remain unchanged.
