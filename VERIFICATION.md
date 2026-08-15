# Verification — v0.14

The package must pass before distribution:

- Python `compileall`.
- Full `pytest` suite.
- YAML parsing for every file under `config/`.
- `bash -n` for shell scripts.
- Docker Compose configuration resolution when Docker is available.
- `.env.example` validated three-GPU profile test.
- CBIS-DDSM missing-metadata preflight test.
- CBIS-DDSM metadata filename-alias test.
- ZIP integrity test.
- Package SHA-256 manifest verification.

Runtime evidence from the target workstation already exists for GMIC, NYU and GLAM GPU smoke tests. Full real-dataset inspection/preparation remains workstation validation work because the 163.51 GB CBIS-DDSM collection is not bundled with this package.
