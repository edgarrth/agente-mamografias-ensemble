# Migración v0.7 — habilitación del GPU host en Fedora Remix / WSL2

## Motivo

Las tres imágenes reales de GMIC, DMV-CNN/NYU y GLAM ya construyen y pasan `smoke_test` en CPU. La workstation también ve la RTX 5060 Ti mediante `nvidia-smi`, pero Docker Engine falla al ejecutar un contenedor con `--gpus all` con:

```text
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Esto ubica el problema en la integración **host WSL2 → Docker Engine → NVIDIA Container Toolkit/CDI**, no en GMIC, NYU, GLAM ni en el Model Runner.

## Regla WSL2

No instalar un driver NVIDIA Linux dentro de WSL. El driver de Windows expone CUDA a WSL. Esta migración instala únicamente **NVIDIA Container Toolkit** en la distribución Fedora WSL donde se ejecuta `dockerd`.

## Opción recomendada: script incluido

Detener el proyecto sin borrar volúmenes:

```bash
docker compose down --remove-orphans
```

Ejecutar:

```bash
./scripts/setup-nvidia-container-toolkit-fedora-wsl.sh
```

El script:

1. confirma que `nvidia-smi` ya funciona en WSL;
2. agrega el repositorio RPM oficial de NVIDIA Container Toolkit;
3. instala `nvidia-container-toolkit`;
4. ejecuta `nvidia-ctk runtime configure --runtime=docker`;
5. reinicia el Docker Engine nativo de Fedora WSL;
6. activa/refresca `nvidia-cdi-refresh` cuando existe;
7. genera `/var/run/cdi/nvidia.yaml` explícitamente;
8. lista los dispositivos CDI descubiertos.

Luego validar:

```bash
./scripts/gpu-doctor.sh
```

El resultado esperado es:

```text
GPU_HOST_READY
```

También debe aparecer un dispositivo como:

```text
nvidia.com/gpu=0
nvidia.com/gpu=all
```

## Validación Docker directa

Después del doctor:

```bash
docker run --rm --gpus all \
  nvidia/cudagl:10.1-devel-ubuntu18.04 \
  nvidia-smi
```

Este comando solo verifica que Docker pueda entregar la GPU al contenedor. Todavía no valida que PyTorch 1.1/CUDA 10.1 de los modelos legacy pueda ejecutar kernels en una RTX 5060 Ti Blackwell.

## Volver a levantar el prototipo

```bash
docker compose up -d --build
```

Mantener inicialmente:

```env
MODEL_DEVICE=cpu
ALLOW_LEGACY_GPU=false
```

Hasta que la prueba Docker anterior sea correcta. La prueba de compatibilidad de cada modelo con GPU se hace después y de forma independiente.

## Trazabilidad

Este cambio no altera modelos, pesos, checkpoints, preprocessing ni ensemble. Solo añade diagnóstico y procedimiento reproducible de configuración del host para habilitar GPU en Docker.
