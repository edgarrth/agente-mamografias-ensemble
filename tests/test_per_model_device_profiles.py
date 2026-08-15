from pathlib import Path
import yaml


def test_gpu_runtime_profile_is_model_metadata_not_env():
    env_text = Path('.env.example').read_text(encoding='utf-8')
    compose_text = Path('docker-compose.yml').read_text(encoding='utf-8')
    cfg = yaml.safe_load(Path('config/models.yaml').read_text(encoding='utf-8'))

    assert 'GPU_RUNTIME_PROFILE' not in env_text
    assert 'GPU_RUNTIME_PROFILE' not in compose_text
    assert cfg['models']['gmic']['gpu_compatibility']['profile'] == 'blackwell-cu128'


def test_device_selection_is_per_model_deployment_configuration():
    compose = yaml.safe_load(Path('docker-compose.yml').read_text(encoding='utf-8'))
    env = compose['services']['model-runner']['environment']

    assert env['DEFAULT_MODEL_DEVICE'] == '${DEFAULT_MODEL_DEVICE:-cpu}'
    assert env['GMIC_DEVICE'] == '${GMIC_DEVICE:-cpu}'
    assert env['NYU_DEVICE'] == '${NYU_DEVICE:-cpu}'
    assert env['GLAM_DEVICE'] == '${GLAM_DEVICE:-cpu}'
    assert 'MODEL_DEVICE' not in env
    assert 'GPU_RUNTIME_PROFILE' not in env
    assert 'ALLOW_LEGACY_GPU' not in env


def test_runner_resolves_profile_from_models_yaml_and_device_per_model():
    text = Path('model_runner/api.py').read_text(encoding='utf-8')
    assert 'os.getenv(f"{model.upper()}_DEVICE", DEFAULT_MODEL_DEVICE)' in text
    assert 'configured_profile = str(compat.get("profile", "")).strip().lower()' in text
    assert 'GPU_PROFILE_MISMATCH' not in text
    assert 'GPU_RUNTIME_PROFILE' not in text
