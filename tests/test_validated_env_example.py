from pathlib import Path


def test_env_example_matches_validated_three_gpu_workstation_profile():
    text = Path('.env.example').read_text(encoding='utf-8')
    required = {
        'DEFAULT_MODEL_DEVICE': 'cpu',
        'GMIC_DEVICE': 'gpu',
        'NYU_DEVICE': 'gpu',
        'GLAM_DEVICE': 'gpu',
        'ALLOW_GPU': 'true',
        'GPU_NUMBER': '0',
    }
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        values[k] = v
    for key, expected in required.items():
        assert values.get(key) == expected
    assert 'GPU_RUNTIME_PROFILE' not in values
    assert 'MODEL_DEVICE' not in values
    assert 'ALLOW_LEGACY_GPU' not in values
