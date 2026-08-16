from pathlib import Path

def test_model_runner_exposes_preprocess_only_endpoint():
    text=Path('model_runner/api.py').read_text(encoding='utf-8')
    assert '@app.post("/models/{model}/preprocess")' in text
    assert 'classifier_inference_performed' in text
    assert 'src/cropping/crop_mammogram.py' in text
    assert 'src/optimal_centers/get_optimal_centers.py' in text

def test_model_client_has_preprocess_call():
    text=Path('mammography_agent/model_client.py').read_text(encoding='utf-8')
    assert 'def preprocess_model(' in text
    assert '/preprocess' in text
