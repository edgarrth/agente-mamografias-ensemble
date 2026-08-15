from mammography_agent import model_client


def test_single_runner_url_is_used_for_all_models():
    assert model_client.RUNNER_URL.endswith("model-runner:8010")
    assert model_client._model_path("gmic").endswith("/models/gmic")
    assert model_client._model_path("nyu").endswith("/models/nyu")
    assert model_client._model_path("glam").endswith("/models/glam")


def test_runner_rejects_unknown_model():
    try:
        model_client._model_path("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model must fail explicitly")
