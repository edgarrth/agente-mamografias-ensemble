import logging
from model_runner import api


def test_high_value_runner_event_is_written_to_console_and_jsonl(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(api, "WORKSPACE", tmp_path)
    caplog.set_level(logging.INFO, logger="mammography-model-runner")
    api.log("MODEL_RUN_STARTED", model="gmic", run_id="r1", device="gpu")
    assert "MODEL_RUN_STARTED" in caplog.text
    text = (tmp_path / "logs" / "model_runner.jsonl").read_text()
    assert '"event": "MODEL_RUN_STARTED"' in text


def test_low_value_command_event_remains_persistent_but_not_console_spam(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(api, "WORKSPACE", tmp_path)
    caplog.set_level(logging.INFO, logger="mammography-model-runner")
    api.log("COMMAND", model="gmic", cmd=["echo", "x"])
    assert "COMMAND" not in caplog.text
    assert '"event": "COMMAND"' in (tmp_path / "logs" / "model_runner.jsonl").read_text()
