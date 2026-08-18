from pathlib import Path
import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_all_configuration_additions_have_required_fields_and_unique_ids():
    cfg = yaml.safe_load((_root() / "config" / "config_additions.yaml").read_text(encoding="utf-8"))
    additions = cfg["additions"]
    required = {"id", "name", "value", "reason"}
    ids = []
    for index, item in enumerate(additions, start=1):
        assert required <= set(item), f"entry #{index} missing {sorted(required - set(item))}: {item}"
        ids.append(item["id"])
    assert len(ids) == len(set(ids)), "configuration addition ids must be unique"
    assert additions[-1]["id"] == "ADD-094"
    assert additions[-1]["name"] == "rsna_study_ground_truth_v1"


def test_configuration_additions_logger_runs_with_packaged_config(tmp_path, monkeypatch):
    # Patch module globals used by ensure_workspace/log output, while retaining packaged config.
    import mammography_agent.config as config
    import mammography_agent.workspace as workspace
    import mammography_agent.logging_utils as logging_utils

    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(logging_utils, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setenv("CONFIG_ROOT", str(_root() / "config"))

    # load_yaml reads CONFIG_ROOT from mammography_agent.config.CONFIG_ROOT, so patch it explicitly.
    monkeypatch.setattr(config, "CONFIG_ROOT", _root() / "config")
    out = logging_utils.log_configuration_additions()
    text = out.read_text(encoding="utf-8")
    assert "ADD-081 | orientation_counterfactual_diagnostic" in text
    assert "ADD-083 | nyu_preprocess_only_pythonpath_fix" in text
