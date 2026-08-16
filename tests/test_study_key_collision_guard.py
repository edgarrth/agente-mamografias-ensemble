from pathlib import Path
import pandas as pd
import pytest
from mammography_agent import metarepo_format


def test_build_batch_rejects_sanitized_study_id_collisions(monkeypatch, tmp_path):
    image = tmp_path / "image.png"; image.write_bytes(b"x")
    monkeypatch.setattr(metarepo_format, "safe_workspace_path", lambda value: image)
    df = pd.DataFrame([
        {"study_id": "A/B", "l_cc": "x", "r_cc": "x", "l_mlo": "x", "r_mlo": "x", "ground_truth": 0},
        {"study_id": "A?B", "l_cc": "x", "r_cc": "x", "l_mlo": "x", "r_mlo": "x", "ground_truth": 1},
    ])
    with pytest.raises(ValueError, match="Sanitized study_id collision"):
        metarepo_format.build_batch(df, tmp_path / "batch")
