from pathlib import Path
import json
import pandas as pd

from mammography_agent import pipeline


def test_infer_three_uses_model_specific_preprocessed_dirs_and_explicit_identity(monkeypatch, tmp_path):
    batch_dir = tmp_path / "run" / "model_batch"
    seen_pre = {}

    def fake_build_batch(df, batch):
        batch.mkdir(parents=True, exist_ok=True)
        images = batch / "images"; images.mkdir()
        pkl = batch / "data.pkl"; pkl.write_bytes(b"x")
        # Manifest order deliberately differs from image-level parser order below.
        pd.DataFrame({
            "position": [0, 1],
            "study_id": ["Study B", "Study A"],
            "study_key": ["Study_B", "Study_A"],
        }).to_csv(batch / "study_order.csv", index=False)
        return images, pkl

    def fake_run_model(model, run_id, image_dir, data_pickle, output_file, preprocessed_dir):
        seen_pre[model] = Path(preprocessed_dir)
        Path(output_file).write_text("placeholder\n", encoding="utf-8")
        artifact = Path(preprocessed_dir) / f"{model}.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"png")
        return {
            "status": "SUCCESS",
            "output_file": output_file,
            "xai_artifacts": [] if model == "nyu" else [str(artifact)],
            "resource_metrics": {"elapsed_seconds": 1.0, "max_gpu_memory_mib": 100.0},
        }

    def fake_image_parser(path, model):
        if model == "gmic":
            return pd.DataFrame({"study_key": ["Study_A", "Study_B"], "gmic_score": [0.11, 0.82]})
        return pd.DataFrame({"study_key": ["Study_A", "Study_B"], "glam_score": [0.22, 0.73]})

    def fake_nyu(path, order):
        return pd.DataFrame({"study_id": ["Study B", "Study A"], "nyu_score": [0.63, 0.34]})

    monkeypatch.setattr(pipeline, "build_batch", fake_build_batch)
    monkeypatch.setattr(pipeline, "run_model", fake_run_model)
    monkeypatch.setattr(pipeline, "parse_image_level", fake_image_parser)
    monkeypatch.setattr(pipeline, "parse_nyu", fake_nyu)

    df = pd.DataFrame({
        "study_id": ["Study B", "Study A"],
        "patient_id": ["P2", "P1"],
        "ground_truth": [1, 0],
        "dataset_source": ["cbis_ddsm", "cbis_ddsm"],
    })
    out = pipeline._infer_three(df, tmp_path / "run", "r1")

    # Explicit mapping, not groupby row order: Study B must receive 0.82/0.73.
    b = out.set_index("study_id").loc["Study B"]
    a = out.set_index("study_id").loc["Study A"]
    assert b.gmic_score == 0.82 and b.glam_score == 0.73
    assert a.gmic_score == 0.11 and a.glam_score == 0.22

    assert seen_pre["gmic"].as_posix().endswith("preprocessed/gmic")
    assert seen_pre["nyu"].as_posix().endswith("preprocessed/nyu")
    assert seen_pre["glam"].as_posix().endswith("preprocessed/glam")
    assert len(set(seen_pre.values())) == 3

    xai = json.loads((tmp_path / "run" / "xai_artifacts.json").read_text())
    assert len(xai["gmic"]) == 1
    assert xai["nyu"] == []
    assert len(xai["glam"]) == 1


def test_chunk_evidence_is_aggregated_to_run_root(tmp_path):
    run = tmp_path / "normal"
    c0 = run / "chunks" / "0000"; c0.mkdir(parents=True)
    c1 = run / "chunks" / "0001"; c1.mkdir(parents=True)
    (c0 / "xai_artifacts.json").write_text(json.dumps({"gmic": ["g0"], "nyu": [], "glam": ["l0"]}))
    (c1 / "xai_artifacts.json").write_text(json.dumps({"gmic": ["g1"], "nyu": [], "glam": ["l1"]}))
    pd.DataFrame([{"model": "gmic", "elapsed_seconds": 1.0}]).to_csv(c0 / "resource_metrics.csv", index=False)
    pd.DataFrame([{"model": "glam", "elapsed_seconds": 2.0}]).to_csv(c1 / "resource_metrics.csv", index=False)

    pipeline._aggregate_chunk_evidence(run, [(0, c0), (1, c1)])

    xai = json.loads((run / "xai_artifacts.json").read_text())
    assert xai["gmic"] == ["g0", "g1"]
    assert xai["glam"] == ["l0", "l1"]
    resources = pd.read_csv(run / "resource_metrics.csv")
    assert resources.chunk.tolist() == [0, 1]
