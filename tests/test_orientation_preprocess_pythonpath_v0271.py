from pathlib import Path


def test_preprocess_only_exports_nyu_repo_root_into_pythonpath():
    text = Path("model_runner/api.py").read_text(encoding="utf-8")
    assert 'repo_pythonpath = "/home/bcc/breast_cancer_classifier" if model == "nyu" else target' in text
    assert 'export PYTHONPATH=' in text
    assert '${{PYTHONPATH:+:$PYTHONPATH}}' in text


def test_upstream_direct_script_import_semantics_are_reproduced(tmp_path):
    """Regression proof for the exact ModuleNotFoundError seen in v0.27.0.

    An individual script below src/ cannot import the top-level src package unless
    the repository root is explicitly placed on PYTHONPATH, matching NYU upstream
    guidance for individual-script execution.
    """
    import os, subprocess, sys

    repo = tmp_path / "breast_cancer_classifier"
    (repo / "src" / "cropping").mkdir(parents=True)
    (repo / "src" / "utilities").mkdir(parents=True)
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "utilities" / "__init__.py").write_text("")
    (repo / "src" / "utilities" / "pickling.py").write_text("VALUE = 27\n")
    script = repo / "src" / "cropping" / "crop_mammogram.py"
    script.write_text("import src.utilities.pickling as pickling\nprint(pickling.VALUE)\n")

    clean = dict(os.environ)
    clean.pop("PYTHONPATH", None)
    failed = subprocess.run([sys.executable, '-S', str(script)], cwd=repo, env=clean, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "No module named 'src" in failed.stderr

    fixed_env = dict(clean)
    fixed_env["PYTHONPATH"] = str(repo)
    passed = subprocess.run([sys.executable, '-S', str(script)], cwd=repo, env=fixed_env, text=True, capture_output=True)
    assert passed.returncode == 0, passed.stderr
    assert passed.stdout.strip() == "27"
