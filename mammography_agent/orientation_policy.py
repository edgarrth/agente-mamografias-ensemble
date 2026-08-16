from __future__ import annotations
from pathlib import Path
import datetime as dt
import json
import pickle
import pandas as pd

from .metarepo_format import build_batch
from .model_client import preprocess_model

VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
POLICY_ID = "strict_four_view_gap_v1"


def _normalize_flip(value: object) -> str:
    return "YES" if str(value if value is not None else "NO").strip().upper() == "YES" else "NO"


def _toggle(value: object) -> str:
    return "NO" if _normalize_flip(value) == "YES" else "YES"


def _scalar(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return float(value)
    except Exception:
        return None


def _label_blind(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # The upstream crop/center stages do not consume cancer labels.  Zero them anyway
    # so the orientation policy is mechanically unable to use ground truth.
    for col in ("ground_truth", "left_ground_truth", "right_ground_truth"):
        out[col] = 0
    if "horizontal_flip" not in out.columns:
        out["horizontal_flip"] = "NO"
    out["horizontal_flip"] = out["horizontal_flip"].map(_normalize_flip)
    return out


def _run_preflight(df: pd.DataFrame, root: Path, run_id: str) -> pd.DataFrame:
    batch = root / "model_batch"
    batch.mkdir(parents=True, exist_ok=True)
    blinded = _label_blind(df)
    images, pkl_path = build_batch(blinded, batch)
    pre = batch / "preprocessed" / "nyu"
    result = preprocess_model("nyu", run_id, str(images), str(pkl_path), str(pre))
    center = Path(result["center_data"])
    with center.open("rb") as fh:
        exams = pickle.load(fh)
    if len(exams) != len(blinded):
        raise ValueError(f"Orientation preflight count mismatch: {len(exams)} metadata exams for {len(blinded)} studies")
    rows = []
    for pos, (_, src) in enumerate(blinded.reset_index(drop=True).iterrows()):
        exam = exams[pos]
        distances = exam.get("distance_from_starting_side", {}) or {}
        for view in VIEWS:
            d = _scalar(distances.get(view))
            rows.append({
                "position": pos,
                "study_id": str(src.study_id),
                "model_view": view,
                "horizontal_flip": _normalize_flip(exam.get("horizontal_flip", src.get("horizontal_flip", "NO"))),
                "distance_from_starting_side": d,
                "distance_nonzero": bool(d is not None and abs(d) > 1e-9),
            })
    return pd.DataFrame(rows)


def resolve_orientation(df: pd.DataFrame, output_dir: str | Path, run_id: str) -> pd.DataFrame:
    """Apply a strict label-independent orientation policy before classifier inference.

    Trigger: all four unique views have non-zero upstream distance_from_starting_side.
    Acceptance: toggling the study-level horizontal_flip makes all four distances zero.
    Ground truth, model scores and AUC are never consulted.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original_df = df.copy().reset_index(drop=True)
    if "horizontal_flip" not in original_df.columns:
        original_df["horizontal_flip"] = "NO"
    original_df["horizontal_flip"] = original_df["horizontal_flip"].map(_normalize_flip)

    original = _run_preflight(original_df, out / "original", f"{run_id}-orientation-original")
    original.to_csv(out / "orientation_original_views.csv", index=False)
    counts = original.groupby("study_id")["distance_nonzero"].sum().astype(int)
    suspects = set(counts[counts == len(VIEWS)].index.astype(str))

    resolution_rows = []
    evidence = original.copy()
    evidence = evidence.rename(columns={
        "horizontal_flip": "original_horizontal_flip",
        "distance_from_starting_side": "original_distance_from_starting_side",
        "distance_nonzero": "original_distance_nonzero",
    })
    corrected = original_df.copy()

    if suspects:
        variant = original_df[original_df.study_id.astype(str).isin(suspects)].copy().reset_index(drop=True)
        variant["horizontal_flip"] = variant["horizontal_flip"].map(_toggle)
        counter = _run_preflight(variant, out / "counterfactual", f"{run_id}-orientation-counterfactual")
        counter.to_csv(out / "orientation_counterfactual_views.csv", index=False)
        csmall = counter.rename(columns={
            "horizontal_flip": "counterfactual_horizontal_flip",
            "distance_from_starting_side": "counterfactual_distance_from_starting_side",
            "distance_nonzero": "counterfactual_distance_nonzero",
        })
        evidence = evidence.merge(
            csmall[["study_id", "model_view", "counterfactual_horizontal_flip", "counterfactual_distance_from_starting_side", "counterfactual_distance_nonzero"]],
            on=["study_id", "model_view"], how="left", validate="one_to_one",
        )
        for sid in sorted(suspects):
            og = evidence[evidence.study_id.astype(str) == sid]
            orig_nonzero = int(og["original_distance_nonzero"].fillna(False).sum())
            cf_nonzero = int(og["counterfactual_distance_nonzero"].astype("boolean").fillna(False).astype(int).sum())
            accept = bool(orig_nonzero == 4 and cf_nonzero == 0)
            old = _normalize_flip(original_df.loc[original_df.study_id.astype(str) == sid, "horizontal_flip"].iloc[0])
            new = _toggle(old) if accept else old
            corrected.loc[corrected.study_id.astype(str) == sid, "horizontal_flip"] = new
            resolution_rows.append({
                "study_id": sid, "policy_id": POLICY_ID, "triggered": True,
                "original_horizontal_flip": old, "counterfactual_horizontal_flip": _toggle(old),
                "original_nonzero_views": orig_nonzero, "counterfactual_nonzero_views": cf_nonzero,
                "orientation_changed": accept, "resolved_horizontal_flip": new,
                "decision_reason": "counterfactual_eliminated_all_four_starting_side_gaps" if accept else "strict_acceptance_rule_not_met",
            })

    for sid in original_df.study_id.astype(str):
        if sid in suspects:
            continue
        old = _normalize_flip(original_df.loc[original_df.study_id.astype(str) == sid, "horizontal_flip"].iloc[0])
        resolution_rows.append({
            "study_id": sid, "policy_id": POLICY_ID, "triggered": False,
            "original_horizontal_flip": old, "counterfactual_horizontal_flip": None,
            "original_nonzero_views": int(counts.get(sid, 0)), "counterfactual_nonzero_views": None,
            "orientation_changed": False, "resolved_horizontal_flip": old,
            "decision_reason": "four_view_gap_trigger_not_met",
        })

    resolution = pd.DataFrame(resolution_rows).sort_values("study_id").reset_index(drop=True)
    evidence.to_csv(out / "orientation_view_evidence.csv", index=False)
    resolution.to_csv(out / "orientation_resolution.csv", index=False)
    corrected.to_csv(out / "resolved_manifest.csv", index=False)
    changed = resolution[resolution.orientation_changed.fillna(False)]
    summary = {
        "policy_id": POLICY_ID,
        "studies": int(len(original_df)),
        "triggered_studies": int(resolution.triggered.fillna(False).sum()),
        "orientation_changed_studies": int(len(changed)),
        "changed_study_ids": changed.study_id.astype(str).tolist(),
        "decision_rule": "trigger only when all 4 views have non-zero distance_from_starting_side; accept toggle only when all 4 become zero",
        "ground_truth_used": False,
        "model_scores_used": False,
        "auc_used": False,
        "classifier_inference_performed": False,
    }
    (out / "orientation_policy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Automatic Orientation Resolution", "",
        f"- **policy**: {POLICY_ID}",
        f"- **studies**: {len(original_df)}",
        f"- **triggered**: {summary['triggered_studies']}",
        f"- **orientation changed**: {summary['orientation_changed_studies']}",
        f"- **ground truth used**: False", "- **model scores/AUC used**: False",
        "- **classifier inference during preflight**: False", "",
        "The policy uses the upstream crop/center geometry only. A flip is accepted only when a four-view starting-side gap is completely eliminated by the counterfactual orientation.",
    ]
    (out / "orientation_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corrected


def audit_existing_run(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Run only the v0.27 orientation policy against an existing selected manifest.

    This executes upstream NYU crop+center preprocessing but no classifier inference.
    """
    run = Path(run_dir).resolve()
    manifest = run / "selected_studies.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"selected_studies.csv not found: {manifest}")
    df = pd.read_csv(manifest)
    if output_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path("/workspace/output/analyses") / f"orientation-policy-{stamp}"
    else:
        out = Path(output_dir).resolve()
    resolve_orientation(df, out, f"orientation-policy-{run.name}")
    return out
