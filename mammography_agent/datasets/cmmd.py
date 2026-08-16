from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib

import pandas as pd

from .adapters import ManifestDatasetAdapter
from .manifest import REQUIRED
from ..workspace import safe_workspace_path
from ..logging_utils import audit


CLINICAL_FILENAME = "CMMD_clinicaldata_revision.xlsx"
REQUIRED_CLINICAL_COLUMNS = ("ID1", "LeftRight", "classification")
VIEW_CODE_MAP = {
    "399162004": "CC",   # cranio-caudal
    "399368009": "MLO",  # medio-lateral oblique
}
REQUIRED_VIEWS = {"L_CC", "R_CC", "L_MLO", "R_MLO"}
CLASS_MAP = {"BENIGN": 0, "MALIGNANT": 1}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(value: object) -> str:
    return str(value or "").strip()


def _norm_side(value: object) -> str:
    value = _norm(value).upper()
    if value in {"L", "LEFT"}:
        return "L"
    if value in {"R", "RIGHT"}:
        return "R"
    return ""


def _norm_class(value: object) -> int | None:
    return CLASS_MAP.get(_norm(value).upper())


def _cohort(patient_id: str) -> str:
    if patient_id.startswith("D1-"):
        return "D1"
    if patient_id.startswith("D2-"):
        return "D2"
    return "OTHER"


class CMMDDatasetAdapter(ManifestDatasetAdapter):
    """Adapter for the manually acquired TCIA CMMD release.

    Scientific policy in v0.29.0:
    - Raw DICOM and the clinical XLSX are never downloaded by the prototype.
    - View is read from DICOM ViewCodeSequence.CodeValue, not ViewPosition.
    - Four-view compatibility requires exactly L-CC, R-CC, L-MLO, R-MLO.
    - Clinical rows are indexed by (patient, breast side); duplicate sides with conflicting
      labels are rejected instead of collapsed.
    - Study-level cancer is MALIGNANT if either explicitly labelled breast is malignant;
      otherwise BENIGN when at least one explicit benign label exists and no malignant
      label exists.
    - The canonical binary benchmark exposed as dataset key ``cmmd`` uses CMMD1 (D1)
      four-view studies with explicit labels for both breasts. CMMD2 is retained in audit
      artifacts but excluded from this binary benchmark because it is a malignant/subtype
      cohort and would strongly confound class with cohort.
    """

    def _cmmd_paths(self) -> dict[str, Path]:
        base = self._paths()
        extra = {
            "dicom_index": self.cfg.get(
                "dicom_index_cache", "/workspace/runtime/dataset_cache/cmmd_dicom_index.csv"
            ),
            "clinical_catalog": self.cfg.get(
                "clinical_catalog", "/workspace/datasets/manifests/cmmd_clinical_rows.csv"
            ),
            "all_four_view": self.cfg.get(
                "all_four_view_manifest", "/workspace/datasets/manifests/cmmd_all_four_view.csv"
            ),
            "incomplete": self.cfg.get(
                "incomplete_manifest", "/workspace/datasets/rejected/cmmd_incomplete_studies.csv"
            ),
            "excluded": self.cfg.get(
                "excluded_manifest", "/workspace/datasets/rejected/cmmd_nonbenchmark_four_view.csv"
            ),
            "conflicts": self.cfg.get(
                "conflicts_manifest", "/workspace/datasets/rejected/cmmd_clinical_conflicts.csv"
            ),
        }
        return {**base, **{k: safe_workspace_path(v) for k, v in extra.items()}}

    def _clinical_candidates(self) -> list[Path]:
        raw = self._cmmd_paths()["raw"]
        candidates = []
        for p in (raw / "metadata" / CLINICAL_FILENAME, raw / CLINICAL_FILENAME):
            if p.is_file():
                candidates.append(p.resolve())
        if candidates:
            return sorted(set(candidates), key=str)
        if raw.exists():
            return sorted((p.resolve() for p in raw.rglob(CLINICAL_FILENAME) if p.is_file()), key=str)
        return []

    def _clinical_file(self, strict: bool = False) -> Path | None:
        candidates = self._clinical_candidates()
        if not candidates:
            if strict:
                raise FileNotFoundError(
                    f"{CLINICAL_FILENAME} not found under {self._cmmd_paths()['raw']}; "
                    "place the official file manually under raw/cmmd/metadata/."
                )
            return None
        if len(candidates) > 1:
            hashes = {_sha256(p) for p in candidates}
            if len(hashes) != 1:
                if strict:
                    raise ValueError(f"Multiple non-identical {CLINICAL_FILENAME} files: {candidates}")
                return None
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]

    def _raw_has_dicom(self) -> bool:
        p = self._cmmd_paths()
        cache = p["dicom_index"]
        if bool(self.cfg.get("reuse_dicom_index_cache", True)) and cache.is_file():
            try:
                df = pd.read_csv(cache, usecols=["is_dicom"])
                if not df.empty and df.is_dicom.fillna(False).astype(bool).any():
                    return True
            except Exception:
                pass
        raw = p["raw"]
        if not raw.exists():
            return False
        for candidate in raw.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".dcm", ".dicom", ""}:
                return True
        return False

    def status(self) -> dict:
        p = self._cmmd_paths()
        clinical = self._clinical_file(strict=False)
        dicom_present = self._raw_has_dicom()
        if p["canonical"].is_file():
            state = "AVAILABLE"
        elif p["source"].is_file():
            state = "INSPECTED_NOT_PREPARED"
        elif clinical and dicom_present:
            state = "READY_FOR_INSPECT"
        elif dicom_present and not clinical:
            state = "METADATA_REQUIRED"
        elif clinical and not dicom_present:
            state = "DICOM_DOWNLOAD_REQUIRED"
        else:
            state = "NOT_DOWNLOADED"
        return {
            "dataset": self.key,
            "name": self.cfg["name"],
            "status": state,
            "raw_dir": str(p["raw"]),
            "canonical_manifest": str(p["canonical"]),
            "clinical_file": str(clinical) if clinical else None,
            "clinical_manual_download": True,
            "dicom_present": bool(dicom_present),
            "dicom_auto_download": False,
            "metadata_auto_download": False,
            "adapter": "official_tcia_cmmd_v029",
            "benchmark_policy": "CMMD1_D1_EXACT_FOUR_VIEW_BILATERAL_LABELS",
        }

    def download(self) -> dict:
        """Write manual acquisition guidance only; never access the network."""
        p = self._cmmd_paths()
        p["raw"].mkdir(parents=True, exist_ok=True)
        instructions = p["raw"] / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# CMMD manual acquisition policy — v0.29.0\n\n"
            "This prototype never downloads CMMD files. Download the official DICOM images "
            "and `CMMD_clinicaldata_revision.xlsx` manually from TCIA. Preserve the raw tree.\n\n"
            "Recommended metadata path:\n\n"
            "`/workspace/datasets/raw/cmmd/metadata/CMMD_clinicaldata_revision.xlsx`\n\n"
            f"Official information: {self.cfg['official_information']}\n",
            encoding="utf-8",
        )
        current = self.status()
        audit("CMMD_MANUAL_ACQUISITION_STATUS", dataset=self.key, status=current["status"])
        return {**current, "instructions": str(instructions), "download_performed": False}

    def _load_clinical(self) -> tuple[pd.DataFrame, dict]:
        path = self._clinical_file(strict=True)
        assert path is not None
        df = pd.read_excel(path, engine="openpyxl")
        missing = [c for c in REQUIRED_CLINICAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{CLINICAL_FILENAME} missing required columns: {missing}")
        out = pd.DataFrame({
            "patient_id": df["ID1"].map(_norm),
            "side": df["LeftRight"].map(_norm_side),
            "classification_text": df["classification"].map(lambda x: _norm(x).upper()),
            "ground_truth": df["classification"].map(_norm_class),
            "age": df["Age"] if "Age" in df.columns else pd.NA,
            "abnormality": df["abnormality"] if "abnormality" in df.columns else pd.NA,
            "subtype": df["subtype"] if "subtype" in df.columns else pd.NA,
        })
        out = out[out.patient_id.ne("")].copy()
        out["cohort"] = out.patient_id.map(_cohort)
        invalid_side = int(out.side.eq("").sum())
        invalid_class = int(out.ground_truth.isna().sum())
        info = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": int(len(out)),
            "patients": int(out.patient_id.nunique()),
            "invalid_side_rows": invalid_side,
            "invalid_classification_rows": invalid_class,
        }
        return out, info

    def _candidate_dicom_files(self) -> list[Path]:
        raw = self._cmmd_paths()["raw"]
        if not raw.exists():
            return []
        return [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".dcm", ".dicom", ""}]

    def _build_dicom_index(self, force: bool = False) -> pd.DataFrame:
        import pydicom
        p = self._cmmd_paths()
        cache = p["dicom_index"]
        if cache.is_file() and not force and bool(self.cfg.get("reuse_dicom_index_cache", True)):
            return pd.read_csv(cache, dtype={"patient_id": str, "study_uid": str, "series_uid": str, "sop_uid": str, "view_code": str})

        rows = []
        for path in self._candidate_dicom_files():
            rec = {
                "path": str(path.resolve()),
                "is_dicom": False,
                "patient_id": "",
                "study_uid": "",
                "series_uid": "",
                "sop_uid": "",
                "laterality": "",
                "view": "",
                "view_code": "",
                "canonical_view": "",
                "bits_stored": "",
                "bits_allocated": "",
                "rows": "",
                "columns": "",
                "photometric": "",
                "read_error": "",
            }
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                rec["is_dicom"] = True
                rec["patient_id"] = _norm(getattr(ds, "PatientID", ""))
                rec["study_uid"] = _norm(getattr(ds, "StudyInstanceUID", ""))
                rec["series_uid"] = _norm(getattr(ds, "SeriesInstanceUID", ""))
                rec["sop_uid"] = _norm(getattr(ds, "SOPInstanceUID", ""))
                rec["laterality"] = _norm_side(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")))
                seq = getattr(ds, "ViewCodeSequence", None)
                code = ""
                if seq:
                    try:
                        code = _norm(getattr(seq[0], "CodeValue", ""))
                    except Exception:
                        code = ""
                rec["view_code"] = code
                rec["view"] = VIEW_CODE_MAP.get(code, "")
                if rec["laterality"] and rec["view"]:
                    rec["canonical_view"] = f"{rec['laterality']}_{rec['view']}"
                rec["bits_stored"] = _norm(getattr(ds, "BitsStored", ""))
                rec["bits_allocated"] = _norm(getattr(ds, "BitsAllocated", ""))
                rec["rows"] = _norm(getattr(ds, "Rows", ""))
                rec["columns"] = _norm(getattr(ds, "Columns", ""))
                rec["photometric"] = _norm(getattr(ds, "PhotometricInterpretation", ""))
            except Exception as exc:
                rec["read_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)

        df = pd.DataFrame(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df

    @staticmethod
    def _clinical_patient_summary(clinical: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
        summaries: dict[str, dict] = {}
        conflict_rows = []
        for patient, group in clinical.groupby("patient_id", sort=True):
            side_labels: dict[str, int] = {}
            has_conflict = False
            for side in ("L", "R"):
                values = sorted(set(int(x) for x in group.loc[group.side.eq(side), "ground_truth"].dropna().tolist()))
                if len(values) > 1:
                    has_conflict = True
                    conflict_rows.append({"patient_id": patient, "side": side, "labels": "|".join(map(str, values))})
                elif len(values) == 1:
                    side_labels[side] = values[0]
            observed = list(side_labels.values())
            study_gt = 1 if 1 in observed else 0 if observed and all(v == 0 for v in observed) else None
            summaries[str(patient)] = {
                "cohort": _cohort(str(patient)),
                "left_ground_truth": side_labels.get("L"),
                "right_ground_truth": side_labels.get("R"),
                "ground_truth": study_gt,
                "clinical_rows": int(len(group)),
                "clinical_side_conflict": has_conflict,
            }
        return summaries, pd.DataFrame(conflict_rows, columns=["patient_id", "side", "labels"])

    def inspect(self, force_dicom_index: bool = False) -> dict:
        p = self._cmmd_paths()
        clinical_file = self._clinical_file(strict=False)
        if not clinical_file:
            return {
                **self.status(),
                "status": "METADATA_REQUIRED",
                "next_action": f"Place {CLINICAL_FILENAME} manually under raw/cmmd/metadata/ and rerun inspect.",
                "dicom_index_started": False,
            }
        if not self._raw_has_dicom():
            return {
                **self.status(),
                "status": "DICOM_DOWNLOAD_REQUIRED",
                "next_action": "Download the official CMMD DICOM collection manually from TCIA under raw/cmmd/ and rerun inspect.",
                "dicom_index_started": False,
            }

        clinical, clinical_info = self._load_clinical()
        dicom = self._build_dicom_index(force=force_dicom_index)
        valid = dicom[dicom.is_dicom.fillna(False).astype(bool)].copy()
        summaries, conflicts = self._clinical_patient_summary(clinical)

        p["clinical_catalog"].parent.mkdir(parents=True, exist_ok=True)
        clinical.to_csv(p["clinical_catalog"], index=False)
        p["conflicts"].parent.mkdir(parents=True, exist_ok=True)
        conflicts.to_csv(p["conflicts"], index=False)

        all_four = []
        incomplete = []
        for patient, group in valid.groupby("patient_id", sort=True):
            views = group.canonical_view.fillna("").astype(str).tolist()
            view_counts = pd.Series([v for v in views if v]).value_counts().to_dict()
            exact = len(group) == 4 and set(view_counts) == REQUIRED_VIEWS and all(int(view_counts[v]) == 1 for v in REQUIRED_VIEWS)
            summary = summaries.get(str(patient), {
                "cohort": _cohort(str(patient)), "left_ground_truth": None, "right_ground_truth": None,
                "ground_truth": None, "clinical_rows": 0, "clinical_side_conflict": False,
            })
            study_uids = sorted(set(x for x in group.study_uid.fillna("").astype(str) if x))
            base = {
                "patient_id": str(patient),
                "study_id": f"CMMD_{patient}",
                "study_uid": study_uids[0] if len(study_uids) == 1 else "|".join(study_uids),
                **summary,
                "dicom_images": int(len(group)),
                "view_set": "|".join(sorted(v for v in view_counts)),
            }
            if not exact:
                incomplete.append({**base, "missing_views": "|".join(sorted(REQUIRED_VIEWS - set(view_counts))), "duplicate_or_extra_views": bool(len(group) != len(set(views)))})
                continue
            by_view = {str(r.canonical_view): str(r.path) for _, r in group.iterrows()}
            all_four.append({
                **base,
                "l_cc": by_view["L_CC"],
                "r_cc": by_view["R_CC"],
                "l_mlo": by_view["L_MLO"],
                "r_mlo": by_view["R_MLO"],
                "horizontal_flip": "NO",
                "all_images_8bit": bool(pd.to_numeric(group.bits_stored, errors="coerce").eq(8).all()),
            })

        all_four_df = pd.DataFrame(all_four)
        incomplete_df = pd.DataFrame(incomplete)
        p["all_four_view"].parent.mkdir(parents=True, exist_ok=True)
        all_four_df.to_csv(p["all_four_view"], index=False)
        p["incomplete"].parent.mkdir(parents=True, exist_ok=True)
        incomplete_df.to_csv(p["incomplete"], index=False)

        if all_four_df.empty:
            benchmark = all_four_df.copy()
            excluded = all_four_df.copy()
        else:
            bilateral_known = all_four_df.left_ground_truth.notna() & all_four_df.right_ground_truth.notna()
            benchmark_mask = (
                all_four_df.cohort.eq("D1")
                & bilateral_known
                & all_four_df.ground_truth.notna()
                & ~all_four_df.clinical_side_conflict.astype(bool)
            )
            benchmark = all_four_df[benchmark_mask].copy()
            excluded = all_four_df[~benchmark_mask].copy()

        p["excluded"].parent.mkdir(parents=True, exist_ok=True)
        excluded.to_csv(p["excluded"], index=False)

        source_cols = [*REQUIRED, "left_ground_truth", "right_ground_truth", "horizontal_flip", "cmmd_cohort", "benchmark_policy"]
        source_rows = []
        for _, r in benchmark.iterrows():
            source_rows.append({
                "study_id": str(r.study_id),
                "patient_id": str(r.patient_id),
                "ground_truth": int(r.ground_truth),
                "l_cc": str(r.l_cc),
                "r_cc": str(r.r_cc),
                "l_mlo": str(r.l_mlo),
                "r_mlo": str(r.r_mlo),
                "left_ground_truth": int(r.left_ground_truth),
                "right_ground_truth": int(r.right_ground_truth),
                "horizontal_flip": "NO",
                "cmmd_cohort": str(r.cohort),
                "benchmark_policy": "D1_EXACT_FOUR_VIEW_BILATERAL_LABELS",
            })
        source = pd.DataFrame(source_rows, columns=source_cols)
        p["source"].parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(p["source"], index=False)

        cohort_counts = all_four_df.cohort.value_counts().to_dict() if not all_four_df.empty else {}
        benchmark_counts = pd.to_numeric(source.ground_truth, errors="coerce").value_counts().to_dict() if not source.empty else {}
        bits = pd.to_numeric(valid.bits_stored, errors="coerce").dropna().astype(int).value_counts().to_dict() if not valid.empty else {}

        result = {
            "dataset": self.key,
            "status": "INSPECTED",
            "raw_dir": str(p["raw"]),
            "clinical_file": str(clinical_file),
            "clinical_sha256": clinical_info["sha256"],
            "clinical_rows": clinical_info["rows"],
            "clinical_patients": clinical_info["patients"],
            "dicom_files_indexed": int(len(dicom)),
            "dicom_headers_valid": int(len(valid)),
            "dicom_patients": int(valid.patient_id.nunique()) if not valid.empty else 0,
            "four_view_patients_all_cmmd": int(len(all_four_df)),
            "two_or_incomplete_patients": int(len(incomplete_df)),
            "four_view_by_cohort": {str(k): int(v) for k, v in cohort_counts.items()},
            "benchmark_studies": int(len(source)),
            "benchmark_ground_truth_counts": {
                "BENIGN": int(benchmark_counts.get(0, 0)),
                "MALIGNANT": int(benchmark_counts.get(1, 0)),
            },
            "clinical_side_conflicts": int(len(conflicts)),
            "bits_stored_counts": {str(k): int(v) for k, v in bits.items()},
            "source_manifest": str(p["source"]),
            "all_four_view_manifest": str(p["all_four_view"]),
            "excluded_manifest": str(p["excluded"]),
            "incomplete_manifest": str(p["incomplete"]),
            "clinical_catalog": str(p["clinical_catalog"]),
            "dicom_index": str(p["dicom_index"]),
            "ensemble_compatible": bool(len(source)),
            "benchmark_policy": "CMMD1/D1 only; exact four-view; explicit bilateral clinical labels; study malignant if either breast malignant.",
            "note": "CMMD2 is retained for audit/external malignant-domain analysis but excluded from the binary benchmark because cohort and class are strongly confounded.",
        }
        audit("CMMD_INSPECTED", **{k: v for k, v in result.items() if k not in {"note"}})
        return result

    def verify_integrity(self) -> dict:
        try:
            result = self.inspect(force_dicom_index=False)
        except Exception as exc:
            return {"dataset": self.key, "valid": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {
            "dataset": self.key,
            "valid": result.get("status") == "INSPECTED" and result.get("benchmark_studies", 0) > 0,
            "benchmark_studies": int(result.get("benchmark_studies", 0)),
            "reason": "ok" if result.get("benchmark_studies", 0) > 0 else result.get("status", "inspection incomplete"),
        }

    def prepare(self) -> dict:
        p = self._cmmd_paths()
        inspection = self.inspect(force_dicom_index=False)
        if inspection.get("status") in {"METADATA_REQUIRED", "DICOM_DOWNLOAD_REQUIRED"}:
            return {**inspection, "converted_studies": 0}
        src = pd.read_csv(p["source"])
        if src.empty:
            if p["canonical"].exists():
                p["canonical"].unlink()
            return {**inspection, "status": "INSUFFICIENT_BENCHMARK_STUDIES", "converted_studies": 0}
        missing = [c for c in REQUIRED if c not in src.columns]
        if missing:
            raise ValueError(f"generated CMMD source_manifest.csv missing {missing}")

        p["processed"].mkdir(parents=True, exist_ok=True)
        p["canonical"].parent.mkdir(parents=True, exist_ok=True)
        image_dir = p["processed"] / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for _, r in src.iterrows():
            row = {
                "study_id": str(r.study_id),
                "patient_id": str(r.patient_id),
                "ground_truth": int(r.ground_truth),
                "left_ground_truth": int(r.left_ground_truth),
                "right_ground_truth": int(r.right_ground_truth),
                "horizontal_flip": str(r.get("horizontal_flip", "NO")),
                "cmmd_cohort": str(r.get("cmmd_cohort", "D1")),
                "benchmark_policy": str(r.get("benchmark_policy", "D1_EXACT_FOUR_VIEW_BILATERAL_LABELS")),
            }
            for col, view in (("l_cc", "L_CC"), ("r_cc", "R_CC"), ("l_mlo", "L_MLO"), ("r_mlo", "R_MLO")):
                source = safe_workspace_path(str(r[col]))
                dest = image_dir / f"{row['study_id']}_{view}.png"
                self._convert_to_png(source, dest)
                row[col] = str(dest)
            out.append(row)
        df = pd.DataFrame(out)
        df.to_csv(p["canonical"], index=False)
        audit("DATASET_PREPARED", dataset=self.key, studies=len(df), manifest=str(p["canonical"]), adapter="official_tcia_cmmd_v029")
        return {**inspection, "status": "AVAILABLE", "studies": int(len(df)), "converted_studies": int(len(df)), "manifest": str(p["canonical"])}
