from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd

from .adapters import ManifestDatasetAdapter
from .manifest import REQUIRED
from ..workspace import safe_workspace_path
from ..logging_utils import audit


ADAPTER_ID = "official_rsna_challenge_train_v030"
POLICY_ID = "RSNA_REQUIRED_FOUR_VIEWS_V1"
SELECTION_POLICY = "DETERMINISTIC_LABEL_BLIND_SHA256_V1"
REQUIRED_VIEWS = ("L_CC", "R_CC", "L_MLO", "R_MLO")
REQUIRED_VIEW_SET = set(REQUIRED_VIEWS)
REQUIRED_METADATA_COLUMNS = ("patient_id", "image_id", "laterality", "view", "cancer")
STANDARD_VIEWS = {"CC", "MLO"}


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _norm_side(value: object) -> str:
    text = _norm(value).upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return ""


def _norm_view(value: object) -> str:
    text = _norm(value).upper().replace("-", "")
    return text if text in STANDARD_VIEWS else ""


def canonical_view(laterality: object, view: object) -> str:
    side = _norm_side(laterality)
    normalized_view = _norm_view(view)
    return f"{side}_{normalized_view}" if side and normalized_view else ""


def deterministic_selection_key(patient_id: object, canonical: object, image_id: object) -> str:
    payload = f"{POLICY_ID}|{_norm(patient_id)}|{_norm(canonical)}|{_norm(image_id)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RSNADatasetAdapter(ManifestDatasetAdapter):
    """Adapter for the manually acquired RSNA Breast Cancer Detection challenge train set.

    Scientific policy:
    - The prototype never downloads RSNA automatically and never modifies raw DICOM/CSV files.
    - ``train.csv`` is authoritative for patient/image identity, laterality, view and ``cancer`` label.
    - A study is ensemble-compatible when it has at least one L_CC, R_CC, L_MLO and R_MLO image.
      It is not required to contain exactly four DICOM objects.
    - If a canonical view has multiple images, exactly one is selected using a deterministic,
      label-blind SHA-256 rule. Cancer, BI-RADS, pixels, model scores and AUC never affect selection.
    - Non-standard views and unselected duplicate standard views are retained in audit manifests.
    - Study ground truth is malignant when either breast is labelled malignant; otherwise benign
      when both observed breast labels are benign. Any within-breast label conflict excludes the study.
    """

    def _rsna_paths(self) -> dict[str, Path]:
        base = self._paths()
        extra = {
            "dicom_index": self.cfg.get(
                "dicom_index_cache", "/workspace/runtime/dataset_cache/rsna_dicom_index.csv"
            ),
            "selected_views": self.cfg.get(
                "selected_views_manifest", "/workspace/datasets/manifests/rsna_selected_views.csv"
            ),
            "unselected_duplicate_views": self.cfg.get(
                "unselected_duplicate_views_manifest",
                "/workspace/datasets/rejected/rsna_unselected_duplicate_views.csv",
            ),
            "nonstandard_views": self.cfg.get(
                "nonstandard_views_manifest", "/workspace/datasets/rejected/rsna_nonstandard_views.csv"
            ),
            "incomplete": self.cfg.get(
                "incomplete_manifest", "/workspace/datasets/rejected/rsna_incomplete_studies.csv"
            ),
            "conflicts": self.cfg.get(
                "conflicts_manifest", "/workspace/datasets/rejected/rsna_label_conflicts.csv"
            ),
            "all_required_four_view": self.cfg.get(
                "all_required_four_view_manifest",
                "/workspace/datasets/manifests/rsna_all_required_four_view.csv",
            ),
        }
        return {**base, **{k: safe_workspace_path(v) for k, v in extra.items()}}

    def _dataset_root(self, strict: bool = False) -> Path | None:
        raw = self._rsna_paths()["raw"]
        candidates = [raw, raw / "rsna"]
        for candidate in candidates:
            if (candidate / "train.csv").is_file() and (candidate / "train_images").is_dir():
                return candidate.resolve()
        if raw.exists():
            found = sorted(
                {
                    p.parent.resolve()
                    for p in raw.rglob("train.csv")
                    if (p.parent / "train_images").is_dir()
                },
                key=str,
            )
            if len(found) == 1:
                return found[0]
            if len(found) > 1 and strict:
                raise ValueError(f"Multiple RSNA train roots found under {raw}: {found}")
        if strict:
            raise FileNotFoundError(
                f"RSNA train.csv + train_images/ not found under {raw}. "
                "Expected either raw/rsna/train.csv or raw/train.csv."
            )
        return None

    def status(self) -> dict:
        p = self._rsna_paths()
        root = self._dataset_root(strict=False)
        if p["canonical"].is_file():
            state = "AVAILABLE"
        elif p["source"].is_file():
            state = "INSPECTED_NOT_PREPARED"
        elif root is not None:
            state = "READY_FOR_INSPECT"
        elif p["raw"].exists() and any(p["raw"].iterdir()):
            state = "METADATA_OR_DICOM_REQUIRED"
        else:
            state = "NOT_DOWNLOADED"
        return {
            "dataset": self.key,
            "name": self.cfg["name"],
            "status": state,
            "raw_dir": str(p["raw"]),
            "dataset_root": str(root) if root else None,
            "canonical_manifest": str(p["canonical"]),
            "adapter": ADAPTER_ID,
            "benchmark_policy": POLICY_ID,
            "automatic_download": False,
        }

    def download(self) -> dict:
        p = self._rsna_paths()
        p["raw"].mkdir(parents=True, exist_ok=True)
        instructions = p["raw"] / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# RSNA Breast Cancer Detection manual acquisition — v0.30.0\n\n"
            "The prototype never downloads RSNA automatically. Acquire the authorized official "
            "challenge/research distribution manually and preserve the raw files unchanged.\n\n"
            "Expected layout (the additional nested `rsna/` directory is accepted):\n\n"
            "```text\n"
            "/workspace/datasets/raw/rsna/rsna/train.csv\n"
            "/workspace/datasets/raw/rsna/rsna/train_images/<patient_id>/<image_id>.dcm\n"
            "```\n\n"
            f"Official information: {self.cfg['official_information']}\n",
            encoding="utf-8",
        )
        current = self.status()
        audit("RSNA_MANUAL_ACQUISITION_STATUS", dataset=self.key, status=current["status"])
        return {**current, "instructions": str(instructions), "download_performed": False}

    def _load_metadata(self) -> tuple[pd.DataFrame, dict]:
        root = self._dataset_root(strict=True)
        assert root is not None
        path = root / "train.csv"
        df = pd.read_csv(
            path,
            dtype={"patient_id": str, "image_id": str, "laterality": str, "view": str},
        )
        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"RSNA train.csv missing required columns: {missing}")
        out = df.copy()
        out["patient_id"] = out["patient_id"].map(_norm)
        out["image_id"] = out["image_id"].map(_norm)
        out["laterality"] = out["laterality"].map(_norm_side)
        out["view_raw"] = out["view"].map(lambda x: _norm(x).upper())
        out["canonical_view"] = [canonical_view(a, b) for a, b in zip(out["laterality"], out["view_raw"])]
        out["cancer"] = pd.to_numeric(out["cancer"], errors="coerce")
        invalid_label = ~out["cancer"].isin([0, 1])
        if invalid_label.any():
            examples = out.loc[invalid_label, ["patient_id", "image_id", "cancer"]].head(10).to_dict("records")
            raise ValueError(f"RSNA train.csv has non-binary/invalid cancer labels: {examples}")
        out["cancer"] = out["cancer"].astype(int)
        if out["patient_id"].eq("").any() or out["image_id"].eq("").any():
            raise ValueError("RSNA train.csv contains blank patient_id/image_id values")
        duplicate_identity = out.duplicated(["patient_id", "image_id"], keep=False)
        if duplicate_identity.any():
            examples = out.loc[duplicate_identity, ["patient_id", "image_id"]].head(10).to_dict("records")
            raise ValueError(f"RSNA train.csv contains duplicate patient_id/image_id identities: {examples}")
        return out, {
            "path": str(path),
            "rows": int(len(out)),
            "patients": int(out.patient_id.nunique()),
        }

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})

    def _build_dicom_index(self, metadata: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        import pydicom

        p = self._rsna_paths()
        cache = p["dicom_index"]
        required_cache_columns = {
            "csv_row_index", "patient_id", "image_id", "path", "exists", "is_dicom", "read_error",
            "transfer_syntax_uid", "transfer_syntax_name", "bits_stored", "bits_allocated", "rows",
            "columns", "photometric", "dicom_laterality",
        }
        if cache.is_file() and not force and bool(self.cfg.get("reuse_dicom_index_cache", True)):
            try:
                cached = pd.read_csv(cache, dtype={"patient_id": str, "image_id": str})
                if required_cache_columns.issubset(cached.columns) and len(cached) == len(metadata):
                    audit("RSNA_DICOM_INDEX_CACHE_REUSED", dataset=self.key, rows=int(len(cached)))
                    return cached
            except Exception:
                pass

        root = self._dataset_root(strict=True)
        assert root is not None
        image_root = root / "train_images"
        rows = []
        for pos, (_, row) in enumerate(metadata.iterrows()):
            path = image_root / str(row.patient_id) / f"{row.image_id}.dcm"
            rec = {
                "csv_row_index": int(pos),
                "patient_id": str(row.patient_id),
                "image_id": str(row.image_id),
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "is_dicom": False,
                "read_error": "",
                "transfer_syntax_uid": "",
                "transfer_syntax_name": "",
                "bits_stored": "",
                "bits_allocated": "",
                "rows": "",
                "columns": "",
                "photometric": "",
                "dicom_laterality": "",
            }
            if path.is_file():
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    rec["is_dicom"] = True
                    uid = ds.file_meta.TransferSyntaxUID
                    rec["transfer_syntax_uid"] = str(uid)
                    rec["transfer_syntax_name"] = getattr(uid, "name", "")
                    rec["bits_stored"] = _norm(getattr(ds, "BitsStored", ""))
                    rec["bits_allocated"] = _norm(getattr(ds, "BitsAllocated", ""))
                    rec["rows"] = _norm(getattr(ds, "Rows", ""))
                    rec["columns"] = _norm(getattr(ds, "Columns", ""))
                    rec["photometric"] = _norm(getattr(ds, "PhotometricInterpretation", ""))
                    rec["dicom_laterality"] = _norm_side(
                        getattr(ds, "ImageLaterality", getattr(ds, "Laterality", ""))
                    )
                except Exception as exc:
                    rec["read_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
            if pos and pos % 5000 == 0:
                audit("RSNA_DICOM_INDEX_PROGRESS", dataset=self.key, processed=pos, total=int(len(metadata)))
        result = pd.DataFrame(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache, index=False)
        audit("RSNA_DICOM_INDEX_BUILT", dataset=self.key, rows=int(len(result)))
        return result

    @staticmethod
    def _label_summary(metadata: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
        summaries: dict[str, dict] = {}
        conflicts = []
        for patient, group in metadata.groupby("patient_id", sort=True):
            side_labels: dict[str, int] = {}
            patient_conflict = False
            for side in ("L", "R"):
                values = sorted(set(int(x) for x in group.loc[group.laterality.eq(side), "cancer"].dropna().tolist()))
                if len(values) > 1:
                    patient_conflict = True
                    conflicts.append({"patient_id": str(patient), "laterality": side, "labels": "|".join(map(str, values))})
                elif len(values) == 1:
                    side_labels[side] = int(values[0])
            observed = list(side_labels.values())
            study_gt = 1 if 1 in observed else 0 if len(observed) == 2 and all(v == 0 for v in observed) else None
            summaries[str(patient)] = {
                "left_ground_truth": side_labels.get("L"),
                "right_ground_truth": side_labels.get("R"),
                "ground_truth": study_gt,
                "label_conflict": bool(patient_conflict),
            }
        return summaries, pd.DataFrame(conflicts, columns=["patient_id", "laterality", "labels"])

    def inspect(self, force_dicom_index: bool = False) -> dict:
        p = self._rsna_paths()
        root = self._dataset_root(strict=False)
        if root is None:
            return {
                **self.status(),
                "status": "METADATA_OR_DICOM_REQUIRED",
                "next_action": "Place the manually acquired RSNA train.csv and train_images/ under raw/rsna/rsna/ and rerun inspect.",
                "dicom_index_started": False,
            }

        metadata, metadata_info = self._load_metadata()
        dicom = self._build_dicom_index(metadata, force=force_dicom_index)
        if len(dicom) != len(metadata):
            raise ValueError(f"RSNA DICOM index row count mismatch: metadata={len(metadata)}, index={len(dicom)}")

        merged = metadata.reset_index(drop=True).copy()
        merged["source_path"] = dicom["path"].astype(str).values
        merged["dicom_exists"] = self._bool_series(dicom["exists"]).values
        merged["dicom_valid"] = self._bool_series(dicom["is_dicom"]).values
        merged["dicom_read_error"] = dicom["read_error"].fillna("").astype(str).values
        valid_pixel_sources = merged.dicom_exists & merged.dicom_valid & merged.dicom_read_error.eq("")

        summaries, conflicts = self._label_summary(metadata)
        conflict_patients = set(conflicts.patient_id.astype(str)) if not conflicts.empty else set()
        p["conflicts"].parent.mkdir(parents=True, exist_ok=True)
        conflicts.to_csv(p["conflicts"], index=False)

        nonstandard = merged[~merged.canonical_view.isin(REQUIRED_VIEW_SET)].copy()
        p["nonstandard_views"].parent.mkdir(parents=True, exist_ok=True)
        nonstandard.to_csv(p["nonstandard_views"], index=False)

        standard = merged[merged.canonical_view.isin(REQUIRED_VIEW_SET) & valid_pixel_sources].copy()
        selected_rows = []
        unselected_rows = []
        complete_rows = []
        incomplete_rows = []

        for patient, group in standard.groupby("patient_id", sort=True):
            patient = str(patient)
            summary = summaries.get(patient, {
                "left_ground_truth": None, "right_ground_truth": None, "ground_truth": None, "label_conflict": False
            })
            counts = group.canonical_view.value_counts().to_dict()
            missing = sorted(REQUIRED_VIEW_SET - set(counts))
            reason = []
            if missing:
                reason.append("missing_required_views")
            if patient in conflict_patients or bool(summary.get("label_conflict")):
                reason.append("label_conflict")
            if summary.get("ground_truth") is None:
                reason.append("incomplete_bilateral_ground_truth")
            if reason:
                incomplete_rows.append({
                    "patient_id": patient,
                    "study_id": f"RSNA_{patient}",
                    "missing_views": "|".join(missing),
                    "reason": "|".join(reason),
                    "standard_valid_images": int(len(group)),
                    "view_counts": "|".join(f"{v}:{int(counts.get(v, 0))}" for v in REQUIRED_VIEWS),
                })
                continue

            chosen_by_view = {}
            for view in REQUIRED_VIEWS:
                candidates = group[group.canonical_view.eq(view)].copy()
                candidates["selection_key"] = candidates["image_id"].map(
                    lambda image_id: deterministic_selection_key(patient, view, image_id)
                )
                candidates = candidates.sort_values(["selection_key", "image_id", "source_path"], kind="stable")
                chosen = candidates.iloc[0]
                chosen_by_view[view] = chosen
                for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
                    rec = candidate.to_dict()
                    rec.update({
                        "study_id": f"RSNA_{patient}",
                        "candidate_count": int(len(candidates)),
                        "selection_policy": SELECTION_POLICY,
                        "selection_key": str(candidate["selection_key"]),
                        "selection_rank": int(rank),
                        "selected": bool(rank == 1),
                    })
                    if rank == 1:
                        selected_rows.append(rec)
                    else:
                        unselected_rows.append(rec)

            complete_rows.append({
                "study_id": f"RSNA_{patient}",
                "patient_id": patient,
                "ground_truth": int(summary["ground_truth"]),
                "left_ground_truth": int(summary["left_ground_truth"]),
                "right_ground_truth": int(summary["right_ground_truth"]),
                "l_cc": str(chosen_by_view["L_CC"].source_path),
                "r_cc": str(chosen_by_view["R_CC"].source_path),
                "l_mlo": str(chosen_by_view["L_MLO"].source_path),
                "r_mlo": str(chosen_by_view["R_MLO"].source_path),
                "horizontal_flip": "NO",
                "benchmark_policy": POLICY_ID,
                "selection_policy": SELECTION_POLICY,
                "standard_valid_images": int(len(group)),
                "duplicate_standard_images_unselected": int(len(group) - 4),
            })

        # Patients with zero valid standard DICOMs would not appear in the groupby above.
        seen = set(standard.patient_id.astype(str).unique())
        for patient in sorted(set(metadata.patient_id.astype(str).unique()) - seen):
            incomplete_rows.append({
                "patient_id": patient,
                "study_id": f"RSNA_{patient}",
                "missing_views": "|".join(REQUIRED_VIEWS),
                "reason": "no_valid_standard_dicom",
                "standard_valid_images": 0,
                "view_counts": "",
            })

        complete = pd.DataFrame(complete_rows)
        selected = pd.DataFrame(selected_rows)
        unselected = pd.DataFrame(unselected_rows)
        incomplete = pd.DataFrame(incomplete_rows)

        p["all_required_four_view"].parent.mkdir(parents=True, exist_ok=True)
        complete.to_csv(p["all_required_four_view"], index=False)
        p["selected_views"].parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(p["selected_views"], index=False)
        p["unselected_duplicate_views"].parent.mkdir(parents=True, exist_ok=True)
        unselected.to_csv(p["unselected_duplicate_views"], index=False)
        p["incomplete"].parent.mkdir(parents=True, exist_ok=True)
        incomplete.to_csv(p["incomplete"], index=False)

        source_cols = [
            *REQUIRED,
            "left_ground_truth", "right_ground_truth", "horizontal_flip", "benchmark_policy", "selection_policy"
        ]
        if complete.empty:
            source = pd.DataFrame(columns=source_cols)
        else:
            source = complete[source_cols].copy()
        p["source"].parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(p["source"], index=False)

        gt_counts = source.ground_truth.value_counts().to_dict() if not source.empty else {}
        transfer_counts = dicom.transfer_syntax_uid.fillna("").astype(str).value_counts().to_dict()
        photo_counts = dicom.photometric.fillna("").astype(str).value_counts().to_dict()
        bits_counts = pd.to_numeric(dicom.bits_stored, errors="coerce").dropna().astype(int).value_counts().to_dict()
        missing_dicom = int((~self._bool_series(dicom["exists"])).sum())
        invalid_dicom = int((~self._bool_series(dicom["is_dicom"])).sum())

        result = {
            "dataset": self.key,
            "status": "INSPECTED",
            "adapter": ADAPTER_ID,
            "benchmark_policy": POLICY_ID,
            "selection_policy": SELECTION_POLICY,
            "raw_dir": str(p["raw"]),
            "dataset_root": str(root),
            "train_csv": metadata_info["path"],
            "csv_rows": int(metadata_info["rows"]),
            "csv_patients": int(metadata_info["patients"]),
            "dicom_files_indexed": int(len(dicom)),
            "dicom_headers_valid": int(self._bool_series(dicom["is_dicom"]).sum()),
            "missing_dicom": missing_dicom,
            "invalid_dicom": invalid_dicom,
            "required_four_view_studies": int(len(source)),
            "incomplete_or_rejected_studies": int(len(incomplete)),
            "selected_canonical_images": int(len(selected)),
            "unselected_duplicate_standard_images": int(len(unselected)),
            "nonstandard_images": int(len(nonstandard)),
            "ground_truth_counts": {
                "BENIGN": int(gt_counts.get(0, 0)),
                "MALIGNANT": int(gt_counts.get(1, 0)),
            },
            "label_conflicts": int(len(conflicts)),
            "transfer_syntax_counts": {str(k): int(v) for k, v in transfer_counts.items() if str(k)},
            "photometric_counts": {str(k): int(v) for k, v in photo_counts.items() if str(k)},
            "bits_stored_counts": {str(k): int(v) for k, v in bits_counts.items()},
            "source_manifest": str(p["source"]),
            "canonical_manifest": str(p["canonical"]),
            "dicom_index": str(p["dicom_index"]),
            "selected_views_manifest": str(p["selected_views"]),
            "unselected_duplicate_views_manifest": str(p["unselected_duplicate_views"]),
            "nonstandard_views_manifest": str(p["nonstandard_views"]),
            "incomplete_manifest": str(p["incomplete"]),
            "conflicts_manifest": str(p["conflicts"]),
            "ensemble_compatible": bool(len(source) > 0 and len(incomplete) == 0 and missing_dicom == 0 and invalid_dicom == 0),
        }
        audit("RSNA_INSPECTED", **result)
        return result

    def verify_integrity(self) -> dict:
        try:
            result = self.inspect(force_dicom_index=False)
        except Exception as exc:
            return {"dataset": self.key, "valid": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {
            "dataset": self.key,
            "valid": bool(result.get("ensemble_compatible")),
            "studies": int(result.get("required_four_view_studies", 0)),
            "reason": "ok" if result.get("ensemble_compatible") else "RSNA inspection reported incompatibilities",
        }

    def prepare(self) -> dict:
        p = self._rsna_paths()
        inspection = self.inspect(force_dicom_index=False)
        if inspection.get("status") != "INSPECTED":
            return {**inspection, "converted_studies": 0}
        src = pd.read_csv(p["source"], dtype={"study_id": str, "patient_id": str})
        if src.empty:
            return {**inspection, "status": "INSUFFICIENT_BENCHMARK_STUDIES", "converted_studies": 0}
        missing = [c for c in REQUIRED if c not in src.columns]
        if missing:
            raise ValueError(f"generated RSNA source_manifest.csv missing {missing}")

        p["processed"].mkdir(parents=True, exist_ok=True)
        p["canonical"].parent.mkdir(parents=True, exist_ok=True)
        image_dir = p["processed"] / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        out = []
        total = int(len(src))
        for i, (_, r) in enumerate(src.iterrows(), start=1):
            row = {
                "study_id": str(r.study_id),
                "patient_id": str(r.patient_id),
                "ground_truth": int(r.ground_truth),
                "left_ground_truth": int(r.left_ground_truth),
                "right_ground_truth": int(r.right_ground_truth),
                "horizontal_flip": str(r.get("horizontal_flip", "NO")),
                "benchmark_policy": str(r.get("benchmark_policy", POLICY_ID)),
                "selection_policy": str(r.get("selection_policy", SELECTION_POLICY)),
            }
            for col, view in (("l_cc", "L_CC"), ("r_cc", "R_CC"), ("l_mlo", "L_MLO"), ("r_mlo", "R_MLO")):
                source = safe_workspace_path(str(r[col]))
                dest = image_dir / f"{row['study_id']}_{view}.png"
                self._convert_to_png(source, dest)
                row[col] = str(dest)
            out.append(row)
            if i % 250 == 0 or i == total:
                audit("RSNA_PREPARE_PROGRESS", dataset=self.key, converted_studies=i, total_studies=total)
        canonical = pd.DataFrame(out)
        canonical.to_csv(p["canonical"], index=False)
        audit("DATASET_PREPARED", dataset=self.key, studies=len(canonical), manifest=str(p["canonical"]), adapter=ADAPTER_ID)
        return {
            **inspection,
            "status": "AVAILABLE",
            "studies": int(len(canonical)),
            "converted_studies": int(len(canonical)),
            "converted_images": int(len(canonical) * 4),
            "manifest": str(p["canonical"]),
        }
