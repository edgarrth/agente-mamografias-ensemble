from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import re
from typing import Iterable
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

from .adapters import ManifestDatasetAdapter
from .manifest import REQUIRED
from ..workspace import safe_workspace_path
from ..logging_utils import audit


OFFICIAL_METADATA_FILES = (
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
)

# Official TCIA supporting-data attachments. The DICOM image collection remains a
# separate NBIA Data Retriever transfer; these small CSV files can be acquired
# automatically and atomically without touching an existing DICOM tree.
OFFICIAL_METADATA_DOWNLOADS = {
    "mass_case_description_train_set.csv": {
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_train_set.csv?api=v2",
        "sha256": "a4bcc1c32bfd040212737e6e8c304819334abe717cfefff56431dc50ae6db723",
    },
    "mass_case_description_test_set.csv": {
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/22516629/mass_case_description_test_set.csv?api=v2",
        "sha256": "60f0b4504ec024106bbdd1555030d220163ea9da0ea4688473a4c9c3e8ab0457",
    },
    "calc_case_description_train_set.csv": {
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_train_set.csv?api=v2",
        "sha256": "3c7b07e8202f66709f446b9695e96c5d17484a65fda6a24d0170a0c1b81d56e3",
    },
    "calc_case_description_test_set.csv": {
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/22516629/calc_case_description_test_set.csv?api=v2",
        "sha256": "03a2f72efb6ed29f417e07f4598950e6a1e02e6b11250ff516ebe7924ace3689",
    },
}

AUXILIARY_METADATA_FILENAME = "metadata.csv"
AUXILIARY_METADATA_REQUIRED_COLUMNS = ("PatientID", "SeriesInstanceUID")

# TCIA has historically exposed the same four classification tables with UI labels
# such as “Mass-Training-Description”.  Accept those downloaded filenames too, while
# keeping one canonical identity per table so downstream provenance remains stable.
OFFICIAL_METADATA_ALIASES = {
    "mass_case_description_train_set.csv": ("mass-training-description.csv", "mass_training_description.csv"),
    "mass_case_description_test_set.csv": ("mass-test-description.csv", "mass_test_description.csv"),
    "calc_case_description_train_set.csv": ("calc-training-description.csv", "calc_training_description.csv"),
    "calc_case_description_test_set.csv": ("calc-test-description.csv", "calc_test_description.csv"),
}

PATHOLOGY_MAP = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

REQUIRED_METADATA_COLUMNS = (
    "patient_id",
    "left or right breast",
    "image view",
    "pathology",
    "image file path",
)

VIEW_COLUMNS = {
    ("LEFT", "CC"): "l_cc",
    ("RIGHT", "CC"): "r_cc",
    ("LEFT", "MLO"): "l_mlo",
    ("RIGHT", "MLO"): "r_mlo",
}


@dataclass(frozen=True)
class ResolvedImage:
    path: Path | None
    method: str
    candidates: int = 0


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_pathology(value: object) -> int | None:
    key = _norm_text(value).upper().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    return PATHOLOGY_MAP.get(key)


def normalize_laterality(value: object) -> str | None:
    key = _norm_text(value).upper()
    if key in {"L", "LEFT"}:
        return "LEFT"
    if key in {"R", "RIGHT"}:
        return "RIGHT"
    return None


def normalize_view(value: object) -> str | None:
    key = _norm_text(value).upper().replace("-", "")
    if key == "CC":
        return "CC"
    if key in {"MLO", "ML0"}:
        return "MLO"
    return None


def normalize_patient_id(value: object) -> str:
    text = _norm_text(value).upper()
    m = re.search(r"P_\d{5}", text)
    return m.group(0) if m else text


def _canonical_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        clean = re.sub(r"\s+", " ", str(col).strip().lower().replace("_", " "))
        aliases = {
            "patient id": "patient_id",
            "patient_id": "patient_id",
            "left or right breast": "left or right breast",
            "image view": "image view",
            "pathology": "pathology",
            "image file path": "image file path",
            "abnormality id": "abnormality id",
            "abnormality type": "abnormality type",
        }
        rename[col] = aliases.get(clean, clean)
    return df.rename(columns=rename)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CBISDDSMDatasetAdapter(ManifestDatasetAdapter):
    """Adapter for the official TCIA CBIS-DDSM DICOM + classification CSV release.

    It deliberately does not infer labels from BI-RADS. Ground truth comes only from the
    official pathology field. The current three-model ensemble requires four standard
    views, so incomplete exams are catalogued/rejected rather than synthesised.
    """

    def _cbis_paths(self) -> dict[str, Path]:
        base = self._paths()
        extra = {
            "metadata_catalog": self.cfg.get(
                "metadata_catalog", "/workspace/datasets/manifests/cbis_ddsm_metadata_rows.csv"
            ),
            "view_catalog": self.cfg.get(
                "view_catalog", "/workspace/datasets/manifests/cbis_ddsm_view_catalog.csv"
            ),
            "incomplete": self.cfg.get(
                "incomplete_manifest", "/workspace/datasets/rejected/cbis_ddsm_incomplete_studies.csv"
            ),
            "unresolved": self.cfg.get(
                "unresolved_manifest", "/workspace/datasets/rejected/cbis_ddsm_unresolved_metadata_rows.csv"
            ),
            "dicom_index": self.cfg.get(
                "dicom_index_cache", "/workspace/runtime/dataset_cache/cbis_ddsm_dicom_index.csv"
            ),
        }
        return {**base, **{k: safe_workspace_path(v) for k, v in extra.items()}}

    def _metadata_candidates(self) -> dict[str, list[Path]]:
        raw = self._cbis_paths()["raw"]
        result = {name: [] for name in OFFICIAL_METADATA_FILES}
        if not raw.exists():
            return result
        wanted: dict[str, str] = {}
        for canonical in OFFICIAL_METADATA_FILES:
            wanted[canonical.lower()] = canonical
            for alias in OFFICIAL_METADATA_ALIASES.get(canonical, ()):
                wanted[alias.lower()] = canonical

        # Fast path: v0.16 writes supporting data to raw/metadata and also accepts
        # files placed directly in raw. Avoid a deep WSL/NTFS tree walk when these
        # deterministic locations already contain everything required.
        for base in (raw / "metadata", raw):
            if not base.exists():
                continue
            for filename, canonical in wanted.items():
                candidate = base / filename
                if candidate.is_file():
                    result[canonical].append(candidate.resolve())
        if all(result[name] for name in OFFICIAL_METADATA_FILES):
            return result

        # Compatibility fallback for older arbitrary layouts. This is used only when
        # one or more required tables were not found in the fast locations.
        seen = {str(x) for values in result.values() for x in values}
        for candidate in raw.rglob("*.csv"):
            key = wanted.get(candidate.name.lower())
            resolved = candidate.resolve()
            if key and str(resolved) not in seen:
                result[key].append(resolved)
                seen.add(str(resolved))
        return result

    def _metadata_files(self, strict: bool = False) -> dict[str, Path]:
        candidates = self._metadata_candidates()
        selected: dict[str, Path] = {}
        errors = []
        for name, paths in candidates.items():
            if not paths:
                errors.append(f"missing {name}")
                continue
            if len(paths) == 1:
                selected[name] = paths[0]
                continue
            # Accept duplicate copies only when byte-identical; otherwise fail instead of
            # silently choosing different ground-truth metadata.
            hashes = {p: _sha256(p) for p in paths}
            if len(set(hashes.values())) != 1:
                errors.append(f"ambiguous non-identical copies of {name}: {[str(p) for p in paths]}")
                continue
            selected[name] = sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]
        if strict and errors:
            raise FileNotFoundError(
                "CBIS-DDSM official metadata incomplete/ambiguous under raw_dir: " + "; ".join(errors)
            )
        return selected

    def _metadata_download_specs(self) -> dict[str, dict[str, str]]:
        configured = self.cfg.get("official_metadata_downloads") or {}
        specs = {}
        for name in OFFICIAL_METADATA_FILES:
            base = dict(OFFICIAL_METADATA_DOWNLOADS[name])
            override = configured.get(name, {}) if isinstance(configured, dict) else {}
            if isinstance(override, dict):
                base.update({k: str(v) for k, v in override.items() if v is not None})
            specs[name] = base
        return specs

    def _validate_metadata_table(self, path: Path, canonical_name: str, verify_sha256: bool = False) -> dict:
        result = {"file": canonical_name, "path": str(path), "valid": False, "sha256": "", "missing_columns": []}
        try:
            digest = _sha256(path)
            header = _canonical_metadata_columns(pd.read_csv(path, nrows=0))
            missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in header.columns]
            result.update({"sha256": digest, "missing_columns": missing})
            expected = self._metadata_download_specs().get(canonical_name, {}).get("sha256", "")
            hash_ok = (not verify_sha256) or (not expected) or digest.lower() == str(expected).lower()
            result["expected_sha256"] = expected
            result["sha256_valid"] = bool(hash_ok)
            result["valid"] = not missing and bool(hash_ok)
            if missing:
                result["error"] = f"missing required columns: {missing}"
            elif not hash_ok:
                result["error"] = f"SHA-256 mismatch: expected {expected}, got {digest}"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _download_url_to_file(url: str, destination: Path, timeout: int = 120) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".part")
        if temp.exists():
            temp.unlink()
        request = Request(url, headers={"User-Agent": "mammography-ai-agent/0.15 research prototype"})
        try:
            with urlopen(request, timeout=timeout) as response, temp.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            temp.replace(destination)
        except Exception:
            if temp.exists():
                temp.unlink()
            raise

    def _ensure_official_metadata(self) -> dict:
        """Acquire only the four small official CSV tables; never transfer DICOM bytes."""
        p = self._cbis_paths()
        metadata_dir = p["raw"] / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        specs = self._metadata_download_specs()
        candidates = self._metadata_candidates()
        downloaded, reused, failed = [], [], []
        validations = {}

        for name in OFFICIAL_METADATA_FILES:
            existing = candidates.get(name, [])
            if existing:
                # _metadata_files already prevents silently selecting non-identical duplicates.
                selected = sorted(existing, key=lambda x: (len(x.parts), str(x)))[0]
                validation = self._validate_metadata_table(selected, name, verify_sha256=bool(specs[name].get("sha256")))
                validations[name] = validation
                if validation["valid"]:
                    reused.append(name)
                    continue
                failed.append({"file": name, "path": str(selected), "error": validation.get("error", "invalid metadata")})
                continue

            dest = metadata_dir / name
            try:
                self._download_url_to_file(specs[name]["url"], dest)
                validation = self._validate_metadata_table(dest, name, verify_sha256=bool(specs[name].get("sha256")))
                validations[name] = validation
                if not validation["valid"]:
                    failed.append({"file": name, "path": str(dest), "error": validation.get("error", "invalid metadata")})
                    continue
                downloaded.append(name)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                failed.append({"file": name, "path": str(dest), "error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                failed.append({"file": name, "path": str(dest), "error": f"{type(exc).__name__}: {exc}"})

        # Re-scan because newly downloaded files were not present in the initial candidates.
        final_candidates = self._metadata_candidates()
        complete = all(final_candidates.get(name) for name in OFFICIAL_METADATA_FILES) and not failed
        audit(
            "CBIS_DDSM_METADATA_DOWNLOAD",
            dataset=self.key,
            downloaded=downloaded,
            reused=reused,
            failed=failed,
            complete=bool(complete),
            dicom_download_performed=False,
        )
        return {
            "metadata_complete": bool(complete),
            "metadata_downloaded": downloaded,
            "metadata_reused": reused,
            "metadata_failed": failed,
            "metadata_validation": validations,
            "metadata_dir": str(metadata_dir),
        }

    def _auxiliary_metadata_candidates(self) -> list[Path]:
        raw = self._cbis_paths()["raw"]
        if not raw.exists():
            return []
        direct = []
        for candidate in (raw / "metadata" / AUXILIARY_METADATA_FILENAME, raw / AUXILIARY_METADATA_FILENAME):
            if candidate.is_file():
                direct.append(candidate.resolve())
        if direct:
            return sorted(set(direct), key=str)
        return sorted((x.resolve() for x in raw.rglob(AUXILIARY_METADATA_FILENAME) if x.is_file()), key=str)

    def _load_auxiliary_metadata(self) -> tuple[pd.DataFrame, dict]:
        candidates = self._auxiliary_metadata_candidates()
        if not candidates:
            return pd.DataFrame(), {"present": False, "used": False, "files": [], "rows": 0, "reason": "metadata.csv not present; optional"}
        if len(candidates) > 1:
            hashes = {_sha256(x) for x in candidates}
            if len(hashes) != 1:
                return pd.DataFrame(), {"present": True, "used": False, "files": [str(x) for x in candidates], "rows": 0, "reason": "multiple non-identical metadata.csv files"}
        path = candidates[0]
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            return pd.DataFrame(), {"present": True, "used": False, "files": [str(path)], "rows": 0, "reason": f"{type(exc).__name__}: {exc}"}
        normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
        patient_col = normalized.get("patientid")
        series_col = normalized.get("seriesinstanceuid")
        study_col = normalized.get("studyinstanceuid")
        if not patient_col or not series_col:
            return pd.DataFrame(), {
                "present": True, "used": False, "files": [str(path)], "rows": int(len(df)),
                "reason": f"missing auxiliary columns {list(AUXILIARY_METADATA_REQUIRED_COLUMNS)}",
            }
        out = pd.DataFrame({
            "auxiliary_patient_text": df[patient_col].fillna("").astype(str),
            "series_uid": df[series_col].fillna("").astype(str),
            "study_uid": df[study_col].fillna("").astype(str) if study_col else "",
        })
        out["patient_id"] = out.auxiliary_patient_text.map(normalize_patient_id)
        upper = out.auxiliary_patient_text.str.upper()
        out["laterality"] = upper.map(lambda x: "LEFT" if "_LEFT_" in x or x.endswith("_LEFT") else "RIGHT" if "_RIGHT_" in x or x.endswith("_RIGHT") else "")
        out["view"] = upper.map(lambda x: "MLO" if re.search(r"(?:_|-)MLO(?:_|-|$)", x) else "CC" if re.search(r"(?:_|-)CC(?:_|-|$)", x) else "")
        out = out[out.series_uid.str.strip().ne("")].copy()
        return out, {
            "present": True, "used": True, "files": [str(path)], "rows": int(len(out)),
            "sha256": _sha256(path), "reason": "used as auxiliary SeriesInstanceUID identity map; never used as pathology ground truth",
        }

    def _enrich_dicom_index_with_auxiliary(self, dicom_index: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        aux, info = self._load_auxiliary_metadata()
        if dicom_index is None or dicom_index.empty or aux.empty:
            return dicom_index, {**info, "matched_dicom_objects": 0, "filled_patient_id": 0, "filled_laterality": 0, "filled_view": 0}
        # Only unique series identities are safe to use automatically.
        counts = aux.groupby("series_uid").size()
        unique_uids = set(counts[counts == 1].index.astype(str))
        aux_unique = aux[aux.series_uid.astype(str).isin(unique_uids)].drop_duplicates("series_uid").set_index("series_uid")
        df = dicom_index.copy()
        matched = 0; filled_patient = 0; filled_lat = 0; filled_view = 0
        aux_patient_text = []
        aux_match = []
        for idx, row in df.iterrows():
            uid = str(row.get("series_uid", ""))
            if not uid or uid not in aux_unique.index:
                aux_patient_text.append(""); aux_match.append(False); continue
            a = aux_unique.loc[uid]
            matched += 1; aux_match.append(True); aux_patient_text.append(str(a.get("auxiliary_patient_text", "")))
            current_patient = str(row.get("patient_id", "") or "").strip()
            current_lat = str(row.get("laterality", "") or "").strip()
            current_view = str(row.get("view", "") or "").strip()
            if (not current_patient or not re.fullmatch(r"P_\d{5}", normalize_patient_id(current_patient))) and str(a.get("patient_id", "")):
                df.at[idx, "patient_id"] = str(a["patient_id"]); filled_patient += 1
            elif current_patient:
                df.at[idx, "patient_id"] = normalize_patient_id(current_patient)
            if not current_lat and str(a.get("laterality", "")):
                df.at[idx, "laterality"] = str(a["laterality"]); filled_lat += 1
            if not current_view and str(a.get("view", "")):
                df.at[idx, "view"] = str(a["view"]); filled_view += 1
            if not str(row.get("study_uid", "") or "").strip() and str(a.get("study_uid", "")):
                df.at[idx, "study_uid"] = str(a["study_uid"])
        df["auxiliary_metadata_matched"] = aux_match
        df["auxiliary_patient_text"] = aux_patient_text
        return df, {
            **info,
            "matched_dicom_objects": int(matched),
            "filled_patient_id": int(filled_patient),
            "filled_laterality": int(filled_lat),
            "filled_view": int(filled_view),
            "unique_series_uids": int(len(unique_uids)),
        }

    def _raw_has_dicom(self) -> bool:
        p = self._cbis_paths()
        raw = p["raw"]
        if not raw.exists():
            return False
        # A successful prior inspection is sufficient for the normal idempotent path.
        # It avoids recursively walking the large raw tree simply to answer status.
        cache = p["dicom_index"]
        if bool(self.cfg.get("reuse_dicom_index_cache", True)) and cache.exists():
            try:
                cached = pd.read_csv(cache, usecols=["is_dicom"] )
                if not cached.empty and cached.is_dicom.fillna(False).astype(bool).any():
                    return True
            except Exception:
                pass
        for candidate in raw.rglob("*"):
            if candidate.is_file() and (candidate.suffix.lower() in {".dcm", ".dicom"} or not candidate.suffix):
                if candidate.name not in {"DOWNLOAD_INSTRUCTIONS.md"}:
                    return True
        return False

    def _metadata_guidance(self) -> dict:
        p = self._cbis_paths()
        candidates = self._metadata_candidates()
        found = {name: [str(x) for x in paths] for name, paths in candidates.items() if paths}
        missing = [name for name in OFFICIAL_METADATA_FILES if not candidates.get(name)]
        instructions = p["raw"] / "METADATA_INSTRUCTIONS.md"
        p["raw"].mkdir(parents=True, exist_ok=True)
        instructions.write_text(
            "# CBIS-DDSM classification metadata required\n\n"
            "NBIA Data Retriever transfers the DICOM image collection; the four classification CSV tables are separate TCIA supporting data.\n\n"
            "In v0.16, run `python -m dataset_pipeline.download --datasets cbis_ddsm` to acquire/verify these four small CSVs automatically. Existing valid copies are reused. If automatic metadata download is unavailable, place them under this directory (recommended: `metadata/`):\n\n"
            + "\n".join(f"- `{name}`" for name in OFFICIAL_METADATA_FILES)
            + "\n\nAccepted alternate downloaded names are also recognized: `Mass-Training-Description.csv`, `Mass-Test-Description.csv`, `Calc-Training-Description.csv`, and `Calc-Test-Description.csv`.\n"
            "Do not edit pathology values or image paths. The adapter uses the official `pathology` field as ground truth and records SHA-256 hashes during inspection.\n\n"
            f"Official collection: {self.cfg['official_information']}\n",
            encoding="utf-8",
        )
        return {
            "metadata_complete": not missing,
            "official_metadata_found": found,
            "official_metadata_missing": missing,
            "official_metadata_expected": list(OFFICIAL_METADATA_FILES),
            "metadata_instructions": str(instructions),
        }

    def status(self) -> dict:
        p = self._cbis_paths()
        metadata = self._metadata_files(strict=False)
        metadata_complete = len(metadata) == len(OFFICIAL_METADATA_FILES)
        dicom_present = self._raw_has_dicom()
        if p["canonical"].exists():
            state = "AVAILABLE"
        elif p["source"].exists():
            state = "INSPECTED_NOT_PREPARED"
        elif metadata_complete and dicom_present:
            state = "READY_FOR_INSPECT"
        elif dicom_present and not metadata_complete:
            state = "METADATA_REQUIRED"
        elif metadata_complete and not dicom_present:
            state = "DICOM_DOWNLOAD_REQUIRED"
        else:
            state = "NOT_DOWNLOADED"
        candidates = self._metadata_candidates()
        missing = [name for name in OFFICIAL_METADATA_FILES if not candidates.get(name)]
        aux_candidates = self._auxiliary_metadata_candidates()
        return {
            "dataset": self.key,
            "name": self.cfg["name"],
            "status": state,
            "raw_dir": str(p["raw"]),
            "canonical_manifest": str(p["canonical"]),
            "official_metadata_found": sorted(metadata),
            "official_metadata_missing": missing,
            "official_metadata_expected": list(OFFICIAL_METADATA_FILES),
            "dicom_present": dicom_present,
            "dicom_auto_download": False,
            "metadata_auto_download": True,
            "auxiliary_metadata_csv": [str(x) for x in aux_candidates],
            "adapter": "official_tcia_cbis_ddsm_v016",
            "requires_four_views_for_ensemble": True,
        }

    def download(self) -> dict:
        """Prepare CBIS-DDSM acquisition without ever re-downloading the DICOM collection.

        v0.16 automatically acquires/verifies only the four small official classification
        CSVs. DICOM transfer remains an explicit TCIA/NBIA user action. Existing DICOM files
        are detected and reused byte-for-byte.
        """
        p = self._cbis_paths()
        p["raw"].mkdir(parents=True, exist_ok=True)
        if p["canonical"].exists():
            audit("DATASET_REUSED", dataset=self.key, status="AVAILABLE")
            return {**self.status(), "action": "reused", "dicom_download_performed": False}

        instructions = p["raw"] / "DOWNLOAD_INSTRUCTIONS.md"
        instructions.write_text(
            "# CBIS-DDSM — v0.16 acquisition policy\n\n"
            "## DICOM image collection\n\n"
            "The prototype NEVER downloads or re-downloads the ~163 GB DICOM collection automatically. "
            "Use the official TCIA/NBIA Data Retriever and place the complete downloaded directory tree anywhere under this directory. "
            "Existing DICOM files are always reused; do not flatten or rename them.\n\n"
            "## Classification metadata\n\n"
            "The four small official TCIA classification CSV files are downloaded and SHA-256/column validated automatically by `dataset_pipeline.download`. "
            "Valid existing copies are reused and are not downloaded again. They are stored under `metadata/` when acquisition is necessary.\n\n"
            + "\n".join(f"- `{name}`" for name in OFFICIAL_METADATA_FILES)
            + "\n\n`metadata.csv`, when present, is optional auxiliary TCIA series metadata. v0.16 may use PatientID/StudyInstanceUID/SeriesInstanceUID to enrich view resolution, but it is NEVER used as pathology ground truth.\n\n"
            "The adapter generates `source_manifest.csv` only after inspection. Ground truth comes only from the official `pathology` field. Missing standard views are never duplicated or synthesized.\n\n"
            f"Official information: {self.cfg['official_information']}\n",
            encoding="utf-8",
        )

        dicom_present_before = self._raw_has_dicom()
        metadata_result = self._ensure_official_metadata()
        current = self.status()
        if metadata_result["metadata_failed"]:
            state = "METADATA_DOWNLOAD_FAILED"
        elif not self._raw_has_dicom():
            state = "DICOM_DOWNLOAD_REQUIRED"
        else:
            state = "READY_FOR_INSPECT"
        result = {
            **current,
            **metadata_result,
            "status": state,
            "instructions": str(instructions),
            "dicom_download_performed": False,
            "dicom_reused": bool(dicom_present_before),
            "dicom_action": "reused_existing_tree" if dicom_present_before else "manual_tcia_nbia_required",
            "next_action": (
                "Run dataset_pipeline.inspect; DICOM and metadata are present."
                if state == "READY_FOR_INSPECT"
                else "Complete the official TCIA/NBIA DICOM transfer under raw_dir, then rerun status/inspect."
                if state == "DICOM_DOWNLOAD_REQUIRED"
                else "Review metadata_failed; existing DICOM files were not touched."
            ),
        }
        audit(
            "DATASET_DOWNLOAD_PREPARED",
            dataset=self.key,
            status=state,
            dicom_present=self._raw_has_dicom(),
            dicom_download_performed=False,
            metadata_downloaded=metadata_result["metadata_downloaded"],
            metadata_reused=metadata_result["metadata_reused"],
        )
        return result

    def _load_official_metadata(self) -> pd.DataFrame:
        files = self._metadata_files(strict=True)
        frames = []
        for name in OFFICIAL_METADATA_FILES:
            path = files[name]
            df = _canonical_metadata_columns(pd.read_csv(path))
            missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"{path.name} missing required official columns: {missing}")
            lesion_type = "mass" if name.startswith("mass_") else "calcification"
            split = "train" if "_train_" in name else "test"
            df = df.copy()
            df["metadata_file"] = str(path)
            df["lesion_type"] = lesion_type
            df["official_split"] = split
            frames.append(df)
        all_rows = pd.concat(frames, ignore_index=True, sort=False)
        all_rows["patient_id"] = all_rows["patient_id"].map(normalize_patient_id)
        all_rows["laterality"] = all_rows["left or right breast"].map(normalize_laterality)
        all_rows["view"] = all_rows["image view"].map(normalize_view)
        all_rows["ground_truth"] = all_rows["pathology"].map(normalize_pathology)
        all_rows["metadata_row"] = range(len(all_rows))
        return all_rows

    def _all_candidate_files(self) -> list[Path]:
        raw = self._cbis_paths()["raw"]
        if not raw.exists():
            return []
        metadata_names = {n.lower() for n in OFFICIAL_METADATA_FILES}
        ignored = {"download_instructions.md", "source_manifest.csv"}
        files = []
        for p in raw.rglob("*"):
            if not p.is_file():
                continue
            if p.name.lower() in metadata_names or p.name.lower() in ignored:
                continue
            if p.suffix.lower() in {".csv", ".md", ".txt", ".tcia", ".json", ".xml"}:
                continue
            files.append(p.resolve())
        return files

    @staticmethod
    def _path_parts(value: object) -> list[str]:
        text = str(value).replace("\\", "/").strip()
        return [x for x in PurePosixPath(text).parts if x not in {"/", ".", ""}]

    def _build_suffix_index(self, files: Iterable[Path]) -> dict[tuple[str, ...], list[Path]]:
        raw = self._cbis_paths()["raw"].resolve()
        index: dict[tuple[str, ...], list[Path]] = {}
        for path in files:
            try:
                rel_parts = path.relative_to(raw).parts
            except ValueError:
                continue
            lowered = tuple(x.lower() for x in rel_parts)
            for n in (1, 2, 3, 4):
                if len(lowered) >= n:
                    index.setdefault(lowered[-n:], []).append(path)
        return index

    def _resolve_by_suffix(self, official_path: object, suffix_index: dict[tuple[str, ...], list[Path]]) -> ResolvedImage:
        parts = [x.lower() for x in self._path_parts(official_path)]
        for n in (4, 3, 2, 1):
            if len(parts) < n:
                continue
            candidates = sorted(set(suffix_index.get(tuple(parts[-n:]), [])), key=str)
            if len(candidates) == 1:
                return ResolvedImage(candidates[0], f"path_suffix_{n}", 1)
            if len(candidates) > 1 and n >= 2:
                # Continue trying a longer/alternative identity before declaring ambiguity.
                return ResolvedImage(None, f"ambiguous_path_suffix_{n}", len(candidates))
        return ResolvedImage(None, "unresolved", 0)

    @staticmethod
    def _dicom_value(ds, *names: str) -> str:
        for name in names:
            value = getattr(ds, name, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _load_cached_dicom_index(self, enrich_auxiliary: bool = True) -> pd.DataFrame | None:
        """Load a completed DICOM header cache without rescanning the large raw tree.

        v0.16 defaults to cache reuse after a successful inspect. If the DICOM tree has
        changed, the researcher must request `--force-dicom-index`; this makes the cost
        explicit instead of silently walking ~10k files on every inspect under WSL/NTFS.
        """
        if not bool(self.cfg.get("reuse_dicom_index_cache", True)):
            return None
        cache = self._cbis_paths()["dicom_index"]
        if not cache.exists():
            return None
        try:
            cached = pd.read_csv(cache)
        except Exception:
            return None
        required = {"path", "is_dicom", "patient_id", "study_uid", "series_uid", "laterality", "view", "pixels"}
        if not required.issubset(cached.columns) or cached.empty:
            return None
        if enrich_auxiliary:
            cached, aux_info = self._enrich_dicom_index_with_auxiliary(cached)
            if aux_info.get("used"):
                cached.to_csv(cache, index=False)
                audit(
                    "CBIS_DDSM_DICOM_INDEX_CACHE_REUSED", dataset=self.key,
                    rows=int(len(cached)), auxiliary_metadata_used=True,
                    auxiliary_matched_dicom_objects=int(aux_info.get("matched_dicom_objects", 0)),
                )
            else:
                audit("CBIS_DDSM_DICOM_INDEX_CACHE_REUSED", dataset=self.key, rows=int(len(cached)), auxiliary_metadata_used=False)
        return cached

    def _build_dicom_index(self, files: Iterable[Path], force: bool = False) -> pd.DataFrame:
        p = self._cbis_paths()
        cache = p["dicom_index"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        files = list(files)
        # Cache is reused only when path, size and mtime still match. This matters when
        # NBIA is downloading in parallel: a file may exist before its final bytes arrive.
        current_state = {}
        for path in files:
            try:
                st = path.stat()
                current_state[str(path)] = (int(st.st_size), int(st.st_mtime_ns))
            except OSError:
                current_state[str(path)] = (-1, -1)
        if cache.exists() and not force:
            try:
                cached = pd.read_csv(cache)
                required_cache_cols = {"path", "file_size", "file_mtime_ns"}
                if required_cache_cols.issubset(cached.columns) and len(cached) == len(files):
                    cached_state = {
                        str(r.path): (int(r.file_size), int(r.file_mtime_ns))
                        for _, r in cached.iterrows()
                    }
                    if cached_state == current_state:
                        enriched, aux_info = self._enrich_dicom_index_with_auxiliary(cached)
                        # Enrichment is metadata-only and intentionally reuses the v0.14 DICOM
                        # header cache; no DICOM file is opened again when file state is unchanged.
                        if aux_info.get("used"):
                            enriched.to_csv(cache, index=False)
                            audit("CBIS_DDSM_DICOM_INDEX_AUXILIARY_ENRICHED", dataset=self.key, cache_reused=True, **{k:v for k,v in aux_info.items() if k != "files"})
                        return enriched
            except Exception:
                pass
        import pydicom

        rows = []
        tags = [
            "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "ImageLaterality", "Laterality", "ViewPosition", "SeriesDescription",
            "Rows", "Columns", "BitsStored", "PhotometricInterpretation",
        ]
        for file_index, path in enumerate(files, start=1):
            state = current_state.get(str(path), (-1, -1))
            rec = {
                "path": str(path), "file_size": state[0], "file_mtime_ns": state[1],
                "is_dicom": False, "patient_id": "", "study_uid": "",
                "series_uid": "", "sop_uid": "", "laterality": "", "view": "",
                "series_description": "", "rows": 0, "columns": 0, "pixels": 0,
                "bits_stored": "", "photometric": "", "read_error": "",
            }
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=tags)
                rows_v = int(getattr(ds, "Rows", 0) or 0)
                cols_v = int(getattr(ds, "Columns", 0) or 0)
                patient = normalize_patient_id(self._dicom_value(ds, "PatientID") or path.as_posix())
                lat = normalize_laterality(self._dicom_value(ds, "ImageLaterality", "Laterality"))
                view = normalize_view(self._dicom_value(ds, "ViewPosition"))
                if not lat:
                    text = path.as_posix().upper()
                    lat = "LEFT" if "_LEFT_" in text else "RIGHT" if "_RIGHT_" in text else None
                if not view:
                    text = path.as_posix().upper()
                    view = "MLO" if "_MLO" in text else "CC" if "_CC" in text else None
                rec.update({
                    "is_dicom": bool(rows_v and cols_v),
                    "patient_id": patient,
                    "study_uid": self._dicom_value(ds, "StudyInstanceUID"),
                    "series_uid": self._dicom_value(ds, "SeriesInstanceUID"),
                    "sop_uid": self._dicom_value(ds, "SOPInstanceUID"),
                    "laterality": lat or "",
                    "view": view or "",
                    "series_description": self._dicom_value(ds, "SeriesDescription"),
                    "rows": rows_v,
                    "columns": cols_v,
                    "pixels": rows_v * cols_v,
                    "bits_stored": self._dicom_value(ds, "BitsStored"),
                    "photometric": self._dicom_value(ds, "PhotometricInterpretation"),
                })
            except Exception as exc:
                rec["read_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
            if file_index % 500 == 0:
                audit("CBIS_DDSM_DICOM_INDEX_PROGRESS", dataset=self.key, processed=file_index, total=len(files))
        df = pd.DataFrame(rows)
        df, aux_info = self._enrich_dicom_index_with_auxiliary(df)
        df.to_csv(cache, index=False)
        audit(
            "CBIS_DDSM_DICOM_INDEX_BUILT", dataset=self.key, files=len(files),
            dicom=int(df.is_dicom.sum()), cache=str(cache),
            auxiliary_metadata_used=bool(aux_info.get("used")),
            auxiliary_matched_dicom_objects=int(aux_info.get("matched_dicom_objects", 0)),
        )
        return df

    def _resolve_with_dicom_index(
        self,
        official_path: object,
        patient_id: str,
        laterality: str | None,
        view: str | None,
        dicom_index: pd.DataFrame,
    ) -> ResolvedImage:
        parts = self._path_parts(official_path)
        uid_parts = [x for x in parts if re.fullmatch(r"\d+(?:\.\d+)+", x)]
        candidates = pd.DataFrame()
        if uid_parts:
            series_uid = uid_parts[-1]
            candidates = dicom_index[dicom_index.series_uid.astype(str) == series_uid]
            if len(uid_parts) >= 2 and len(candidates) != 1:
                study_uid = uid_parts[-2]
                candidates = dicom_index[
                    (dicom_index.study_uid.astype(str) == study_uid)
                    & (dicom_index.series_uid.astype(str) == series_uid)
                ]
            if len(candidates) == 1:
                return ResolvedImage(Path(candidates.iloc[0].path), "dicom_uid", 1)

        # Deterministic fallback for layouts that discard the metadata path hierarchy.
        # Full mammograms are much larger than ROI/cropped images. We do not select a
        # candidate below the configured minimum pixel count.
        subset = dicom_index[
            (dicom_index.is_dicom == True)  # noqa: E712
            & (dicom_index.patient_id.astype(str) == patient_id)
        ]
        if laterality:
            subset = subset[subset.laterality.astype(str) == laterality]
        if view:
            subset = subset[subset.view.astype(str) == view]
        min_pixels = int(self.cfg.get("minimum_full_image_pixels", 1_000_000))
        subset = subset[pd.to_numeric(subset.pixels, errors="coerce").fillna(0).astype(int) >= min_pixels]
        if subset.empty:
            return ResolvedImage(None, "unresolved", 0)
        subset = subset.sort_values(["pixels", "path"], ascending=[False, True])
        top_pixels = int(subset.iloc[0].pixels)
        top = subset[pd.to_numeric(subset.pixels, errors="coerce").fillna(0).astype(int) == top_pixels]
        if len(top) == 1:
            return ResolvedImage(Path(top.iloc[0].path), "dicom_patient_view_largest", len(subset))
        return ResolvedImage(None, "ambiguous_dicom_patient_view", len(top))

    def _resolve_metadata_images(self, metadata: pd.DataFrame, force_dicom_index: bool = False):
        cached_index = None if force_dicom_index else self._load_cached_dicom_index(enrich_auxiliary=True)
        # When a prior successful inspection exists, use its recorded paths as the
        # suffix/UID index. This performs no raw-tree traversal and opens no DICOM file.
        files = [Path(x) for x in cached_index.path.astype(str)] if cached_index is not None else self._all_candidate_files()
        suffix_index = self._build_suffix_index(files)
        records = []
        unresolved_rows = []
        pending = []
        for _, r in metadata.iterrows():
            rec = r.to_dict()
            invalid = []
            if not rec.get("laterality"):
                invalid.append("invalid_laterality")
            if not rec.get("view"):
                invalid.append("invalid_view")
            if pd.isna(rec.get("ground_truth")):
                invalid.append("unsupported_pathology")
            if invalid:
                rec.update({"resolved_image": "", "resolution_method": "metadata_invalid", "resolution_candidates": 0, "reject_reason": ";".join(invalid)})
                unresolved_rows.append(rec)
                records.append(rec)
                continue
            resolved = self._resolve_by_suffix(rec["image file path"], suffix_index)
            rec.update({
                "resolved_image": str(resolved.path) if resolved.path else "",
                "resolution_method": resolved.method,
                "resolution_candidates": resolved.candidates,
                "reject_reason": "" if resolved.path else "unresolved_image_path",
            })
            records.append(rec)
            if not resolved.path:
                pending.append(len(records) - 1)

        dicom_index = cached_index
        if pending:
            if dicom_index is None:
                dicom_index = self._build_dicom_index(files, force=force_dicom_index)
            for idx in pending:
                rec = records[idx]
                resolved = self._resolve_with_dicom_index(
                    rec["image file path"], rec["patient_id"], rec.get("laterality"), rec.get("view"), dicom_index
                )
                rec["resolved_image"] = str(resolved.path) if resolved.path else ""
                rec["resolution_method"] = resolved.method
                rec["resolution_candidates"] = resolved.candidates
                rec["reject_reason"] = "" if resolved.path else "unresolved_image_path"

        resolved_df = pd.DataFrame(records)
        unresolved_df = resolved_df[resolved_df.resolved_image.astype(str).eq("")].copy()
        return resolved_df, unresolved_df, dicom_index

    @staticmethod
    def _series_uid_from_metadata_path(value: object) -> str:
        parts = [x for x in str(value).replace("\\", "/").split("/") if x]
        uids = [x for x in parts if re.fullmatch(r"\d+(?:\.\d+)+", x)]
        return uids[-1] if uids else ""

    def _supplement_unreferenced_views(self, resolved: pd.DataFrame, dicom_index: pd.DataFrame | None) -> pd.DataFrame:
        """Add unreferenced full mammograms only when DICOM metadata identifies patient/laterality/view.

        These rows carry no new pathology label; patient/side labels are still derived solely
        from official CSV pathology. This can recover contralateral standard views if TCIA's
        downloaded image tree contains them even though a lesion CSV does not reference them.
        """
        if dicom_index is None or dicom_index.empty:
            return resolved
        known_paths = set(resolved.loc[resolved.resolved_image.astype(str).ne(""), "resolved_image"].astype(str))
        patient_ids = set(resolved.patient_id.astype(str))
        min_pixels = int(self.cfg.get("minimum_full_image_pixels", 1_000_000))
        auxiliary_series = set()
        for col in ("cropped image file path", "roi mask file path"):
            if col in resolved.columns:
                auxiliary_series.update(
                    uid for uid in resolved[col].dropna().map(self._series_uid_from_metadata_path) if uid
                )
        path_text = dicom_index.path.astype(str).str.upper()
        desc_text = dicom_index.series_description.fillna("").astype(str).str.upper() if "series_description" in dicom_index else pd.Series("", index=dicom_index.index)
        looks_auxiliary = (
            path_text.str.contains("ROI", regex=False)
            | path_text.str.contains("CROP", regex=False)
            | path_text.str.contains("MASK", regex=False)
            | desc_text.str.contains("ROI", regex=False)
            | desc_text.str.contains("CROP", regex=False)
            | desc_text.str.contains("MASK", regex=False)
            | dicom_index.series_uid.astype(str).isin(auxiliary_series)
        )
        extra = dicom_index[
            (dicom_index.is_dicom == True)  # noqa: E712
            & (dicom_index.patient_id.astype(str).isin(patient_ids))
            & (dicom_index.laterality.astype(str).isin(["LEFT", "RIGHT"]))
            & (dicom_index.view.astype(str).isin(["CC", "MLO"]))
            & (pd.to_numeric(dicom_index.pixels, errors="coerce").fillna(0).astype(int) >= min_pixels)
            & (~dicom_index.path.astype(str).isin(known_paths))
            & (~looks_auxiliary)
        ].copy()
        if extra.empty:
            return resolved
        rows = []
        for _, x in extra.iterrows():
            rows.append({
                "patient_id": str(x.patient_id),
                "left or right breast": str(x.laterality),
                "image view": str(x.view),
                "pathology": "",
                "image file path": "",
                "metadata_file": "DICOM_HEADER_SUPPLEMENT",
                "lesion_type": "unreferenced_full_mammogram",
                "official_split": "unknown",
                "laterality": str(x.laterality),
                "view": str(x.view),
                "ground_truth": pd.NA,
                "metadata_row": pd.NA,
                "resolved_image": str(x.path),
                "resolution_method": "dicom_header_supplement",
                "resolution_candidates": 1,
                "reject_reason": "",
            })
        return pd.concat([resolved, pd.DataFrame(rows)], ignore_index=True, sort=False)

    @staticmethod
    def _select_view_path(group: pd.DataFrame) -> tuple[str | None, int]:
        paths = sorted(set(group.loc[group.resolved_image.astype(str).ne(""), "resolved_image"].astype(str)))
        if not paths:
            return None, 0
        return paths[0], len(paths)

    def _build_study_catalog(self, resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Ground truth comes only from official pathology rows (supplemental DICOM rows have NA).
        official = resolved[pd.to_numeric(resolved.ground_truth, errors="coerce").notna()].copy()
        if official.empty:
            return pd.DataFrame(), pd.DataFrame()

        study_rows = []
        for patient_id, patient_rows in resolved.groupby("patient_id", sort=True):
            official_patient = official[official.patient_id.astype(str) == str(patient_id)]
            if official_patient.empty:
                continue
            gt = int(pd.to_numeric(official_patient.ground_truth, errors="coerce").max())
            side_gt = {}
            for side in ("LEFT", "RIGHT"):
                sr = official_patient[official_patient.laterality.astype(str) == side]
                side_gt[side] = int(pd.to_numeric(sr.ground_truth, errors="coerce").max()) if not sr.empty else pd.NA
            splits = sorted(set(official_patient.official_split.dropna().astype(str)))
            split = splits[0] if len(splits) == 1 else "mixed"
            row = {
                "study_id": f"CBIS-DDSM_{patient_id}",
                "patient_id": str(patient_id),
                "ground_truth": gt,
                "left_ground_truth": side_gt["LEFT"],
                "right_ground_truth": side_gt["RIGHT"],
                "horizontal_flip": "NO",
                "official_split": split,
                "official_splits": "|".join(splits),
            }
            missing = []
            duplicate_views = []
            for (side, view), col in VIEW_COLUMNS.items():
                vg = patient_rows[
                    (patient_rows.laterality.astype(str) == side)
                    & (patient_rows.view.astype(str) == view)
                ]
                chosen, count = self._select_view_path(vg)
                row[col] = chosen or ""
                row[f"{col}_candidate_count"] = count
                if not chosen:
                    missing.append(f"{side}_{view}")
                elif count > 1:
                    duplicate_views.append(f"{side}_{view}:{count}")
            row["complete_four_view"] = not missing
            row["missing_views"] = "|".join(missing)
            row["duplicate_view_candidates"] = "|".join(duplicate_views)
            row["metadata_rows"] = int(len(official_patient))
            study_rows.append(row)

        catalog = pd.DataFrame(study_rows)
        complete = catalog[catalog.complete_four_view == True].copy() if not catalog.empty else catalog.copy()  # noqa: E712
        return catalog, complete

    def _write_adapter_artifacts(self, resolved: pd.DataFrame, unresolved: pd.DataFrame, catalog: pd.DataFrame, complete: pd.DataFrame):
        p = self._cbis_paths()
        for key in ("metadata_catalog", "view_catalog", "incomplete", "unresolved", "source"):
            p[key].parent.mkdir(parents=True, exist_ok=True)
        resolved.to_csv(p["metadata_catalog"], index=False)
        catalog.to_csv(p["view_catalog"], index=False)
        incomplete = catalog[catalog.complete_four_view != True].copy() if not catalog.empty else catalog.copy()  # noqa: E712
        incomplete.to_csv(p["incomplete"], index=False)
        unresolved.to_csv(p["unresolved"], index=False)
        source_cols = [*REQUIRED, "left_ground_truth", "right_ground_truth", "horizontal_flip", "official_split"]
        for c in source_cols:
            if c not in complete.columns:
                complete[c] = pd.Series(dtype="object")
        complete[source_cols].to_csv(p["source"], index=False)

    def _dicom_object_inventory(self, metadata: pd.DataFrame, dicom_index: pd.DataFrame) -> dict:
        """Classify downloaded DICOM objects by the official path tables, without pixel decoding."""
        files = [Path(x) for x in dicom_index.path.astype(str)] if dicom_index is not None and not dicom_index.empty else []
        suffix_index = self._build_suffix_index(files)
        categories = {
            "full_mammogram_images": ("image file path", set()),
            "cropped_images": ("cropped image file path", set()),
            "roi_masks": ("ROI mask file path", set()),
        }
        unresolved = {}
        for label, (column, paths) in categories.items():
            missing = 0
            if column not in metadata.columns:
                unresolved[label] = 0
                continue
            values = sorted(set(str(x).strip() for x in metadata[column].dropna() if str(x).strip()))
            for value in values:
                resolved = self._resolve_by_suffix(value, suffix_index)
                if resolved.path:
                    paths.add(str(resolved.path))
                else:
                    # Deterministic UID fallback when hierarchy/suffix changed.
                    uid = self._series_uid_from_metadata_path(value)
                    subset = dicom_index[dicom_index.series_uid.astype(str) == uid] if uid else pd.DataFrame()
                    if len(subset) == 1:
                        paths.add(str(subset.iloc[0].path))
                    else:
                        missing += 1
            unresolved[label] = missing
        valid_dicom_paths = set(
            dicom_index.loc[dicom_index.is_dicom == True, "path"].astype(str)  # noqa: E712
        ) if not dicom_index.empty else set()
        full = categories["full_mammogram_images"][1] & valid_dicom_paths
        cropped = categories["cropped_images"][1] & valid_dicom_paths
        roi = categories["roi_masks"][1] & valid_dicom_paths
        referenced = full | cropped | roi
        return {
            "dicom_objects": int(len(valid_dicom_paths)),
            "full_mammogram_images": int(len(full)),
            "cropped_images": int(len(cropped)),
            "roi_masks": int(len(roi)),
            "other_dicom_images": int(len(valid_dicom_paths - referenced)),
            "object_inventory_unresolved_paths": unresolved,
        }

    @staticmethod
    def _selected_full_view_count(complete: pd.DataFrame) -> int:
        if complete is None or complete.empty:
            return 0
        paths = set()
        for col in ("l_cc", "r_cc", "l_mlo", "r_mlo"):
            if col in complete.columns:
                paths.update(str(x) for x in complete[col].dropna() if str(x).strip())
        return len(paths)

    def inspect(self, force_dicom_index: bool = False) -> dict:
        p = self._cbis_paths()
        guidance = self._metadata_guidance()
        if not guidance["metadata_complete"]:
            result = {
                "dataset": self.key,
                "status": "METADATA_REQUIRED",
                "raw_dir": str(p["raw"]),
                "dicom_present": self._raw_has_dicom(),
                **guidance,
                "next_action": "Download the four official CBIS-DDSM classification CSV tables from TCIA Data Access and place them anywhere under raw_dir, then rerun inspect.",
                "dicom_index_started": False,
            }
            audit("CBIS_DDSM_METADATA_REQUIRED", dataset=self.key, missing=guidance["official_metadata_missing"], instructions=guidance["metadata_instructions"] )
            return result
        if not self._raw_has_dicom():
            result = {
                "dataset": self.key,
                "status": "DICOM_DOWNLOAD_REQUIRED",
                "raw_dir": str(p["raw"]),
                **guidance,
                "next_action": "Download the official CBIS-DDSM DICOM collection with TCIA/NBIA Data Retriever under raw_dir. v0.16 will reuse it and never auto-download DICOM bytes.",
                "dicom_index_started": False,
            }
            audit("CBIS_DDSM_DICOM_REQUIRED", dataset=self.key, raw_dir=str(p["raw"]))
            return result
        metadata_files = self._metadata_files(strict=True)
        metadata_hashes = {name: _sha256(path) for name, path in metadata_files.items()}
        metadata = self._load_official_metadata()
        resolved, unresolved, dicom_index = self._resolve_metadata_images(metadata, force_dicom_index=force_dicom_index)
        # Build the DICOM index even when all official paths resolved so we can detect
        # unreferenced standard views (e.g. contralateral images) without pixel decoding.
        if dicom_index is None:
            dicom_index = self._build_dicom_index(self._all_candidate_files(), force=force_dicom_index)
        resolved = self._supplement_unreferenced_views(resolved, dicom_index)
        supplemental_rows = int((resolved.resolution_method.astype(str) == "dicom_header_supplement").sum())
        catalog, complete = self._build_study_catalog(resolved)
        self._write_adapter_artifacts(resolved, unresolved, catalog, complete)
        pathology_counts = (
            metadata.assign(pathology_normalized=metadata.pathology.astype(str).str.upper())
            .pathology_normalized.value_counts(dropna=False).to_dict()
        )
        inventory = self._dicom_object_inventory(metadata, dicom_index)
        aux_df, aux_info = self._load_auxiliary_metadata()
        if dicom_index is not None and "auxiliary_metadata_matched" in dicom_index.columns:
            aux_info = {**aux_info, "matched_dicom_objects": int(dicom_index.auxiliary_metadata_matched.fillna(False).astype(bool).sum())}
        complete_gt_counts = (
            pd.to_numeric(complete.ground_truth, errors="coerce").dropna().astype(int).value_counts().to_dict()
            if not complete.empty and "ground_truth" in complete.columns else {}
        )
        result = {
            "dataset": self.key,
            "status": "INSPECTED",
            "raw_dir": str(p["raw"]),
            "metadata_rows": int(len(metadata)),
            "resolved_metadata_rows": int((resolved.resolution_method != "dicom_header_supplement").sum() - len(unresolved)),
            "unresolved_metadata_rows": int(len(unresolved)),
            "dicom_files_indexed": int(len(dicom_index)) if dicom_index is not None else 0,
            "dicom_headers_valid": int(dicom_index.is_dicom.sum()) if dicom_index is not None and "is_dicom" in dicom_index else 0,
            **inventory,
            "auxiliary_metadata": aux_info,
            "supplemental_standard_views": supplemental_rows,
            "patients": int(catalog.patient_id.nunique()) if not catalog.empty else 0,
            "complete_four_view_studies": int(len(complete)),
            "incomplete_studies": int(len(catalog) - len(complete)),
            "selected_full_view_images": self._selected_full_view_count(complete),
            "complete_study_ground_truth_counts": {
                "BENIGN": int(complete_gt_counts.get(0, 0)),
                "MALIGNANT": int(complete_gt_counts.get(1, 0)),
            },
            "pathology_counts": pathology_counts,
            "official_metadata_sha256": metadata_hashes,
            "metadata_catalog": str(p["metadata_catalog"]),
            "view_catalog": str(p["view_catalog"]),
            "source_manifest": str(p["source"]),
            "incomplete_manifest": str(p["incomplete"]),
            "unresolved_manifest": str(p["unresolved"]),
            "dicom_index": str(p["dicom_index"]),
            "ensemble_compatible": bool(len(complete)),
            "note": "Current NYU/DMV-CNN exam-level path requires all four standard views. v0.16 may recover standard views from optional TCIA metadata.csv SeriesInstanceUID identity, but missing views are never synthesized and pathology still comes only from the four official case-description CSVs.",
        }
        audit("CBIS_DDSM_INSPECTED", **{k: v for k, v in result.items() if k not in {"pathology_counts", "note"}})
        return result

    def verify_integrity(self) -> dict:
        try:
            result = self.inspect(force_dicom_index=False)
        except Exception as exc:
            return {"dataset": self.key, "valid": False, "reason": f"{type(exc).__name__}: {exc}"}
        if result.get("status") != "INSPECTED":
            return {"dataset": self.key, "valid": False, "reason": result.get("status", "inspection incomplete")}
        return {
            "dataset": self.key,
            "valid": result["unresolved_metadata_rows"] == 0 and result["complete_four_view_studies"] > 0,
            "metadata_rows": result["metadata_rows"],
            "unresolved_metadata_rows": result["unresolved_metadata_rows"],
            "complete_four_view_studies": result["complete_four_view_studies"],
            "reason": "ok" if result["complete_four_view_studies"] > 0 else "no complete four-view studies for current ensemble",
        }

    def prepare(self) -> dict:
        p = self._cbis_paths()
        inspection = self.inspect(force_dicom_index=False)
        if inspection.get("status") in {"METADATA_REQUIRED", "DICOM_DOWNLOAD_REQUIRED"}:
            return {**inspection, "converted_studies": 0}
        src = pd.read_csv(p["source"])
        if src.empty:
            if p["canonical"].exists():
                p["canonical"].unlink()
            audit(
                "CBIS_DDSM_PREPARED_NO_FOUR_VIEW_STUDIES",
                dataset=self.key,
                patients=inspection["patients"],
                incomplete_studies=inspection["incomplete_studies"],
            )
            return {
                **inspection,
                "status": "INSUFFICIENT_FOUR_VIEW_STUDIES",
                "canonical_manifest": str(p["canonical"]),
                "converted_studies": 0,
            }

        missing = [c for c in REQUIRED if c not in src.columns]
        if missing:
            raise ValueError(f"generated source_manifest.csv missing {missing}")
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
            }
            for col, view in [("l_cc", "L_CC"), ("r_cc", "R_CC"), ("l_mlo", "L_MLO"), ("r_mlo", "R_MLO")]:
                source = safe_workspace_path(str(r[col]))
                dest = image_dir / f"{row['study_id']}_{view}.png"
                self._convert_to_png(source, dest)
                row[col] = str(dest)
            for col in ["left_ground_truth", "right_ground_truth", "horizontal_flip", "official_split"]:
                if col in src.columns and pd.notna(r.get(col)):
                    row[col] = r.get(col)
            row.setdefault("horizontal_flip", "NO")
            out.append(row)
        df = pd.DataFrame(out)
        df.to_csv(p["canonical"], index=False)
        audit(
            "DATASET_PREPARED",
            dataset=self.key,
            studies=len(df),
            manifest=str(p["canonical"]),
            adapter="official_tcia_cbis_ddsm",
        )
        return {
            **inspection,
            "status": "AVAILABLE",
            "studies": len(df),
            "converted_studies": len(df),
            "manifest": str(p["canonical"]),
        }
