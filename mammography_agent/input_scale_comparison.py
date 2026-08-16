from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import struct
import zlib

from .config import WORKSPACE_ROOT
from .metarepo_format import build_batch
from .model_client import ensure_metarepository, preprocess_model

VIEW_COLUMNS = {
    "l_cc": "L-CC",
    "r_cc": "R-CC",
    "l_mlo": "L-MLO",
    "r_mlo": "R-MLO",
}


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _iter_grayscale_png_rows(path: Path):
    """Yield 16-bit grayscale PNG rows without external image libraries.

    The thesis input contract is non-interlaced, single-plane 16-bit grayscale PNG.
    Supporting the standard PNG filters here keeps this diagnostic independent from
    model/runtime image libraries and from optional local validation dependencies.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    width = height = bitdepth = color_type = interlace = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        kind = data[pos+4:pos+8]
        payload = data[pos+8:pos+8+length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bitdepth, color_type, _comp, _filter, interlace = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if None in (width, height, bitdepth, color_type, interlace):
        raise ValueError(f"PNG missing IHDR: {path}")
    if color_type != 0 or bitdepth != 16 or interlace != 0:
        raise ValueError(
            f"Expected non-interlaced 16-bit grayscale PNG: {path}; "
            f"bitdepth={bitdepth} color_type={color_type} interlace={interlace}"
        )
    raw = zlib.decompress(bytes(idat))
    bpp = 2
    row_bytes = int(width) * bpp
    expected = int(height) * (1 + row_bytes)
    if len(raw) != expected:
        raise ValueError(f"Unexpected decompressed PNG size for {path}: {len(raw)} != {expected}")
    prev = bytearray(row_bytes)
    offset = 0
    for _ in range(int(height)):
        filter_type = raw[offset]
        scan = raw[offset+1:offset+1+row_bytes]
        offset += 1 + row_bytes
        recon = bytearray(row_bytes)
        for i, x in enumerate(scan):
            left = recon[i-bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i-bpp] if i >= bpp else 0
            if filter_type == 0:
                value = x
            elif filter_type == 1:
                value = (x + left) & 0xFF
            elif filter_type == 2:
                value = (x + up) & 0xFF
            elif filter_type == 3:
                value = (x + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                value = (x + _paeth(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type} in {path}")
            recon[i] = value
        prev = recon
        yield int(width), int(height), np.frombuffer(bytes(recon), dtype=">u2").astype(np.float64)


def _pixel_stats(path: Path, *, source: str, stage: str, image_id: str, max_quantile_samples: int = 200_000) -> dict:
    total_expected = None
    total = 0
    zero_count = 0
    sum_ = 0.0
    sumsq = 0.0
    min_v = None
    max_v = None
    width = height = None
    samples: list[np.ndarray] = []
    stride = 1
    for row_idx, (w, h, arr) in enumerate(_iter_grayscale_png_rows(path)):
        if width is None:
            width, height = w, h
            total_expected = int(width) * int(height)
            stride = max(1, int(math.ceil(total_expected / max_quantile_samples)))
        total += int(arr.size)
        zero_count += int(np.count_nonzero(arr == 0))
        sum_ += float(arr.sum(dtype=np.float64))
        sumsq += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))
        row_min = float(arr.min()); row_max = float(arr.max())
        min_v = row_min if min_v is None else min(min_v, row_min)
        max_v = row_max if max_v is None else max(max_v, row_max)
        # Offset sampling phase by row to avoid repeatedly sampling only the same columns.
        phase = row_idx % stride
        samples.append(arr[phase::stride].copy())
    if total == 0 or width is None or height is None:
        raise ValueError(f"Empty PNG: {path}")
    sample = np.concatenate(samples) if samples else np.array([], dtype=np.float64)
    mean = sum_ / total
    variance = max(0.0, (sumsq / total) - mean * mean)
    std = math.sqrt(variance)
    nominal_max = 65535.0
    q = np.quantile(sample, [0.01, 0.05, 0.50, 0.95, 0.99]) if sample.size else [None] * 5
    return {
        "source": source,
        "stage": stage,
        "image_id": image_id,
        "path": str(path),
        "width": int(width),
        "height": int(height),
        "bitdepth": 16,
        "pixel_count": int(total),
        "sampled_pixels_for_quantiles": int(sample.size),
        "min": float(min_v),
        "max": float(max_v),
        "mean": float(mean),
        "std": float(std),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "median": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "zero_fraction": float(zero_count / total),
        "nonzero_fraction": float(1.0 - zero_count / total),
        "nominal_max": nominal_max,
        "dynamic_range_fraction": float(max_v / nominal_max),
        "normalized_mean": float(mean / nominal_max),
        "normalized_std": float(std / nominal_max),
        "normalized_q99": float(q[4] / nominal_max),
    }


def _stats_for_paths(paths: list[tuple[str, Path]], *, source: str, stage: str) -> list[dict]:
    rows = []
    for image_id, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(_pixel_stats(path, source=source, stage=stage, image_id=image_id))
    return rows


def _cbis_paths(selected: pd.DataFrame) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for _, r in selected.iterrows():
        sid = str(r["study_id"])
        for col, view in VIEW_COLUMNS.items():
            rows.append((f"{sid}:{view}", Path(str(r[col]))))
    return rows


def _all_pngs(directory: Path) -> list[tuple[str, Path]]:
    return [(p.stem, p) for p in sorted(directory.rglob("*.png"))]


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dynamic_range_fraction", "normalized_mean", "normalized_std", "normalized_q99", "zero_fraction"]
    rows = []
    for (source, stage), group in df.groupby(["source", "stage"], sort=True):
        rec = {"source": source, "stage": stage, "images": int(len(group))}
        for col in cols:
            rec[f"median_{col}"] = float(group[col].median())
            rec[f"mean_{col}"] = float(group[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_input_scale(run_dir: str | Path, output_dir: str | Path | None = None, include_nyu_crop: bool = True) -> Path:
    """Compare CBIS-DDSM input intensity scale against the official upstream sample.

    This is a classifier-free diagnostic. Raw PNG statistics are always computed.
    When include_nyu_crop=True, the same official NYU crop+optimal-center preprocessing
    is run on both sources and pixel statistics are also compared after cropping.
    These are image-pixel statistics, not exact classifier tensors; if upstream runtime
    reproduction later fails, exact model tensor instrumentation is the next escalation.
    """
    run_dir = Path(run_dir).resolve()
    selected_path = run_dir / "selected_studies.csv"
    if not selected_path.is_file():
        raise FileNotFoundError(f"selected_studies.csv not found: {selected_path}")
    selected = pd.read_csv(selected_path)
    required = {"study_id", *VIEW_COLUMNS.keys()}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"selected_studies.csv missing required columns: {missing}")

    out = Path(output_dir).resolve() if output_dir else WORKSPACE_ROOT / "output" / "analyses" / f"input-scale-{_timestamp()}"
    out.mkdir(parents=True, exist_ok=True)

    meta = ensure_metarepository()
    runtime_meta = WORKSPACE_ROOT / "runtime" / "mammography_metarepository"
    official_images = runtime_meta / "sample_data" / "images"
    official_data = runtime_meta / "sample_data" / "data.pkl"
    if not official_images.is_dir() or not official_data.is_file():
        raise FileNotFoundError(f"Official sample data missing under {runtime_meta / 'sample_data'}")

    rows: list[dict] = []
    rows += _stats_for_paths(_cbis_paths(selected), source="cbis_ddsm", stage="raw_prepared_png")
    rows += _stats_for_paths(_all_pngs(official_images), source="official_sample", stage="raw_prepared_png")

    preprocess_meta: dict[str, object] = {"performed": False}
    if include_nyu_crop:
        work = out / "preprocess_work"
        cbis_batch = work / "cbis_batch"
        cbis_images, cbis_data = build_batch(selected, cbis_batch)
        cbis_pre = work / "cbis_nyu"
        official_pre = work / "official_nyu"
        cbis_result = preprocess_model("nyu", f"input-scale-cbis-{out.name}", str(cbis_images), str(cbis_data), str(cbis_pre))
        official_result = preprocess_model("nyu", f"input-scale-official-{out.name}", str(official_images), str(official_data), str(official_pre))
        cbis_cropped = Path(str(cbis_result["cropped_images"]))
        official_cropped = Path(str(official_result["cropped_images"]))
        rows += _stats_for_paths(_all_pngs(cbis_cropped), source="cbis_ddsm", stage="nyu_upstream_cropped")
        rows += _stats_for_paths(_all_pngs(official_cropped), source="official_sample", stage="nyu_upstream_cropped")
        preprocess_meta = {
            "performed": True,
            "classifier_inference_performed": False,
            "model": "nyu",
            "cbis_preprocess": cbis_result,
            "official_preprocess": official_result,
        }

    df = pd.DataFrame(rows)
    df.to_csv(out / "input_scale_image_stats.csv", index=False)
    groups = _group_summary(df)
    groups.to_csv(out / "input_scale_group_summary.csv", index=False)

    comparisons = []
    for stage in sorted(df["stage"].unique()):
        g = groups[groups.stage == stage].set_index("source")
        if not {"cbis_ddsm", "official_sample"}.issubset(g.index):
            continue
        for metric in ["median_dynamic_range_fraction", "median_normalized_mean", "median_normalized_std", "median_normalized_q99", "median_zero_fraction"]:
            cbis = float(g.loc["cbis_ddsm", metric])
            official = float(g.loc["official_sample", metric])
            comparisons.append({
                "stage": stage,
                "metric": metric,
                "cbis_ddsm": cbis,
                "official_sample": official,
                "ratio_cbis_to_official": (cbis / official) if official != 0 else None,
                "absolute_difference": cbis - official,
            })
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(out / "input_scale_comparison.csv", index=False)

    summary = {
        "source_run": str(run_dir),
        "cbis_studies": int(len(selected)),
        "cbis_raw_images": int(len(_cbis_paths(selected))),
        "official_sample_images": int(len(_all_pngs(official_images))),
        "metarepository": meta,
        "nyu_crop_comparison": preprocess_meta,
        "interpretation": {
            "purpose": "Detect gross shared input-scale/intensity mismatches before GPU reference inference.",
            "not_proof": "A difference in pixel statistics is a review signal, not proof of incorrect preprocessing or domain shift.",
            "tensor_scope": "Statistics are measured on PNG pixels before classifier tensorization and, when enabled, after official NYU crop preprocessing. Exact model tensors are not instrumented in this version.",
        },
        "research_guards": {
            "diagnostic_only": True,
            "eligible_for_freeze": False,
            "ground_truth_used": False,
            "model_scores_used": False,
            "classifier_inference_performed": False,
            "images_modified": False,
            "dataset_modified": False,
        },
    }
    _json(out / "input_scale_summary.json", summary)

    lines = [
        "# Input Scale Comparison",
        "",
        "> CPU/classifier-free diagnostic comparing CBIS-DDSM prepared inputs with the official NYU metarepository sample.",
        "",
        f"- **source_run**: {run_dir}",
        f"- **CBIS studies/images**: {len(selected)} / {len(_cbis_paths(selected))}",
        f"- **official sample images**: {len(_all_pngs(official_images))}",
        f"- **NYU crop comparison performed**: {bool(include_nyu_crop)}",
        "- **ground truth used**: False",
        "- **classifier inference performed**: False",
        "",
        "## Group summaries",
        "",
        "| source | stage | images | median normalized mean | median normalized std | median normalized q99 | median dynamic range | median zero fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in groups.iterrows():
        lines.append(
            f"| {r.source} | {r.stage} | {int(r.images)} | {r.median_normalized_mean:.6f} | {r.median_normalized_std:.6f} | "
            f"{r.median_normalized_q99:.6f} | {r.median_dynamic_range_fraction:.6f} | {r.median_zero_fraction:.6f} |"
        )
    lines += [
        "",
        "## Interpretation guard",
        "",
        "- This check is designed to catch gross shared scale/intensity differences before spending GPU time on reference inference.",
        "- Similar statistics do not prove the inputs are semantically equivalent; different acquisition domains can legitimately differ.",
        "- Different statistics do not automatically prove a bug; they identify where to inspect normalization/windowing next.",
        "- The post-crop stage uses upstream NYU crop/optimal-center preprocessing but does not instrument the exact classifier tensor. If upstream reference reproduction fails, exact tensor-boundary instrumentation is the next diagnostic.",
    ]
    (out / "input_scale_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
