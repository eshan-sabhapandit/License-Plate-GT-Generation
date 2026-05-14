from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Hashable

from fast_plate_ocr import LicensePlateRecognizer

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def _is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES


def list_plate_images(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return sorted image paths under ``folder`` (non-hidden files only)."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if recursive:
        paths = (p for p in root.rglob("*") if _is_image_path(p))
    else:
        paths = (p for p in root.iterdir() if _is_image_path(p))

    return sorted(paths, key=lambda p: p.as_posix().lower())


def _stem_to_key_and_variant(stem: str) -> tuple[str, int] | None:
    """Parse ``{detection_key}_{variant}`` with variant the last ``_`` segment (digits only)."""
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    if not prefix or not suffix.isdigit():
        return None
    return prefix, int(suffix)


def _group_license_plate_images_by_detection(
    folder: str | Path,
    *,
    recursive: bool = False,
) -> dict[str, list[Path]]:
    """Group crops named ``<key>_<variant>.jpg`` by string key (prefix before last ``_``)."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if recursive:
        candidates = (p for p in root.rglob("*") if _is_image_path(p))
    else:
        candidates = (p for p in root.iterdir() if _is_image_path(p))

    groups: dict[str, list[Path]] = {}
    for p in candidates:
        parsed = _stem_to_key_and_variant(p.stem)
        if parsed is None:
            continue
        key_str, _variant = parsed
        groups.setdefault(key_str, []).append(p)

    for key_str in groups:
        groups[key_str].sort(
            key=lambda path: (
                _stem_to_key_and_variant(path.stem)[1],
                path.name.lower(),
            ),
        )
    return groups


def _six_char_plate_mode_first_on_tie(strings: list[str]) -> str:
    """Pick the most frequent 6-char (stripped) OCR string; ties → first in crop order.

    ``strings`` order is variant / reading order. Among strings tied for the highest
    count, returns the one whose **first qualifying occurrence** appears earliest in
    that list.
    """
    stripped = [s.strip() for s in strings]
    six_only = [t for t in stripped if len(t) == 6]
    if not six_only:
        return ""
    counts = Counter(six_only)
    best = max(counts.values())
    for s in strings:
        t = s.strip()
        if len(t) == 6 and counts[t] == best:
            return t
    return ""


def infer_license_plates_using_OCR(
    folder: str | Path,
    exit_seconds_by_detection: dict[Hashable, float],
    model_variant: str,
    *,
    recursive: bool = False,
    return_confidence: bool = False,
) -> dict[Hashable, str]:
    """Run OCR on plate crops and pick one plate string per detection key.

    For each key, among OCR strings with **exactly 6 characters** (after strip), the
    **most common** wins; if several strings tie for that count, the **first** such
    string in variant order (first crop whose reading is a tied winner) is returned.

    Expects crops under ``folder`` named ``<key>_<variant>.jpg`` where ``<key>`` matches
    ``str(k)`` for each ``k`` in ``exit_seconds_by_detection`` (e.g. ``0_00.jpg`` or
    ``02:29_00.jpg``). Variant is a non-negative integer suffix after the last ``_``.

    Returns a dict with the same keys and key order as ``exit_seconds_by_detection``;
    missing image groups map to ``""``.
    """
    groups = _group_license_plate_images_by_detection(folder, recursive=recursive)

    ordered_keys = list(exit_seconds_by_detection.keys())
    segments: list[tuple[Hashable, list[Path]]] = [
        (k, groups.get(str(k), [])) for k in ordered_keys
    ]
    all_paths = [p for _, paths in segments for p in paths]

    if not all_paths:
        return {k: "" for k in ordered_keys}

    model = LicensePlateRecognizer(hub_ocr_model=model_variant)
    preds = model.run(
        [str(p) for p in all_paths],
        return_confidence=return_confidence,
    )

    out: dict[Hashable, str] = {}
    idx = 0
    for det_key, paths in segments:
        n = len(paths)
        chunk = preds[idx : idx + n]
        idx += n
        plates = [pr.plate for pr in chunk]
        out[det_key] = _six_char_plate_mode_first_on_tie(plates)

    return out


if __name__ == "__main__":
    model_variant = "cct-s-v2-global-model"

    image_folder = (
        "outputs/yolov8_lp_lux_640_192-168-100-22_05_13.mp4_test/license_plates"
    )

    exit_seconds_by_detection = {'00:47': 47.800000000000004, '13:55': 835.400088888889, '15:33': 933.8000888888889, '15:53': 953.6001, '23:01': 1381.8001000000002, '23:27': 1407.8001000000002, '30:42': 1842.0001111111112, '34:17': 2057.800111111111, '38:13': 2293.0001111111114, '40:39': 2439.8001, '40:57': 2457.4001000000003, '41:46': 2506.0001111111114, '45:16': 2716.800088888889, '52:10': 3130.2001}

    start_time = time.time()
    plates_by_detection = infer_license_plates_using_OCR(
        image_folder,
        exit_seconds_by_detection,
        model_variant,
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Number of detections: {len(plates_by_detection)}")
    for det_id, text in plates_by_detection.items():
        print(f"{det_id}: {text}")
