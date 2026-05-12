from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from fast_plate_ocr import LicensePlateRecognizer

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})

# Filenames like ``00_01.jpg``: detection id before ``_``, variant index after.
_STEM_RE = re.compile(r"^(\d+)_(\d+)$")


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


def _group_license_plate_images_by_detection(
    folder: str | Path,
    *,
    recursive: bool = False,
) -> dict[int, list[Path]]:
    """Group ``{det}_{variant}.jpg`` crops under ``folder`` by detection id."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if recursive:
        candidates = (p for p in root.rglob("*") if _is_image_path(p))
    else:
        candidates = (p for p in root.iterdir() if _is_image_path(p))

    groups: dict[int, list[Path]] = {}
    for p in candidates:
        m = _STEM_RE.match(p.stem)
        if not m:
            continue
        det_id = int(m.group(1))
        groups.setdefault(det_id, []).append(p)

    for det_id in groups:
        groups[det_id].sort(
            key=lambda path: (
                int(_STEM_RE.match(path.stem).group(2)),
                path.name.lower(),
            ),
        )
    return groups


def _majority_plate(strings: list[str]) -> str:
    """Most frequent OCR string; ties broken lexicographically for stability."""
    if not strings:
        return ""
    counts = Counter(strings)
    top_n = max(counts.values())
    winners = [s for s, n in counts.items() if n == top_n]
  
    if len(winners) > 1:
        return winners[0]
    else:
        return min(winners)


def infer_license_plates_using_OCR(
    folder: str | Path,
    exit_seconds_by_detection: dict[int, float],
    model_variant: str,
    *,
    recursive: bool = False,
    return_confidence: bool = False,
) -> dict[int, str]:
    """Run OCR on plate crops and pick the majority plate string per detection id.

    Expects crops named ``{{detection_id}}_{{variant}}.jpg`` (e.g. ``00_00.jpg`` … ``00_02.jpg``)
    under ``folder``. The integer before ``_`` must match keys in
    ``exit_seconds_by_detection``. All images for one detection are inferred in one batched
    ``run`` call (together with other groups), then the most common plate text wins.

    Returns a dict with the same keys as ``exit_seconds_by_detection``; missing image groups
    map to ``""``.
    """
    groups = _group_license_plate_images_by_detection(folder, recursive=recursive)

    ordered_keys = sorted(exit_seconds_by_detection.keys())
    segments: list[tuple[int, list[Path]]] = [
        (k, groups.get(k, [])) for k in ordered_keys
    ]
    all_paths = [p for _, paths in segments for p in paths]

    if not all_paths:
        return {k: "" for k in ordered_keys}

    model = LicensePlateRecognizer(hub_ocr_model=model_variant)
    preds = model.run(
        [str(p) for p in all_paths],
        return_confidence=return_confidence,
    )

    out: dict[int, str] = {}
    idx = 0
    for det_id, paths in segments:
        n = len(paths)
        chunk = preds[idx : idx + n]
        idx += n
        plates = [pr.plate for pr in chunk]
        out[det_id] = _majority_plate(plates)

    return out


if __name__ == "__main__":
    model_variant = "cct-s-v2-global-model"

    image_folder = (
        "outputs/192-168-100-22_05/13.mp4/license_plates"
    )

    exit_seconds_by_detection = {0: '02:30', 1: '02:35', 2: '23:24', 3: '31:58'}

    start_time = time.time()
    plates_by_detection = infer_license_plates_using_OCR(
        image_folder,
        exit_seconds_by_detection,
        model_variant,
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Number of detections: {len(plates_by_detection)}")
    for det_id, text in sorted(plates_by_detection.items()):
        print(f"{det_id}: {text}")
