"""
Parse all pipeline ``*.log`` files under ``<repo>/logs`` and write one CSV per log
under ``<repo>/csv``, mirroring any subdirectories. Columns: Date, Time, Camera ID,
Timestamp, Plate Detected, License Plate, Comments.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# Log line: ``YYYY-MM-DD HH:MM:SS LEVEL message``
_LOG_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\w+\s+(.+)$",
)

# Video name: … INFO Video name: 05/13.mp4
_VIDEO = re.compile(r"^Video name:\s*(.+?)\s*$")

# Stage 1 — timestamp_mm_ss[int] = MM:SS
_STAGE1_MMSS_A = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+1\s+[-—]\s+timestamp_mm_ss\[(?P<key>\d+)\]\s*=\s*(?P<val>.+?)\s*$",
    re.I,
)
# Stage 1 — Timestamp[key] = … (key may be MM:SS or numeric index; value may be MM:SS or "47.800 s")
_STAGE1_TIMESTAMP = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+1\s+[-—]\s+Timestamp\[(?P<key>[^\]]+)\]\s*=\s*(?P<val>.+?)\s*$",
    re.I,
)

# Stage 2 — Detected[key] = True|False (key is int index or MM:SS)
_STAGE2_DETECTED = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+2\s+[-—]\s+Detected\[(?P<key>[^\]]+)\]\s*=\s*(?P<val>True|False)\s*$",
    re.I,
)
# Alternate: detection_key=… timestamps_seconds=… license_plate_detected=…
_STAGE2_LEGACY = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+2\s+[-—]\s+detection_key=(?P<key>\d+)\s+"
    r"timestamps_seconds=(?P<secs>[^\s]+)\s+license_plate_detected=(?P<val>True|False)\s*$",
    re.I,
)

# Stage 3 — OCR plate text (key int or MM:SS)
_STAGE3_PLATE_A = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+3\s+[-—]\s+plates_by_detection\[(?P<key>[^\]]+)\]\s*=\s*(?P<val>.*)$",
    re.I,
)
_STAGE3_PLATE_B = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+3\s+[-—]\s+Plate\[(?P<key>[^\]]+)\]\s*=\s*(?P<val>.*)$",
    re.I,
)

_MMSS_TS = re.compile(r"^(\d{1,4}):(\d{2})$")


def _timestamp_sort_key(ts: str) -> tuple[float, str]:
    """Seconds since start of video for MM:SS or float seconds; unknown/empty last."""
    s = (ts or "").strip()
    if not s:
        return (float("inf"), "")
    m = _MMSS_TS.match(s)
    if m:
        minutes, seconds = int(m.group(1)), int(m.group(2))
        return (float(minutes * 60 + seconds), s)
    try:
        return (float(s), s)
    except ValueError:
        return (float("inf"), s)


def _detection_key_sort(k: str) -> tuple[float, str]:
    """Sort numeric keys by index, MM:SS keys by video time, else last."""
    k = str(k).strip()
    if _MMSS_TS.match(k):
        return _timestamp_sort_key(k)
    if k.isdigit():
        return (float(int(k)), k)
    return (float("inf"), k)


def _log_message(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    m = _LOG_LINE.match(line)
    return m.group(1).strip() if m else None


def parse_pipeline_log(log_path: Path) -> tuple[str, dict[str, dict[str, dict[str, object]]]]:
    """Return ``(video_name, per_camera)`` where ``per_camera[cam_id]['ts'|'det'|'plate'][key]``.

    Detection ``key`` is either a numeric string (``"0"``, ``"1"``) or ``MM:SS`` (``"00:47"``).
    """
    video_name = ""
    buckets: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(dict),
    )

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        msg = _log_message(raw)
        if msg is None:
            continue

        vm = _VIDEO.match(msg)
        if vm:
            video_name = vm.group(1).strip()
            continue

        m = _STAGE1_MMSS_A.match(msg)
        if m:
            cam = m.group("cam")
            key = m.group("key").strip()
            val = m.group("val").strip()
            buckets[cam]["ts"][key] = val
            continue

        m = _STAGE1_TIMESTAMP.match(msg)
        if m:
            cam = m.group("cam")
            key = m.group("key").strip()
            val = m.group("val").strip()
            if _MMSS_TS.match(key):
                buckets[cam]["ts"][key] = key
            else:
                buckets[cam]["ts"][key] = val
            continue

        m = _STAGE2_DETECTED.match(msg)
        if m:
            cam = m.group("cam")
            key = m.group("key").strip()
            buckets[cam]["det"][key] = m.group("val").strip().lower() == "true"
            continue

        m = _STAGE2_LEGACY.match(msg)
        if m:
            cam = m.group("cam")
            key = m.group("key").strip()
            buckets[cam]["det"][key] = m.group("val").strip().lower() == "true"
            buckets[cam]["ts_sec"][key] = m.group("secs").strip()
            continue

        for rx in (_STAGE3_PLATE_A, _STAGE3_PLATE_B):
            m = rx.match(msg)
            if m:
                cam = m.group("cam")
                key = m.group("key").strip()
                buckets[cam]["plate"][key] = m.group("val").strip()
                break

    return video_name, dict(buckets)


def build_rows(
    video_name: str,
    camera_id: str,
    cam_data: dict[str, dict[str, object]],
) -> list[tuple[str, str, str, str, bool, str]]:
    """Merge ts / det / plate by detection key for one camera."""
    ts_map = cam_data.get("ts", {})
    ts_sec_map = cam_data.get("ts_sec", {})
    det_map = cam_data.get("det", {})
    plate_map = cam_data.get("plate", {})

    keys = {str(k) for k in (set(ts_map) | set(ts_sec_map) | set(det_map) | set(plate_map))}
    sorted_keys = sorted(keys, key=_detection_key_sort)

    rows: list[tuple[str, str, str, str, bool, str]] = []
    for k in sorted_keys:
        if k in ts_map:
            timestamp = str(ts_map[k])
        elif k in ts_sec_map:
            timestamp = str(ts_sec_map[k])
        elif _MMSS_TS.match(k):
            timestamp = k
        else:
            timestamp = ""

        detected = bool(det_map[k]) if k in det_map else False
        plate = str(plate_map.get(k, ""))

        parts = Path(video_name.replace("\\", "/")).as_posix().split("/")
        if len(parts) >= 2:
            date_part = parts[-2]
            time_part = Path(parts[-1]).stem
        else:
            date_part = ""
            time_part = Path(video_name).stem

        rows.append((date_part, time_part, camera_id, timestamp, detected, plate))

    return rows


def write_pipeline_csv(
    log_path: Path,
    *,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Parse ``log_path`` and write a single combined CSV. Returns the path written.

    If ``output_path`` is set, that file is used. Otherwise ``output_dir / {log_stem}.csv``
    (default directory: same folder as the log).
    """
    log_path = log_path.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"No log file at {log_path}")

    video_name, cameras = parse_pipeline_log(log_path)

    if output_path is not None:
        out = output_path.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = (
            output_dir.expanduser().resolve()
            if output_dir
            else log_path.parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{log_path.stem}.csv"

    all_rows: list[tuple[str, str, str, str, bool, str]] = []
    for cam_id, data in sorted(cameras.items(), key=lambda x: x[0]):
        all_rows.extend(build_rows(video_name, cam_id, data))

    all_rows.sort(
        key=lambda r: (_timestamp_sort_key(r[3]), r[2]),
    )

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Time", "Camera ID", "Timestamp", "Plate Detected", "License Plate", "Comments"])
        for d, t, cid, ts, det, plate in all_rows:
            w.writerow([d, t, cid, ts, "True" if det else "False", plate, ""])

    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    repo = _repo_root()
    logs_dir = repo / "logs"
    csv_dir = repo / "csv"

    if not logs_dir.is_dir():
        print(f"No logs directory at {logs_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(logs_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files under {logs_dir}", file=sys.stderr)
        sys.exit(0)

    csv_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for log_path in log_files:
        rel = log_path.relative_to(logs_dir)
        out_path = (csv_dir / rel).with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written.append(write_pipeline_csv(log_path, output_path=out_path))

    for p in written:
        print(p)


if __name__ == "__main__":
    main()
