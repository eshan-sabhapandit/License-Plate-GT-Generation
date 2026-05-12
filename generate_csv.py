"""
Parse pipeline log files and emit one CSV with columns:
Video name, Camera ID, Timestamp, Detected, Plate.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# Log line: ``YYYY-MM-DD HH:MM:SS LEVEL message``
_LOG_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+\w+\s+(.+)$",
)

# Video name: … INFO Video name: 05/13.mp4
_VIDEO = re.compile(r"^Video name:\s*(.+?)\s*$")

# Stage 1 — MM:SS exit time per detection index
_STAGE1_MMSS_A = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+1\s+[-—]\s+timestamp_mm_ss\[(?P<idx>\d+)\]\s*=\s*(?P<val>.+?)\s*$",
    re.I,
)
_STAGE1_MMSS_B = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+1\s+[-—]\s+Timestamp\[(?P<idx>\d+)\]\s*=\s*(?P<val>.+?)\s*$",
    re.I,
)

# Stage 2 — LP detector saw a plate crop (current run_pipeline format)
_STAGE2_DETECTED = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+2\s+[-—]\s+Detected\[(?P<idx>\d+)\]\s*=\s*(?P<val>True|False)\s*$",
    re.I,
)
# Alternate: detection_key=… timestamps_seconds=… license_plate_detected=…
_STAGE2_LEGACY = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+2\s+[-—]\s+detection_key=(?P<idx>\d+)\s+"
    r"timestamps_seconds=(?P<secs>[^\s]+)\s+license_plate_detected=(?P<val>True|False)\s*$",
    re.I,
)

# Stage 3 — OCR plate text
_STAGE3_PLATE_A = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+3\s+[-—]\s+plates_by_detection\[(?P<idx>\d+)\]\s*=\s*(?P<val>.*)$",
    re.I,
)
_STAGE3_PLATE_B = re.compile(
    r"^\[(?P<cam>[^\]]+)\]\s+Stage\s+3\s+[-—]\s+Plate\[(?P<idx>\d+)\]\s*=\s*(?P<val>.*)$",
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


def _log_message(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    m = _LOG_LINE.match(line)
    return m.group(1).strip() if m else None


def parse_pipeline_log(log_path: Path) -> tuple[str, dict[str, dict[str, dict[int, object]]]]:
    """Return ``(video_name, per_camera)`` where ``per_camera[cam_id]['ts'|'det'|'plate'][idx]``."""
    video_name = ""
    # cam -> category -> idx -> value
    buckets: dict[str, dict[str, dict[int, object]]] = defaultdict(
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

        for rx in (_STAGE1_MMSS_A, _STAGE1_MMSS_B):
            m = rx.match(msg)
            if m:
                cam, idx, val = m.group("cam"), int(m.group("idx")), m.group("val").strip()
                buckets[cam]["ts"][idx] = val
                break
        else:
            m = _STAGE2_DETECTED.match(msg)
            if m:
                cam, idx = m.group("cam"), int(m.group("idx"))
                buckets[cam]["det"][idx] = m.group("val").strip().lower() == "true"
                continue

            m = _STAGE2_LEGACY.match(msg)
            if m:
                cam, idx = m.group("cam"), int(m.group("idx"))
                buckets[cam]["det"][idx] = m.group("val").strip().lower() == "true"
                buckets[cam]["ts_sec"][idx] = m.group("secs").strip()
                continue

            for rx in (_STAGE3_PLATE_A, _STAGE3_PLATE_B):
                m = rx.match(msg)
                if m:
                    cam, idx = m.group("cam"), int(m.group("idx"))
                    buckets[cam]["plate"][idx] = m.group("val").strip()
                    break

    return video_name, dict(buckets)


def build_rows(
    video_name: str,
    camera_id: str,
    cam_data: dict[str, dict[int, object]],
) -> list[tuple[str, str, str, bool, str]]:
    """Merge ts / det / plate by detection index for one camera."""
    ts_map = cam_data.get("ts", {})
    ts_sec_map = cam_data.get("ts_sec", {})
    det_map = cam_data.get("det", {})
    plate_map = cam_data.get("plate", {})

    indices = sorted(
        set(ts_map) | set(ts_sec_map) | set(det_map) | set(plate_map),
        key=int,
    )

    rows: list[tuple[str, str, str, bool, str]] = []
    for i in indices:
        if i in ts_map:
            timestamp = str(ts_map[i])
        elif i in ts_sec_map:
            timestamp = str(ts_sec_map[i])
        else:
            timestamp = ""

        detected = bool(det_map[i]) if i in det_map else False
        plate = str(plate_map.get(i, ""))

        date = video_name.split('/')[-2]
        time = video_name.split('/')[-1][:-4]
        rows.append((date, time, camera_id, timestamp, detected, plate))

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
        w.writerow(["Date", "Time", "Camera ID", "Timestamp", "Detected", "License Plate"])
        for d, t, cid, ts, det, plate in all_rows:
            w.writerow([d, t, cid, ts, "True" if det else "False", plate])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pipeline log to a single combined CSV file.",
    )
    parser.add_argument(
        "log_file",
        type=Path,
        nargs="?",
        default=Path("logs/05_13_20260511_163511.log"),
        help="Path to pipeline .log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        dest="output_path",
        metavar="OUTPUT.csv",
        help="Output CSV path (default: <log_dir>/<log_stem>.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the CSV when -o is not set (default: same folder as the log)",
    )
    args = parser.parse_args()

    path = write_pipeline_csv(
        args.log_file,
        output_path=args.output_path,
        output_dir=args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
