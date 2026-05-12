from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class VideoDetails:
    """Metadata read from the video container / decoder (may be inaccurate for VFR sources)."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def read_mp4_video_details(video_path: str | Path) -> VideoDetails:
    """Open a video file (e.g. MP4) and return resolution, FPS, and reported frame count."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    return VideoDetails(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def read_frames_at_timestamps(
    video_path: str | Path,
    timestamps_seconds: Sequence[float],
) -> List[Tuple[float, np.ndarray]]:
    """Seek to each timestamp (seconds), decode one frame; returns (timestamp, BGR frame) pairs."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    out: List[Tuple[float, np.ndarray]] = []
    try:
        for ts in timestamps_seconds:
            ts = float(ts)
            if ts < 0:
                raise ValueError(f"Negative timestamp not allowed: {ts}")
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"No frame decoded at {ts}s (check duration / codec)")
            out.append((ts, frame))
    finally:
        cap.release()
    return out


def center_crop(frame: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """Crop a BGR image to ``crop_width`` x ``crop_height`` from the geometric center."""
    h, w = frame.shape[:2]
    if crop_width > w or crop_height > h:
        raise ValueError(
            f"Crop size ({crop_width}x{crop_height}) exceeds frame size ({w}x{h})"
        )
    x1 = (w - crop_width) // 2
    y1 = (h - crop_height) // 2
    x2 = x1 + crop_width
    y2 = y1 + crop_height
    return frame[y1:y2, x1:x2].copy()


def crop_by_zone(frame: np.ndarray, zone: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a BGR image using zone=(x, y, width, height)."""
    x, y, w, h = zone
    if w <= 0 or h <= 0:
        raise ValueError(f"Zone width/height must be > 0, got ({w}x{h})")

    frame_h, frame_w = frame.shape[:2]
    x2 = x + w
    y2 = y + h
    if x < 0 or y < 0 or x2 > frame_w or y2 > frame_h:
        raise ValueError(
            f"Zone ({x}, {y}, {w}, {h}) is outside frame bounds ({frame_w}x{frame_h})"
        )
    return frame[y:y2, x:x2].copy()


def display_first_frame_center_crop_subplots(
    video_path: str | Path,
    crop_width: int | None = None,
    crop_height: int | None = None,
    *,
    zone: tuple[int, int, int, int] | None = None,
    figsize: tuple[float, float] = (12, 5),
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Read first frame and plot original + cropped image (zone crop or center crop)."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise ValueError(f"Could not read first frame from {path}")

    if zone is not None:
        cropped = crop_by_zone(frame, zone)
        crop_title = f"Zone crop (x={zone[0]}, y={zone[1]}, w={zone[2]}, h={zone[3]})"
    else:
        if crop_width is None or crop_height is None:
            raise ValueError(
                "Provide crop_width and crop_height for center crop, or pass zone=(x, y, w, h)."
            )
        cropped = center_crop(frame, crop_width, crop_height)
        crop_title = f"Center crop ({crop_width}×{crop_height})"

    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(rgb_full)
    axes[0].set_title("First frame (original)")
    axes[0].axis("off")
    axes[1].imshow(rgb_crop)
    axes[1].set_title(crop_title)
    axes[1].axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return frame, cropped
