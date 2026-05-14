from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from onnx_model_utils import (
    get_project_root,
    prepare_onnx_for_onnxruntime,
    read_images_input_batch_size,
)
from video_utils import read_mp4_video_details, display_first_frame_center_crop_subplots


def save_inference_plot(
    image: Union[str, Path, np.ndarray],
    result: dict,
    output_path: Union[str, Path],
) -> None:
    """Draw object-detection boxes (Roboflow center x,y + width,height) and save."""
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")

    for pred in result.get("predictions") or []:
        x = pred.get("x")
        y = pred.get("y")
        w = pred.get("width")
        h = pred.get("height")
        if None in (x, y, w, h):
            continue
        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = pred.get("class", "")
        conf = pred.get("confidence")
        label = f"{cls}" if conf is None else f"{cls} {float(conf):.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), img)


def bbox_crop_from_prediction(img_bgr: np.ndarray, pred: dict[str, Any]) -> np.ndarray:
    """Crop the axis-aligned box given by Roboflow-style center x,y and width,height."""
    x = pred.get("x")
    y = pred.get("y")
    w = pred.get("width")
    h = pred.get("height")
    if None in (x, y, w, h):
        raise ValueError("prediction missing box geometry")
    H, W = img_bgr.shape[:2]
    x1 = int(round(float(x) - float(w) / 2))
    y1 = int(round(float(y) - float(h) / 2))
    x2 = int(round(float(x) + float(w) / 2))
    y2 = int(round(float(y) + float(h) / 2))
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return img_bgr[y1:y2, x1:x2].copy()


def ultralytics_to_dict(r) -> dict:
    predictions = []
    if r.boxes is None or len(r.boxes) == 0:
        return {"predictions": predictions}
    names = r.names or {}
    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        conf = float(r.boxes.conf[i])
        cls_id = int(r.boxes.cls[i])
        predictions.append({
            "x": cx,
            "y": cy,
            "width": w,
            "height": h,
            "confidence": conf,
            "class": names.get(cls_id, str(cls_id)),
        })
    return {"predictions": predictions}


@dataclass(frozen=True)
class SavedWindowPrediction:
    """One saved crop annotated with a single detection."""

    path: Path
    detected_at_sec: float
    confidence: float
    inference: dict
    license_plate_path: Path | None = None


@dataclass
class VideoTimestampInferResult:
    """One entry per anchor timestamp from ``timestamps_seconds``."""

    anchor_timestamp_sec: float
    saves: List[SavedWindowPrediction]


def center_crop_square(frame_bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """Return a ``size``×``size`` BGR crop from the frame center.

    If the frame is smaller than ``size`` in either dimension, the frame is scaled
    uniformly so the shorter side becomes ``size``, then center-cropped.
    """
    h, w = frame_bgr.shape[:2]
    if w >= size and h >= size:
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        return frame_bgr[y1 : y1 + size, x1 : x1 + size].copy()
    scale = size / min(w, h)
    nw = max(size, int(round(w * scale)))
    nh = max(size, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x1 = (nw - size) // 2
    y1 = (nh - size) // 2
    return resized[y1 : y1 + size, x1 : x1 + size].copy()


def infer_video_at_timestamps(
    video_path: str | Path,
    timestamps_detection_exit: dict[str, float],
    model: YOLO,
    output_dir: str | Path,
    *,
    conf: float = 0.2,
    window_begin: float = 5,
    window_end: float = 5,
    onnx_fixed_batch_size: int = 8,
    top_candidates_count: int = 5,
    anchor_fallback_frames: int = 3,
) -> List[VideoTimestampInferResult]:
    """For each anchor time ``t``, use ``[t - window_begin, t + window_end]``.

    Runs inference on frames in that interval in chunks of ``onnx_fixed_batch_size``
    (distinct consecutive frames; the last chunk may be padded for fixed-batch ONNX),
    collects all box predictions,
    keeps the top ``top_candidates_count`` boxes by confidence, saves one annotated
    crop per box under ``output_dir``, and saves the bbox crop of each box under
    ``output_dir/license_plates/`` with the same filename. If the model finds no
    boxes in the window, saves **three** center crops under ``detections/`` only from
    the frames whose timestamps are closest to the anchor (empty ``predictions``);
    nothing is written to ``license_plates/`` in that case.

    For ONNX models on Apple Silicon, pass ``device='mps'`` (default when ``device`` is
    ``None`` and MPS is available) so ONNX Runtime uses ``CoreMLExecutionProvider``.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_dir = out_dir / "detections"
    license_plates_dir = out_dir / "license_plates"
    detections_dir.mkdir(parents=True, exist_ok=True)
    license_plates_dir.mkdir(parents=True, exist_ok=True)

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    results: List[VideoTimestampInferResult] = []
    try:
        for id in timestamps_detection_exit:
            anchor = timestamps_detection_exit[id]

            # Crete a window of window_begin and window_end seconds before and after the anchor timestamp
            window_begin_sec = max(0.0, anchor - window_begin)
            window_end_sec = anchor + window_end
            cap.set(cv2.CAP_PROP_POS_MSEC, window_begin_sec * 1000.0)

            # (confidence, timestamp_sec, crop BGR, single prediction dict)
            candidates: List[Tuple[float, float, np.ndarray, dict[str, Any]]] = []

            # Every frame in the window (timestamp, center crop) for anchor fallback saves.
            window_snapshots: List[Tuple[float, np.ndarray]] = []

            # Distinct frames in the window, inferred in batches of ``onnx_fixed_batch_size``.
            batch_frames: List[Tuple[float, np.ndarray]] = []

            def run_predict_batch(
                frames: List[Tuple[float, np.ndarray]],
                score_threshold: float,
            ) -> None:
                if not frames:
                    return
                times = [t for t, _ in frames]
                crops = [c for _, c in frames]
                n_real = len(frames)

                if onnx_fixed_batch_size > 1:
                    if len(crops) < onnx_fixed_batch_size:
                        crops = crops + [crops[-1]] * (
                            onnx_fixed_batch_size - len(crops)
                        )
                    predict_source: Union[np.ndarray, List[np.ndarray]] = crops
                else:
                    predict_source = crops[0]

                pred_results = model.predict(
                    predict_source,
                    imgsz=640,
                    conf=score_threshold,
                    classes=[0],
                    verbose=False,
                )
                if not isinstance(pred_results, list):
                    pred_results = [pred_results]

                for j in range(n_real):
                    r = pred_results[j]
                    pos_sec = times[j]
                    crop_bgr = frames[j][1]
                    inference_try = ultralytics_to_dict(r)
                    for pred in inference_try.get("predictions") or []:
                        pc = pred.get("confidence")
                        if pc is None:
                            continue
                        candidates.append(
                            (float(pc), pos_sec, crop_bgr, pred),
                        )

            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                pos_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if pos_sec < window_begin_sec:
                    continue
                if pos_sec > window_end_sec:
                    break

                crop_bgr = center_crop_square(frame_bgr, size=2000)
                window_snapshots.append((pos_sec, crop_bgr))
                batch_frames.append((pos_sec, crop_bgr))

                if onnx_fixed_batch_size <= 1:
                    run_predict_batch(batch_frames, conf)
                    batch_frames.clear()
                elif len(batch_frames) >= onnx_fixed_batch_size:
                    run_predict_batch(batch_frames, conf)
                    batch_frames.clear()

            if batch_frames:
                run_predict_batch(batch_frames, conf)
                batch_frames.clear()

            candidates.sort(key=lambda t: t[0], reverse=True)
            top_candidates = candidates[:top_candidates_count]

            saves: List[SavedWindowPrediction] = []
            if top_candidates:
                for rank, (det_conf, pos_sec, crop_bgr, pred) in enumerate(top_candidates):
                    inference_single = {"predictions": [pred]}
                    out_file = detections_dir / f"{id}_{rank:02d}.jpg"
                    plate_file = license_plates_dir / out_file.name
                    save_inference_plot(crop_bgr, inference_single, out_file)
                    plate_roi = bbox_crop_from_prediction(crop_bgr, pred)
                    cv2.imwrite(str(plate_file), plate_roi)
                    saves.append(
                        SavedWindowPrediction(
                            path=out_file,
                            detected_at_sec=pos_sec,
                            confidence=det_conf,
                            inference=inference_single,
                            license_plate_path=plate_file,
                        ),
                    )
            elif window_snapshots and anchor_fallback_frames > 0:
                ordered = sorted(
                    window_snapshots,
                    key=lambda t: (abs(t[0] - anchor), t[0]),
                )
                for rank, (pos_sec, crop_bgr) in enumerate(
                    ordered[:anchor_fallback_frames],
                ):
                    inference_single: dict[str, Any] = {"predictions": []}
                    out_file = detections_dir / f"{id}_{rank:02d}.jpg"
                    save_inference_plot(crop_bgr, inference_single, out_file)
                    saves.append(
                        SavedWindowPrediction(
                            path=out_file,
                            detected_at_sec=pos_sec,
                            confidence=0.0,
                            inference=inference_single,
                            license_plate_path=None,
                        ),
                    )

            results.append(
                VideoTimestampInferResult(
                    anchor_timestamp_sec=anchor,
                    saves=saves,
                ),
            )
    finally:
        cap.release()

    return results


######### Ultralytics ONNX (opset 22+ may fail on older ONNX Runtime; see ``prepare_onnx_for_onnxruntime``)
_REPO = get_project_root()
_ONNX_PREPARED = prepare_onnx_for_onnxruntime(_REPO / "lp_lux_yolov8_mask_640.onnx")
_ONNX_BATCH = read_images_input_batch_size(_ONNX_PREPARED)
print("ONNX Batch Size: ", _ONNX_BATCH)
model = YOLO(str(_ONNX_PREPARED), task="detect")


if __name__ == "__main__":

    exit_seconds_by_detection = {'00:47': 47.800000000000004, '13:55': 835.400088888889, '15:33': 933.8000888888889, '15:53': 953.6001, '23:01': 1381.8001000000002, '23:27': 1407.8001000000002, '30:42': 1842.0001111111112, '34:17': 2057.800111111111, '38:13': 2293.0001111111114, '40:39': 2439.8001, '40:57': 2457.4001000000003, '41:46': 2506.0001111111114, '45:16': 2716.800088888889, '52:10': 3130.2001}
    
    video = "videos/192-168-100-22/05/13.mp4"


    start_time = time.time()
    rows = infer_video_at_timestamps(
        video,
        exit_seconds_by_detection,
        model,
        conf=0.2,
        output_dir=f"outputs/yolov8_lp_lux_640_{video.split('/')[-3]}_{video.split('/')[-2]}_{video.split('/')[-1]}_test",
        onnx_fixed_batch_size=_ONNX_BATCH,
        top_candidates_count=5,
        window_end=7,
        window_begin=3,
        anchor_fallback_frames=3,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")