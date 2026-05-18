from __future__ import annotations

import argparse
import logging
import time
import tomllib
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence


# from license_plate_detection import VideoTimestampInferResult, infer_video_at_timestamps
# from license_plate_recognition import infer_license_plates_using_OCR
from vehicle_detection import detect_motion_timestamps

# from onnx_model_utils import (
# get_project_root,
# prepare_onnx_for_onnxruntime,
# read_images_input_batch_size,
# )

logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    """Load and return pipeline TOML config."""
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def load_model(model_path: str, *, task: str = "segment"):
    ######### Ultralytics ONNX (opset 22+ may fail on older ONNX Runtime; see ``prepare_onnx_for_onnxruntime``)
    root = get_project_root()
    onnx_src = (root / Path(model_path)).expanduser()
    _ONNX_PREPARED = prepare_onnx_for_onnxruntime(onnx_src)
    _ONNX_BATCH = read_images_input_batch_size(_ONNX_PREPARED)
    logger.info("ONNX Batch Size: %s", _ONNX_BATCH)
    # model = YOLO(str(_ONNX_PREPARED), task=task)
    model = YOLO(model_path, task=task)
    return model, _ONNX_BATCH


def _video_parts(video: str) -> tuple[str, str]:
    """Return ``(middle_folder, stem)`` for paths like ``05/13.mp4`` → ``("05", "13")``."""
    posix = Path(video).as_posix()
    parts = posix.split("/")
    if len(parts) >= 2:
        return parts[-2], Path(parts[-1]).stem
    return "", Path(video).stem


def _log_stage1_camera(
    camera_id: str,
    exit_timestamps: Mapping[str, float],
    *,
    number_of_detections: int,
) -> None:
    """Log each exit (MM:SS key → seconds) and detection count."""
    logger.info("Camera ID: %s", camera_id)
    logger.info("[%s] Stage 1 — Number of detections: %s", camera_id, number_of_detections)
    for mm_ss in sorted(exit_timestamps.keys(), key=lambda k: exit_timestamps[k]):
        sec = exit_timestamps[mm_ss]
        logger.info(
            "[%s] Stage 1 - Timestamp[%s] = %.3f s",
            camera_id,
            mm_ss,
            sec,
        )


def _log_stage2_camera(
    camera_id: str,
    exit_timestamps: Mapping[str, float],
    results: Sequence[VideoTimestampInferResult],
) -> None:
    """Log each detection key, exit time (seconds), and whether LP crops were found."""
    logger.info("Camera ID: %s", camera_id)
    for (det_key, ts_sec), r in zip(exit_timestamps.items(), results, strict=True):
        had_detection = bool(r.saves)
        logger.info(
            "[%s] Stage 2 - Detected[%s] = %s",
            camera_id,
            det_key,
            had_detection,
        )


def _log_stage3_camera(
    camera_id: str,
    plates_by_detection: Mapping[Hashable, str],
) -> None:
    """Log OCR consensus plate string per detection key (one row per key)."""
    logger.info("Camera ID: %s", camera_id)
    for det_key in plates_by_detection:
        plate_text = plates_by_detection[det_key]
        logger.info(
            "[%s] Stage 3 — Plate[%s] = %s",
            camera_id,
            det_key,
            plate_text,
        )


def _setup_logging(log_dir_rel: str, video: str) -> Path:
    log_dir = (log_dir_rel).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    mm_dir, stem = _video_parts(video)
    log_prefix = f"{mm_dir}_{stem}" if mm_dir else stem
    log_file = log_dir / f"{log_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("Log file: %s", log_file)
    logger.info("Video name: %s", video)
    return log_file


def run_pipeline(config_path: str | Path) -> None:
    """Run vehicle detection → LP detection → OCR using parameters from a TOML config file."""
    cfg = load_pipeline_config(Path(config_path))

    video = cfg["video"]
    cameras_cfg = cfg["cameras"]
    if not cameras_cfg:
        raise ValueError("Config must include at least one [[cameras]] table.")
    vd = cfg.get("vehicle_detection", {})
    lpd = cfg.get("license_plate_detection", {})
    ocr_cfg = cfg.get("ocr", {})
    log_cfg = cfg.get("logging", {})

    videos_subdir = cfg.get("videos_subdir", "videos")

    _setup_logging(log_cfg.get("log_dir", "logs"), video)

    vd_threshold = int(vd.get("threshold", 10000))
    vd_history = int(vd.get("history", 1000))
    vd_var = int(vd.get("var_threshold", 50))

    model_rel = lpd.get("model_path", "lp_lux_yolov8_mask_640.onnx")
    yolo_task = lpd.get("yolo_task", "segment")
    top_k = int(lpd.get("top_candidates_count", 5))
    half_win = float(lpd.get("half_window_sec", 5.0))
    window_begin = float(lpd.get("window_begin", half_win))
    window_end = float(lpd.get("window_end", half_win))
    lpd_conf = float(lpd.get("conf", 0.2))
    anchor_fallback_frames = int(lpd.get("anchor_fallback_frames", 3))
    ocr_variant = ocr_cfg.get("model_variant", "cct-s-v2-global-model")

    mm_dir, file_stem = _video_parts(video)

    start_time = time.time()

    ########################### Stage 1: Vehicle Detection ###########################
    logger.info("Stage 1: Vehicle Detection")
    stage1: list[tuple[str, dict[str, float], Path]] = []

    for cam in cameras_cfg:
        cam_id = cam["id"]
        zone = tuple(int(x) for x in cam["zone"])
        video_path = videos_subdir / cam_id / video
        tic = time.time()
        exit_timestamps = detect_motion_timestamps(
            str(video_path),
            zone=zone,
            threshold=vd_threshold,
            history=vd_history,
            varThreshold=vd_var,
        )
        toc = time.time()
        logger.info("Time taken: %.3f seconds", toc - tic)
        if isinstance(exit_timestamps, str):
            raise RuntimeError(
                f"Vehicle detection failed for {cam_id} ({video_path}): {exit_timestamps}",
            )
        stage1.append((cam_id, exit_timestamps, video_path))
            
    for cam_id, exit_ts, _ in stage1:
        n_det = len(exit_ts)
        _log_stage1_camera(cam_id, exit_ts, number_of_detections=n_det)


    ########################### Stage 2: License Plate Detection ###########################
    # from ultralytics import YOLO
    logger.info("Stage 2: License Plate Detection")
    model, onnx_batch = load_model(model_rel, task=yolo_task)

    stage2_results: list[
        tuple[str, dict[str, float], list[VideoTimestampInferResult], Path]
    ] = []

    for cam_id, exit_ts, video_path in stage1:
        if mm_dir:
            output_dir = repo_root / "outputs" / cam_id / mm_dir / file_stem
        else:
            output_dir = repo_root / "outputs" / cam_id / file_stem

        results = infer_video_at_timestamps(
            video_path,
            exit_ts,
            model,
            output_dir=output_dir,
            conf=lpd_conf,
            window_begin=window_begin,
            window_end=window_end,
            onnx_fixed_batch_size=onnx_batch,
            top_candidates_count=top_k,
            anchor_fallback_frames=anchor_fallback_frames,
        )
        stage2_results.append((cam_id, exit_ts, results, output_dir))
        _log_stage2_camera(cam_id, exit_ts, results)

    ########################### Stage 3: License Plate Recognition ###########################
    logger.info("Stage 3: License Plate Recognition")

    for cam_id, exit_ts, _, output_dir in stage2_results:
        # plates = infer_license_plates_using_OCR(
        #     output_dir / "license_plates",
        #     exit_ts,
        #     ocr_variant,
        # )
        plates = {key: "" for key in exit_ts}
        _log_stage3_camera(cam_id, plates)

    elapsed = time.time() - start_time
    logger.info("Total time taken: %.3f seconds", elapsed)


def main() -> None:
    # repo_root = get_project_root()
    parser = argparse.ArgumentParser(description="License plate GT pipeline (config-driven).")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="pipeline.toml",
        help="Path to pipeline TOML config (default: pipeline.toml at project root)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
