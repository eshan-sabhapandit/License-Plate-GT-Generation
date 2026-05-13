License Plate GT Generation
============================

Pipeline: vehicle motion (MOG2) → YOLOv8 ONNX plate detection → Fast Plate OCR.

## Project structure
- videos
- logs
- outputs

Prerequisites
-------------
- Python 3.12 or newer (see pyproject.toml)
- A YOLO ONNX model file at the repo root (e.g. lp_lux_yolov8_mask_640.onnx)
- Videos
- uv


1. Clone and enter the project
-----------------------------
  git clone <repository-url>
  cd License-Plate-GT-Generation


2. Create a virtual environment and install dependencies
-------------------------------
```shell
  uv venv
  source .venv/bin/activate
  uv sync --all
```

3. To run it, edit the `pipeline.toml` with video name and pipeline configurations and run the pipeline.
```shell
uv run run_pipeline.py -c pipeline.toml
```



4. Configure and data layout
----------------------------
  - Edit pipeline.toml: video path, [[cameras]] ids and zones, detection/OCR settings.
  - Put input videos under:  videos/<camera-id>/<path-to-file>.mp4
    Example: videos/192-168-100-22/05/13.mp4  when video = "05/13.mp4" in the config.
  - Ensure the ONNX file named in pipeline.toml [license_plate_detection] model_path
    exists under the project root (or adjust model_path).


5. Run the pipeline
-------------------
  With the venv activated, from the project root:

  python run_pipeline.py -c pipeline.toml

  Logs are written under logs/; crops under outputs/<camera-id>/...


6. Generate CSV from a log (optional)
--------------------------------------
  python generate_csv.py logs/<your-run>.log

  Writes a single combined CSV next to the log (see generate_csv.py -h for options).


Troubleshooting
---------------
  - Import or CUDA/CoreML: onnxruntime wheels vary by OS; on Apple Silicon CoreML may be used.
  - If pip fails on onnxruntime version, match the range in pyproject.toml for your platform.
