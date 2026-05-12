"""Helpers for YOLO *.onnx exports that target unsupported ONNX opsets or fixed batch sizes."""

from __future__ import annotations

from pathlib import Path

import onnx

# ONNX Runtime up to 1.19.x officially validates ai.onnx opset ≤ 21 for many builds.
DEFAULT_MAX_AI_ONNX_OPSET = 21


def read_images_input_batch_size(onnx_path: str | Path) -> int:
    """Return the batch dimension of input ``images`` if fixed; else ``1``."""
    path = Path(onnx_path).expanduser().resolve()
    model = onnx.load(str(path))
    for inp in model.graph.input:
        if inp.name != "images":
            continue
        dim0 = inp.type.tensor_type.shape.dim[0]
        if dim0.HasField("dim_value") and dim0.dim_value > 0:
            return int(dim0.dim_value)
        return 1
    return 1


def prepare_onnx_for_onnxruntime(
    onnx_path: str | Path,
    *,
    max_ai_onnx_opset: int = DEFAULT_MAX_AI_ONNX_OPSET,
) -> Path:
    """Stamp ``ai.onnx`` / default-domain opset to at most ``max_ai_onnx_opset`` if needed.

    Newer Ultralytics exports may use opset 22+, which some ONNX Runtime builds reject at
    load time. This only adjusts the declared opset version (same operator set for typical
    YOLO graphs). Writes ``<stem>_opset{max}_ort.onnx`` beside the source when a patch is
    applied; otherwise returns the original path.
    """
    path = Path(onnx_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No ONNX file at {path}")

    model = onnx.load(str(path))
    changed = False
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version > max_ai_onnx_opset:
            opset.version = max_ai_onnx_opset
            changed = True

    if not changed:
        return path

    out = path.with_name(f"{path.stem}_opset{max_ai_onnx_opset}_ort{path.suffix}")
    onnx.save(model, str(out))
    onnx.checker.check_model(onnx.load(str(out)))
    return out
