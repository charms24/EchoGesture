"""Offline EchoGesture inference wrapper used by the Samsung evaluator."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:  # Supports package imports and direct loading from the src directory.
    from .preprocess import engineer_model_features, extract_keypoints
except ImportError:
    from preprocess import engineer_model_features, extract_keypoints

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("ECHOGESTURE_MODEL", ROOT / "artifacts" / "model_int8.onnx"))
if not MODEL_PATH.exists():
    MODEL_PATH = Path(os.environ.get("ECHOGESTURE_MODEL", ROOT / "artifacts" / "model.onnx"))
SEED2_MODEL_PATH = Path(os.environ.get("ECHOGESTURE_MODEL_SEED2", ROOT / "artifacts" / "model_seed2_int8.onnx"))
SEED3_MODEL_PATH = Path(os.environ.get("ECHOGESTURE_MODEL_SEED3", ROOT / "artifacts" / "model_seed3_int8.onnx"))
if "ECHOGESTURE_VOCAB" in os.environ:
    VOCAB_PATH = Path(os.environ["ECHOGESTURE_VOCAB"])
else:
    VOCAB_PATH = ROOT / "vocab.json"
    if not VOCAB_PATH.is_file():
        VOCAB_PATH = ROOT / "data" / "vocab.json"
_sessions: list[ort.InferenceSession] | None = None
_vocab: dict[str, int] | None = None


def _load_vocab() -> dict[str, int]:
    global _vocab
    if _vocab is None:
        raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            _vocab = {str(word).upper(): index for index, word in enumerate(raw)}
        elif isinstance(raw, dict):
            _vocab = {str(word).upper(): int(index) for word, index in raw.items()}
        else:
            raise ValueError("vocab.json must be a word list or map words to integer IDs")
        if not _vocab or sorted(_vocab.values()) != list(range(len(_vocab))):
            raise ValueError("vocab.json must contain contiguous class IDs")
    return _vocab


def _get_sessions() -> list[ort.InferenceSession]:
    global _sessions
    if _sessions is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        paths = [MODEL_PATH]
        paths.extend(path for path in (SEED2_MODEL_PATH, SEED3_MODEL_PATH)
                     if path.is_file())
        _sessions = [ort.InferenceSession(str(path), sess_options=options,
                                          providers=["CPUExecutionProvider"])
                     for path in paths]
    return _sessions


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _result_from_probabilities(probabilities: np.ndarray, clip_id: str,
                               latency_ms: float) -> dict[str, Any]:
    vocab = _load_vocab()
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probabilities.size != len(vocab):
        raise ValueError(f"Model returned {probabilities.size} classes, vocab has {len(vocab)}")
    id_to_word = {index: word for word, index in vocab.items()}
    order = np.argsort(probabilities)[::-1][:min(5, len(vocab))]
    top5 = [id_to_word[int(index)] for index in order]
    return {
        "clip_id": str(clip_id),
        "prediction": str(top5[0]).upper(),
        "top5": [str(word).upper() for word in top5],
        "confidence": round(float(probabilities[int(order[0])]), 4),
        "latency_ms": round(float(latency_ms), 2),
    }


def _ensemble_probabilities(features: np.ndarray) -> tuple[np.ndarray, float]:
    """Infer pass batches and combine models according to runtime flags."""
    if features.ndim != 3 or features.shape[1:] != (64, 486):
        raise ValueError(f"expected engineered features shaped (N, 64, 486), got {features.shape}")
    sessions = _get_sessions()
    raw_weights = os.environ.get("ECHOGESTURE_WEIGHTS", "").strip()
    if raw_weights:
        try:
            weights = np.asarray([float(value.strip()) for value in raw_weights.split(",")],
                                 dtype=np.float64)
        except ValueError as exc:
            raise ValueError("ECHOGESTURE_WEIGHTS must be comma-separated numbers") from exc
        if (weights.size != len(sessions) or not np.isfinite(weights).all()
                or np.any(weights < 0.0) or weights.sum() <= 0.0):
            raise ValueError(f"ECHOGESTURE_WEIGHTS requires {len(sessions)} finite nonnegative weights with positive sum")
        weights /= weights.sum()
    else:
        weights = np.full(len(sessions), 1.0 / len(sessions), dtype=np.float64)

    average_mode = os.environ.get("ECHOGESTURE_AVG", "logit").strip().lower()
    if average_mode not in {"logit", "softmax"}:
        raise ValueError("ECHOGESTURE_AVG must be 'logit' or 'softmax'")

    start = time.perf_counter()
    logits_by_model = []
    for session in sessions:
        input_name = session.get_inputs()[0].name
        output = np.asarray(session.run(None, {input_name: features})[0], dtype=np.float64)
        if output.ndim == 1:
            output = output[np.newaxis, :]
        if output.shape[0] != features.shape[0]:
            raise ValueError(f"Model returned {output.shape[0]} rows for {features.shape[0]} input passes")
        logits_by_model.append(output)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if average_mode == "logit":
        combined_logits = np.zeros_like(logits_by_model[0], dtype=np.float64)
        for weight, logits in zip(weights, logits_by_model):
            combined_logits += weight * logits
        per_pass_probabilities = _softmax(combined_logits)
    else:
        per_pass_probabilities = np.zeros_like(logits_by_model[0], dtype=np.float64)
        for weight, logits in zip(weights, logits_by_model):
            per_pass_probabilities += weight * _softmax(logits)
    return np.mean(per_pass_probabilities, axis=0), latency_ms


def predict_engineered(features: np.ndarray, clip_id: str) -> dict[str, Any]:
    """Predict from one engineered clip, averaging probabilities across models."""
    if features.shape != (64, 486):
        raise ValueError(f"expected engineered features shaped (64, 486), got {features.shape}")
    probabilities, latency_ms = _ensemble_probabilities(features[np.newaxis, ...])
    return _result_from_probabilities(probabilities, clip_id, latency_ms)


def predict_keypoints(keypoints: np.ndarray, clip_id: str) -> dict[str, Any]:
    """Predict raw landmarks with optional four-view deterministic TTA."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (64, 162):
        raise ValueError(f"expected keypoints shaped (64, 162), got {points.shape}")
    use_tta = os.environ.get("ECHOGESTURE_TTA", "0").strip()
    if use_tta not in {"0", "1"}:
        raise ValueError("ECHOGESTURE_TTA must be '0' or '1'")
    variants = [points]
    if use_tta == "1":
        point_tensor = points.reshape(64, 54, 3)
        # Center temporal zoom using the specified span and linear interpolation.
        sample_at = np.linspace(3.15, 59.85, 64, dtype=np.float32)
        lower = np.floor(sample_at).astype(np.int64)
        upper = np.minimum(lower + 1, 63)
        fraction = (sample_at - lower)[:, None, None]
        zoomed = point_tensor[lower] * (1.0 - fraction) + point_tensor[upper] * fraction
        variants.append(zoomed.reshape(64, 162).astype(np.float32, copy=False))

        # Preserve all-zero landmark rows for the two in-plane views.
        for scale, angle_degrees in ((1.05, 4.0), (0.95, -4.0)):
            transformed = point_tensor.copy()
            present = np.any(transformed != 0.0, axis=2)
            angle = np.deg2rad(angle_degrees)
            cosine, sine = np.cos(angle), np.sin(angle)
            x = transformed[:, :, 0].copy()
            y = transformed[:, :, 1].copy()
            transformed[:, :, 0][present] = scale * (cosine * x[present] - sine * y[present])
            transformed[:, :, 1][present] = scale * (sine * x[present] + cosine * y[present])
            variants.append(transformed.reshape(64, 162).astype(np.float32, copy=False))

    engineered = np.stack([engineer_model_features(variant) for variant in variants], axis=0)
    probabilities, latency_ms = _ensemble_probabilities(engineered)
    return _result_from_probabilities(probabilities, clip_id, latency_ms)


def predict(video_path: str) -> dict[str, Any]:
    """Return the submission schema for a video clip."""
    return predict_keypoints(extract_keypoints(video_path), Path(video_path).stem)
