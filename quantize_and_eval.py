#!/usr/bin/env python3
"""Quantize up to three ONNX seeds and evaluate data/val submission metrics."""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.preprocess import NUM_LANDMARKS, extract_keypoints


def vocab_map(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        result = {str(word).upper(): i for i, word in enumerate(raw)}
    elif isinstance(raw, dict):
        result = {str(word).upper(): int(index) for word, index in raw.items()}
    else:
        raise ValueError("vocab must be a list of words or word-to-ID object")
    if not result or sorted(result.values()) != list(range(len(result))):
        raise ValueError("vocabulary IDs must be contiguous from zero")
    return result


def validation_sources(folder: Path, vocab: dict[str, int]) -> list[tuple[str, str, Path]]:
    """Collect class-labeled clips, preferring cached NPY by matching clip ID."""
    collected: dict[tuple[str, str], dict[str, Path]] = {}
    for class_dir in sorted(folder.iterdir()) if folder.is_dir() else []:
        word = class_dir.name.upper()
        if not class_dir.is_dir() or word not in vocab:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".mp4", ".npy"} or path.stem.endswith("_aug"):
                continue
            key = (word, path.stem)
            collected.setdefault(key, {})[path.suffix.lower()] = path
    sources = []
    for (word, clip_id), paths in sorted(collected.items()):
        cached = paths.get(".npy")
        video = paths.get(".mp4")
        if cached is not None:
            sources.append((word, clip_id, cached))
        elif video is not None:
            sources.append((word, clip_id, video))
    return sources


def corrupt_keypoints(keypoints: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(keypoints, dtype=np.float32).reshape(64, NUM_LANDMARKS, 3).copy()
    points[:, rng.random(NUM_LANDMARKS) < 0.10, :] = 0.0
    observed = np.any(points != 0.0, axis=2)
    noise = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
    points += noise * observed[:, :, None]
    return points.reshape(64, NUM_LANDMARKS * 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.onnx"))
    parser.add_argument("--quantized", type=Path, default=Path("artifacts/model_int8.onnx"))
    parser.add_argument("--model-seed2", type=Path, default=Path("artifacts/model_seed2.onnx"))
    parser.add_argument("--quantized-seed2", type=Path, default=Path("artifacts/model_seed2_int8.onnx"))
    parser.add_argument("--model-seed3", type=Path, default=Path("artifacts/model_seed3.onnx"))
    parser.add_argument("--quantized-seed3", type=Path, default=Path("artifacts/model_seed3_int8.onnx"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--val", type=Path, default=Path("data/val"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.quantized.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(args.model), str(args.quantized), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    model_paths = [args.quantized]
    for source, target in ((args.model_seed2, args.quantized_seed2),
                           (args.model_seed3, args.quantized_seed3)):
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gemm"])
        if target.is_file():
            model_paths.append(target)

    combined_size_mb = sum(path.stat().st_size for path in model_paths) / 1_000_000
    if combined_size_mb >= 20:
        raise SystemExit(f"Combined INT8 model size is {combined_size_mb:.4f} MB; must be < 20 MB")

    vocab = vocab_map(args.vocab)
    sources = validation_sources(args.val, vocab)
    if not sources:
        raise SystemExit(f"No class-labeled .npy or .mp4 clips found under {args.val}/<WORD>/")

    import src.main as inference
    inference.MODEL_PATH = args.quantized.resolve()
    inference.SEED2_MODEL_PATH = args.quantized_seed2.resolve()
    inference.SEED3_MODEL_PATH = args.quantized_seed3.resolve()
    inference.VOCAB_PATH = args.vocab.resolve()
    inference._sessions = None
    inference._vocab = None

    rng = np.random.default_rng(args.seed)
    predictions: list[dict] = []
    latencies: list[float] = []
    clean_correct = robust_correct = 0
    for index, (truth, clip_id, source) in enumerate(sources, start=1):
        raw_keypoints: np.ndarray | None = None
        if source.suffix.lower() == ".npy":
            cached = np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
            if cached.shape == (64, 162):
                raw_keypoints = cached
                result = inference.predict_keypoints(raw_keypoints, clip_id)
            elif cached.shape == (64, 486):
                result = inference.predict_engineered(cached, clip_id)
                raw_keypoints = cached[:, :162]
            else:
                raise ValueError(f"{source}: expected (64,162) or (64,486), got {cached.shape}")
        else:
            # Keep the public video inference path when no cached landmark file exists.
            result = inference.predict(str(source))
            raw_keypoints = extract_keypoints(source)

        predictions.append(result)
        latencies.append(float(result["latency_ms"]))
        clean_correct += result["prediction"] == truth

        noisy = corrupt_keypoints(raw_keypoints, rng)
        noisy_result = inference.predict_keypoints(noisy, clip_id)
        robust_correct += noisy_result["prediction"] == truth
        if index % 25 == 0 or index == len(sources):
            print(f"Evaluated {index}/{len(sources)} validation clips", flush=True)

    clean_top1 = clean_correct / len(sources) * 100.0
    robust_top1 = robust_correct / len(sources) * 100.0
    robustness_drop = clean_top1 - robust_top1
    p95_latency = float(np.percentile(latencies, 95))
    runtime = f"ONNX Runtime {ort.__version__} CPUExecutionProvider"
    device = platform.machine() or "unknown"
    rows = [
        {"metric": "top1_accuracy", "target": ">=70", "measured": f"{clean_top1:.4f}",
         "unit": "%", "device": device, "runtime": runtime,
         "pass": "PASS" if clean_top1 >= 70.0 else "FAIL"},
        {"metric": "latency_p95", "target": "<=50", "measured": f"{p95_latency:.4f}",
         "unit": "ms", "device": device, "runtime": runtime,
         "pass": "PASS" if p95_latency <= 50.0 else "FAIL"},
        {"metric": "model_size", "target": "<=20", "measured": f"{combined_size_mb:.4f}",
         "unit": "MB", "device": device, "runtime": runtime,
         "pass": "PASS" if combined_size_mb <= 20.0 else "FAIL"},
        {"metric": "accuracy_drop_robustness", "target": "<=5", "measured": f"{robustness_drop:.4f}",
         "unit": "pt", "device": device, "runtime": runtime,
         "pass": "PASS" if robustness_drop <= 5.0 else "FAIL"},
    ]

    args.results.mkdir(parents=True, exist_ok=True)
    with (args.results / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")
    (args.results / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    with (args.results / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["metric", "target", "measured", "unit", "device", "runtime", "pass"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Top-1={clean_top1:.2f}% | robust Top-1={robust_top1:.2f}% | "
          f"drop={robustness_drop:.2f} pt | combined size={combined_size_mb:.4f} MB | "
          f"combined P95={p95_latency:.2f} ms")
    print(f"Wrote {args.results / 'predictions.jsonl'}, {args.results / 'predictions.json'}, "
          f"and {args.results / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
