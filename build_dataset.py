#!/usr/bin/env python3
"""Build WLASL keypoint arrays using the inference-time extractor.

Expected input layout (as in common WLASL processed mirrors):
  --metadata data/wlasl/WLASL_v0.3.json
  --videos data/wlasl/videos/<instance-id>.mp4

Outputs data/train/<WORD>/ and data/val/<WORD>/ with one augmented copy
for each training clip, plus a sorted uppercase vocab.json list.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.preprocess import extract_keypoints


def _augment(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply clip-consistent in-plane rotation/scale and landmark dropout."""
    result = np.array(array, dtype=np.float32, copy=True).reshape(64, 54, 3)
    angle = np.deg2rad(rng.uniform(-12.0, 12.0))
    scale = rng.uniform(0.9, 1.1)
    c, s = np.cos(angle), np.sin(angle)
    xy = result[:, :, :2]
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    result[:, :, :2] = (xy @ rotation.T) * scale

    # Drop the same anatomical landmark across the full clip, preserving all
    # three coordinates and yielding coherent partial occlusion.
    dropped = rng.random(54) < 0.10
    result[:, dropped, :] = 0.0
    return result.reshape(64, 162).astype(np.float32, copy=False)


def _process(task: tuple[str, str, str, bool, int]) -> tuple[str, str | None]:
    video_path, output_path, augmented_path, make_aug, seed = task
    try:
        array = np.asarray(extract_keypoints(video_path), dtype=np.float32)
        if array.shape != (64, 162):
            raise ValueError(f"extract_keypoints returned {array.shape}, expected (64, 162)")
        if not np.isfinite(array).all():
            raise ValueError("extract_keypoints returned non-finite values")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, array)
        if make_aug:
            np.save(augmented_path, _augment(array, np.random.default_rng(seed)))
        return video_path, None
    except Exception as exc:  # report bad source clips and continue the batch
        return video_path, f"{type(exc).__name__}: {exc}"


def _entries(metadata_path: Path, videos_dir: Path, limit: int) -> tuple[list[str], list[dict]]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected WLASL metadata JSON to contain a list of gloss entries")
    # The official metadata is ordered by gloss frequency; the first 100
    # entries define WLASL-100.
    selected = data[:limit]
    words = [str(item["gloss"]).strip().upper() for item in selected]
    tasks = []
    for item, word in zip(selected, words):
        for instance in item.get("instances", []):
            split = instance.get("split")
            if split not in {"train", "val", "validation"}:
                continue
            video_id = str(instance.get("video_id", "")).strip()
            if not video_id:
                continue
            path = videos_dir / f"{video_id}.mp4"
            if path.is_file():
                tasks.append({"word": word, "split": "val" if split in {"val", "validation"} else "train",
                              "video": path, "video_id": video_id})
    return sorted(set(words)), tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/wlasl/WLASL_v0.3.json"))
    parser.add_argument("--videos", type=Path, default=Path("data/wlasl/videos"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--classes", type=int, default=100, help="top-K WLASL glosses (default: 100)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.classes < 1 or args.workers < 1:
        parser.error("--classes and --workers must be positive")
    words, entries = _entries(args.metadata, args.videos, args.classes)
    if not entries:
        raise SystemExit(f"No matching MP4s found under {args.videos}; expected <video_id>.mp4 files")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vocab.json").write_text(json.dumps(words, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for index, entry in enumerate(entries):
        base = args.output / entry["split"] / entry["word"] / entry["video_id"]
        jobs.append((str(entry["video"]), str(base.with_suffix(".npy")),
                     str(base.with_name(base.name + "_aug").with_suffix(".npy")),
                     entry["split"] == "train", args.seed + index))

    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process, job) for job in jobs]
        for future in as_completed(futures):
            source, error = future.result()
            if error:
                failures.append((source, error))
                print(f"FAILED {source}: {error}", flush=True)
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Processed {completed}/{len(jobs)}", flush=True)

    print(f"Saved vocab for {len(words)} classes; processed {len(jobs) - len(failures)}/{len(jobs)} clips.")
    if failures:
        print(f"{len(failures)} clips failed; see messages above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
