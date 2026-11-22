"""Shared EchoGesture video-to-keypoint preprocessing.

Copy this module unchanged into the dataset-generation environment so training
arrays and evaluator inputs use the same sampling, point order, and
normalization. ``extract_keypoints`` returns a float32 clip with shape
``(64, 162)``; save it directly with ``numpy.save``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

NUM_FRAMES = 64
NUM_LANDMARKS = 54
FEATURES = NUM_LANDMARKS * 3
MODEL_FEATURES = FEATURES * 3

# Order: left hand (21), right hand (21), then pose indices 11–16 and 23–28.
POSE_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def _landmark_xyz(landmark: Any) -> tuple[float, float, float]:
    return float(landmark.x), float(landmark.y), float(landmark.z)


def frame_to_features(results: Any) -> np.ndarray:
    """Convert one MediaPipe Holistic result to one flat ``(162,)`` row."""
    points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    if results.left_hand_landmarks:
        points[0:21] = [_landmark_xyz(p) for p in results.left_hand_landmarks.landmark]
    if results.right_hand_landmarks:
        points[21:42] = [_landmark_xyz(p) for p in results.right_hand_landmarks.landmark]
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        points[42:54] = [_landmark_xyz(pose[i]) for i in POSE_IDS]

        # The first two selected pose points are the left and right shoulders.
        shoulders = points[42:44]
        if np.any(shoulders):
            center = (shoulders[0] + shoulders[1]) * 0.5
            shoulder_width = float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]))
            scale = max(shoulder_width, 1e-3)
            detected = np.any(points != 0.0, axis=1)
            points[detected] = (points[detected] - center) / scale

    return points.reshape(FEATURES).astype(np.float32, copy=False)


def engineer_model_features(keypoints: np.ndarray) -> np.ndarray:
    """Combine positions, temporal velocity, and parent-relative bone vectors.

    Input is one ``(64, 162)`` keypoint clip. Output is ``(64, 486)`` in the
    fixed order ``[position, velocity, parent-relative vector]``. Keep this
    transformation shared by training, file inference, and evaluation.
    """
    array = np.asarray(keypoints, dtype=np.float32)
    if array.shape != (NUM_FRAMES, FEATURES):
        raise ValueError(f"expected keypoints shaped (64, 162), got {array.shape}")
    points = array.reshape(NUM_FRAMES, NUM_LANDMARKS, 3)
    velocity = np.diff(points, axis=0, prepend=points[:1])

    # Hand parent tree: wrist, then thumb and four fingers (each rooted at
    # the wrist). Pose links follow the selected shoulder/elbow/wrist and
    # hip/knee/ankle chains.
    parent = np.arange(NUM_LANDMARKS)
    hand_parent = (0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15,
                   0, 17, 18, 19)
    parent[:21] = hand_parent
    parent[21:42] = np.asarray(hand_parent) + 21
    parent[42:54] = (43, 42, 42, 43, 44, 45, 42, 43, 48, 49, 50, 51)
    relative = points - points[:, parent, :]
    valid = np.any(points != 0.0, axis=2) & np.any(points[:, parent, :] != 0.0, axis=2)
    relative *= valid[:, :, None]
    combined = np.concatenate((points, velocity, relative), axis=2)
    return combined.reshape(NUM_FRAMES, MODEL_FEATURES).astype(np.float32, copy=False)


def extract_keypoints(video_path: str | Path) -> np.ndarray:
    """Return uniformly sampled, normalized keypoints with shape ``(64,162)``.

    Unreadable videos raise ``ValueError``. A sampled frame that cannot be
    decoded, or has no detected hands/body, remains zero-filled.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video has no readable frames: {video_path}")

        frame_indices = np.linspace(0, frame_count - 1, NUM_FRAMES).round().astype(int)
        rows_by_frame: dict[int, list[int]] = {}
        for row, frame_index in enumerate(frame_indices):
            rows_by_frame.setdefault(int(frame_index), []).append(row)
        sequence = np.zeros((NUM_FRAMES, FEATURES), dtype=np.float32)
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # Process every decoded frame in order so MediaPipe's temporal
                # tracking and smoothing retain state, including on VFR clips.
                selected_rows = rows_by_frame.get(frame_index)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                if selected_rows and (results.left_hand_landmarks
                                      or results.right_hand_landmarks
                                      or results.pose_landmarks):
                    features = frame_to_features(results)
                    for row in selected_rows:
                        sequence[row] = features
                frame_index += 1

        return sequence.astype(np.float32, copy=False)
    finally:
        capture.release()
