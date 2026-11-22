#!/usr/bin/env python3
"""Offline webcam recognizer with speech and a recognized-word buffer."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

from src.preprocess import NUM_FRAMES, engineer_model_features, frame_to_features


class OfflineSpeaker:
    """Speak recognized words off the camera thread using local system TTS."""
    def __init__(self) -> None:
        self.jobs: queue.Queue[str | None] = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def speak(self, text: str) -> None:
        self.jobs.put(text)

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=0.5)

    def _worker(self) -> None:
        mac_say = shutil.which("say") if os.uname().sysname == "Darwin" else None
        engine = None
        while True:
            text = self.jobs.get()
            if text is None:
                return
            if mac_say:
                subprocess.run([mac_say, text], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
                continue
            try:
                if engine is None:
                    import pyttsx3  # Optional local-only fallback.
                    engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"Offline speech unavailable: {exc}", flush=True)
                return


def load_vocab(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        words = [str(word).upper() for word in raw]
    elif isinstance(raw, dict):
        words = [word for word, _ in sorted(((str(k).upper(), int(v)) for k, v in raw.items()), key=lambda x: x[1])]
    else:
        raise ValueError("vocab must be a list or word-to-ID object")
    if not words:
        raise ValueError("vocabulary is empty")
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model_int8.onnx"))
    parser.add_argument("--model-seed2", type=Path, default=Path("artifacts/model_seed2_int8.onnx"))
    parser.add_argument("--model-seed3", type=Path, default=Path("artifacts/model_seed3_int8.onnx"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--speech", action=argparse.BooleanOptionalAction, default=True,
                        help="speak stable predictions using offline system TTS")
    parser.add_argument("--speech-threshold", type=float, default=0.65)
    args = parser.parse_args()
    words = load_vocab(args.vocab)
    if not args.model.is_file():
        raise SystemExit(f"Primary ONNX model not found: {args.model}")
    model_paths = [args.model] + [path for path in (args.model_seed2, args.model_seed3)
                                  if path.is_file()]
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for path in model_paths]
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}")

    speaker = OfflineSpeaker() if args.speech else None
    history: list[np.ndarray] = []
    latest_words: list[str] = []
    latest_probs = np.zeros(3, dtype=np.float32)
    latency = 0.0
    candidate = ""
    candidate_frames = 0
    last_committed = ""
    sentence: list[str] = []
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    try:
        with mp_holistic.Holistic(static_image_mode=False, model_complexity=0,
                                  smooth_landmarks=True, enable_segmentation=False,
                                  refine_face_landmarks=False, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = holistic.process(rgb)
                rgb.flags.writeable = True
                if result.left_hand_landmarks:
                    mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                           mp_styles.get_default_hand_landmarks_style(),
                                           mp_styles.get_default_hand_connections_style())
                if result.right_hand_landmarks:
                    mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                           mp_styles.get_default_hand_landmarks_style(),
                                           mp_styles.get_default_hand_connections_style())
                if result.pose_landmarks:
                    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                           mp_styles.get_default_pose_landmarks_style())

                history.append(frame_to_features(result))
                if len(history) > NUM_FRAMES:
                    history.pop(0)
                if len(history) == NUM_FRAMES:
                    raw_clip = np.stack(history).astype(np.float32)
                    model_input = engineer_model_features(raw_clip)[None, ...]
                    start = time.perf_counter()
                    logits_by_model = []
                    for session in sessions:
                        input_name = session.get_inputs()[0].name
                        logits_by_model.append(
                            np.asarray(session.run(None, {input_name: model_input})[0],
                                       dtype=np.float64).reshape(-1)
                        )
                    latency = (time.perf_counter() - start) * 1000.0
                    logits = np.mean(logits_by_model, axis=0)
                    logits -= logits.max()
                    probs = np.exp(logits)
                    probs /= probs.sum()
                    order = np.argsort(probs)[::-1][:3]
                    latest_words = [words[int(i)] for i in order]
                    latest_probs = probs[order]

                    top_word = latest_words[0]
                    if float(latest_probs[0]) >= args.speech_threshold:
                        if top_word == candidate:
                            candidate_frames += 1
                        else:
                            candidate, candidate_frames = top_word, 1
                        if candidate_frames == 6 and top_word != last_committed:
                            sentence.append(top_word)
                            sentence[:] = sentence[-12:]
                            last_committed = top_word
                            if speaker:
                                speaker.speak(top_word)
                    else:
                        candidate, candidate_frames = "", 0
                        last_committed = ""

                h, w = frame.shape[:2]
                panel = frame.copy()
                cv2.rectangle(panel, (12, 12), (min(w - 12, 460), min(h - 12, 252)), (15, 20, 30), -1)
                frame = cv2.addWeighted(panel, 0.82, frame, 0.18, 0)
                main_word = latest_words[0] if latest_words else "READY"
                cv2.putText(frame, main_word, (28, 62), cv2.FONT_HERSHEY_DUPLEX, 1.15,
                            (100, 255, 170), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Latency: {latency:.1f} ms", (28, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 255), 1, cv2.LINE_AA)
                for i, (word, probability) in enumerate(zip(latest_words, latest_probs)):
                    y = 128 + i * 23
                    cv2.putText(frame, word, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                (230, 235, 245), 1, cv2.LINE_AA)
                    x0, x1 = 150, min(w - 70, 385)
                    cv2.rectangle(frame, (x0, y - 11), (x1, y - 2), (60, 68, 82), -1)
                    cv2.rectangle(frame, (x0, y - 11),
                                  (x0 + int((x1 - x0) * float(probability)), y - 2),
                                  (65, 185, 255), -1)
                    cv2.putText(frame, f"{probability:.0%}", (x1 + 8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 255), 1, cv2.LINE_AA)
                transcript = " ".join(sentence) if sentence else "(sign a word to begin)"
                cv2.putText(frame, "WORDS: " + transcript[:42], (28, 224),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (125, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "C clear  Q / ESC quit", (w - 190, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 235, 245), 1, cv2.LINE_AA)
                cv2.imshow("EchoGesture | Offline Webcam HUD", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    sentence.clear()
                    last_committed = ""
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if speaker:
            speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
