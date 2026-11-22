# EchoGesture inference

This folder contains the offline `predict(video_path) -> dict` wrapper and shared MediaPipe preprocessing. Raw clips are sampled to 64 frames with 54 landmarks × 3 coordinates (`(64, 162)`), then converted to `(64, 486)` model features by concatenating position, first-order temporal velocity, and parent-relative bone vectors.

## Python 3.11 setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

For a Linux headless container, install the system libraries needed by OpenCV and MediaPipe before installing Python packages:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/requirements.txt ./src/requirements.txt
RUN python -m pip install --no-cache-dir -r src/requirements.txt
COPY . .
```

## Model and vocabulary files

Place the primary quantized model at `artifacts/model_int8.onnx`. If present, `artifacts/model_seed2_int8.onnx` is also loaded. The CPU inference wrapper averages both models' raw logits before applying softmax. If the INT8 primary model is absent, it falls back to `artifacts/model.onnx`.

Vocabulary lookup checks repository-root `vocab.json`, then `data/vocab.json`. Set `ECHOGESTURE_VOCAB` to override that lookup. Set `ECHOGESTURE_MODEL` or `ECHOGESTURE_MODEL_SEED2` to override model paths.

`predict(video_path)` returns exactly these JSON-serializable keys:

- `clip_id`: input file stem
- `prediction`: uppercase top-1 word
- `top5`: uppercase words in descending probability order
- `confidence`: top-1 softmax probability rounded to four decimal places
- `latency_ms`: combined ONNX inference latency rounded to two decimal places

Latency covers ONNX model inference, including both seeds when the optional second model is available. It excludes video decoding and landmark extraction.

## Evaluation outputs

Run `python quantize_and_eval.py` from the repository root after training/export. The script quantizes MatMul and Gemm operators only, using the same operator allowlist for an optional second seed. It prefers matching cached `.npy` landmarks in `data/val/<WORD>/`, otherwise evaluates the `.mp4` through `predict()`.

It writes `results/predictions.json` (array), `results/predictions.jsonl` (one object per line), and `results/results.csv` with top-1 accuracy, combined P95 latency, combined model size, and clean-versus-corrupted landmark accuracy drop.

MediaPipe may download landmark model assets on first use depending on the installed package. Cache those assets before disconnecting if deployment must remain fully offline.
