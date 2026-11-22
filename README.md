[README.md](https://github.com/user-attachments/files/35235405/README.md)
# EchoGesture Track 02

EchoGesture recognizes word-level ASL clips from WLASL keypoints and provides an offline webcam demo.

## Submission layout

The submission bundle should contain these files at these exact paths:

```text
team_name/
├── README.md
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── requirements.txt
│   └── README.md
└── results/
    ├── predictions.json
    ├── predictions.jsonl
    └── results.csv
```

Keep inference models at `artifacts/model_int8.onnx`; an optional second seed is loaded from `artifacts/model_seed2_int8.onnx` and its raw logits are averaged with the first model before softmax.

## Offline inference

Use Python 3.11 and install `src/requirements.txt`. Place the vocabulary at the repository root as `vocab.json`, or at `data/vocab.json` as a fallback. Call `src.main.predict(video_path)` for a JSON-ready object with `clip_id`, uppercase `prediction`, uppercase `top5`, `confidence`, and `latency_ms`.

`src/preprocess.py` extracts a `(64, 162)` raw landmark sequence and transforms it into the model's `(64, 486)` input by concatenating positions, temporal velocity, and parent-relative vectors. The same transform is used by training and evaluation.

## Build the results files

After the ONNX model(s), vocabulary, and validation set are available, run:

```bash
python quantize_and_eval.py
```

The evaluator writes `results/predictions.json`, `results/predictions.jsonl`, and `results/results.csv`. Cached `.npy` landmark sequences in `data/val/<WORD>/` are preferred; `.mp4` clips are processed through `predict()` when no cache exists.

For a bundle from the repository root:

```bash
zip -r team_name.zip README.md src/main.py src/preprocess.py src/requirements.txt src/README.md results/predictions.json results/predictions.jsonl results/results.csv
```
