#!/usr/bin/env python3
"""Train a compact temporal CNN for 64-frame sign-language keypoint clips.

Expected layout (class folder names are uppercase vocabulary words):
  data/train/HAPPY/*.npy
  data/val/HAPPY/*.npy   # optional; otherwise stratified split from train

Every array must be float-convertible and have shape (64, 162). The values
must use the same normalization convention used by the video preprocessor.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.preprocess import engineer_model_features

INPUT_FEATURES = 486
NUM_LANDMARKS = 54


def augment_keypoints(array: np.ndarray, rng: np.random.Generator, *,
                      spatial_temporal: bool, robust: bool) -> np.ndarray:
    """Apply original-clip transforms and/or robust occlusion noise."""
    points = np.asarray(array, dtype=np.float32).reshape(64, NUM_LANDMARKS, 3).copy()

    if spatial_temporal:
        fraction = rng.uniform(0.78, 1.0)
        span = max(2.0, (64 - 1) * fraction)
        start = rng.uniform(0.0, 63.0 - span)
        sample_at = np.linspace(start, start + span, 64, dtype=np.float32)
        lower = np.floor(sample_at).astype(np.int64)
        upper = np.minimum(lower + 1, 63)
        weight = (sample_at - lower)[:, None, None]
        points = points[lower] * (1.0 - weight) + points[upper] * weight

        angle = np.deg2rad(rng.uniform(-10.0, 10.0))
        scale = rng.uniform(0.92, 1.08)
        cosine, sine = np.cos(angle), np.sin(angle)
        x, y = points[:, :, 0].copy(), points[:, :, 1].copy()
        points[:, :, 0] = scale * (cosine * x - sine * y)
        points[:, :, 1] = scale * (sine * x + cosine * y)

    if robust:
        points[:, rng.random(NUM_LANDMARKS) < 0.10, :] = 0.0
        observed = np.any(points != 0.0, axis=2)
        jitter = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
        points += jitter * observed[:, :, None]
    return points.reshape(64, 162).astype(np.float32, copy=False)


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        branch_width = max(16, channels // 3)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, branch_width, kernel_size=k,
                                    padding=k // 2, bias=False),
                          nn.BatchNorm1d(branch_width), nn.GELU())
            for k in (3, 5, 9)
        ])
        self.project = nn.Sequential(
            nn.Conv1d(branch_width * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.dropout(self.project(mixed)))


class EchoGestureNet(nn.Module):
    """Multi-scale residual temporal CNN; input is N x 64 x 486."""
    def __init__(self, num_classes: int, width: int = 128, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_FEATURES, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            MultiScaleResidualBlock(width, dropout),
            MultiScaleResidualBlock(width, dropout),
            MultiScaleResidualBlock(width, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ONNX input convention: N,T,F. Conv1d convention: N,F,T.
        x = x.transpose(1, 2)
        return self.head(self.temporal(self.stem(x)))


class DilatedSeparableBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.35):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels), nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class WideCNN(nn.Module):
    """Wider depthwise-separable CNN with temporal dilation 1, 2, 4, 8."""
    def __init__(self, num_classes: int, width: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_FEATURES, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.temporal = nn.Sequential(*[
            DilatedSeparableBlock(width, dilation) for dilation in (1, 2, 4, 8)
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.4), nn.Linear(width, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.temporal(self.stem(x.transpose(1, 2))))


class GRUCNN(nn.Module):
    """Compact temporal convolution + BiGRU + learned attention pooling."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(INPUT_FEATURES, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.gru = nn.GRU(128, 160, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(320, 1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(320, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.gru(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)


# Parent assignments transcribed from src/preprocess.py's engineer_model_features:
# the two 21-point hand trees followed by pose points 11-16 and 23-28.
_HAND_PARENTS = (0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15,
                 0, 17, 18, 19)
LANDMARK_PARENTS = (
    *_HAND_PARENTS,
    *(parent + 21 for parent in _HAND_PARENTS),
    43, 42, 42, 43, 44, 45, 42, 43, 48, 49, 50, 51,
)


def landmark_adjacency() -> torch.Tensor:
    adjacency = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.float32)
    for vertex, parent in enumerate(LANDMARK_PARENTS):
        if vertex != parent:
            adjacency[vertex, parent] = 1.0
            adjacency[parent, vertex] = 1.0
    adjacency += np.eye(NUM_LANDMARKS, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return torch.from_numpy(normalized.astype(np.float32))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                      padding=(4, 0), bias=False),
            nn.BatchNorm2d(out_channels), nn.Dropout(0.35),
        )
        self.residual = (nn.Identity() if in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                 bias=False), nn.BatchNorm2d(out_channels)))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("nctv,vw->nctw", x, adjacency)
        output = self.temporal(self.spatial(spatial))
        return self.activation(output + self.residual(x))


class STGCN(nn.Module):
    """Spatial-temporal graph CNN over the shared 54-landmark graph."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.register_buffer("A", landmark_adjacency(), persistent=True)
        self.blocks = nn.ModuleList([
            STGCNBlock(9, 64), STGCNBlock(64, 96), STGCNBlock(96, 128),
        ])
        self.temporal_attention = nn.Conv1d(128, 1, kernel_size=1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, 64, NUM_LANDMARKS, 9).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=3)
        attention = torch.softmax(self.temporal_attention(x), dim=2)
        pooled = torch.sum(x * attention, dim=2)
        return self.classifier(pooled)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int = 128, reduction: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=2)
        return x * self.gate(summary).unsqueeze(-1)


class AttentionCNN(nn.Module):
    """EchoGesture multi-scale CNN with SE gates and temporal attention."""
    def __init__(self, num_classes: int):
        super().__init__()
        width, dropout = 128, 0.4
        self.stem = nn.Sequential(
            nn.Conv1d(INPUT_FEATURES, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.blocks = nn.ModuleList([MultiScaleResidualBlock(width, dropout) for _ in range(3)])
        self.se_gates = nn.ModuleList([SqueezeExcitation(width, reduction=4) for _ in range(3)])
        self.temporal_attention = nn.Conv1d(width, 1, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x.transpose(1, 2))
        for block, gate in zip(self.blocks, self.se_gates):
            x = gate(block(x))
        attention = torch.softmax(self.temporal_attention(x), dim=2)
        pooled = torch.sum(x * attention, dim=2)
        return self.classifier(pooled)


def create_model(arch: str, num_classes: int, width: int = 128) -> nn.Module:
    if arch == "cnn":
        return EchoGestureNet(num_classes, width=width)
    if arch == "gru_cnn":
        return GRUCNN(num_classes)
    if arch == "wide_cnn":
        return WideCNN(num_classes)
    if arch == "stgcn":
        return STGCN(num_classes)
    if arch == "attn_cnn":
        return AttentionCNN(num_classes)
    raise ValueError(f"unsupported architecture: {arch}")


class NpyClips(Dataset):
    def __init__(self, items: Sequence[tuple[Path, int]], augment: bool = False,
                 robust_aug: bool = True, seed: int = 42):
        self.items = list(items)
        self.augment = augment
        self.robust_aug = robust_aug
        self.seed = int(seed)
        self.epoch = 0
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rng = None
        self._rng_seed = None

    def _get_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        # DataLoader's worker seed derives from its seeded generator and is
        # renewed for each worker iterator/epoch. In the main process, derive
        # an epoch-specific stream directly from the configured base seed.
        seed = int(worker.seed) if worker is not None else self.seed + self.epoch * 1_000_003
        if self._rng is None or self._rng_seed != seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed
        return self._rng

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[index]
        array = np.load(path, allow_pickle=False)
        if array.shape != (64, 162):
            raise ValueError(f"{path}: expected (64, 162), got {array.shape}")
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: contains NaN or infinite values")
        is_pre_aug = path.stem.endswith("_aug")
        if self.augment and (not is_pre_aug or self.robust_aug):
            rng = self._get_rng()
            if not is_pre_aug:
                array = augment_keypoints(array, rng, spatial_temporal=True, robust=False)
            if self.robust_aug:
                array = augment_keypoints(array, rng, spatial_temporal=False, robust=True)
        array = engineer_model_features(array)
        return torch.from_numpy(array), torch.tensor(label, dtype=torch.long)


def read_vocab(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        vocab = {str(word).upper(): idx for idx, word in enumerate(raw)}
    elif isinstance(raw, dict):
        vocab = {str(word).upper(): int(idx) for word, idx in raw.items()}
    else:
        raise ValueError("vocab.json must be a non-empty word list or word-to-ID object")
    if not vocab:
        raise ValueError("vocab.json must contain at least one class")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be contiguous integers from 0 to C-1")
    return vocab


def collect(folder: Path, vocab: dict[str, int]) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for word, label in vocab.items():
        class_dir = folder / word
        if class_dir.exists():
            items.extend((p, label) for p in sorted(class_dir.glob("*.npy")))
    if not items:
        raise ValueError(f"No class-named .npy files found under {folder}")
    return items


def stratified_split(items: list[tuple[Path, int]], val_fraction: float,
                     seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)
    train, val = [], []
    for group in by_class.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    if not val:
        raise ValueError("Need at least two clips in one or more classes to make validation split")
    return train, val


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == targets).sum().item())


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None,
              mixup_alpha: float = 0.1,
              ema_model: AveragedModel | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if training and labels.size(0) > 1 and mixup_alpha > 0.0:
                mix = float(np.random.beta(mixup_alpha, mixup_alpha))
                permutation = torch.randperm(labels.size(0), device=device)
                mixed_clips = mix * clips + (1.0 - mix) * clips[permutation]
                logits = model(mixed_clips)
                mixed_logits_target = criterion(logits, labels)
                mixed_logits_target_other = criterion(logits, labels[permutation])
                loss = mix * mixed_logits_target + (1.0 - mix) * mixed_logits_target_other
            else:
                logits = model(clips)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
            batch = labels.size(0)
            total_loss += float(loss.item()) * batch
            total_correct += accuracy(logits, labels)
            total += batch
    return total_loss / total, total_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Folder containing train/ and optional val/")
    parser.add_argument("--vocab", type=Path, default=Path("vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", choices=("cnn", "gru_cnn", "wide_cnn", "stgcn", "attn_cnn"),
                        default="cnn")
    parser.add_argument("--robust-aug", action=argparse.BooleanOptionalAction, default=True,
                        help="enable train-only frame/hand occlusion and coordinate jitter")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    vocab = read_vocab(args.vocab)
    train_dir = args.data / "train"
    val_dir = args.data / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    train_items = collect(train_dir, vocab)
    if val_dir.is_dir():
        val_items = collect(val_dir, vocab)
    else:
        train_items, val_items = stratified_split(train_items, args.val_fraction, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = NpyClips(train_items, augment=True, robust_aug=args.robust_aug,
                             seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(NpyClips(val_items, augment=False, seed=args.seed), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.seed))

    counts = torch.bincount(torch.tensor([y for _, y in train_items]), minlength=len(vocab)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights /= weights[present].mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    model = create_model(args.arch, len(vocab), width=args.width).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.995), use_buffers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = args.output / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        mixup_alpha = 0.1 if epoch <= 135 else 0.0
        train_dataset.set_epoch(epoch)
        train_dataset.robust_aug = args.robust_aug and epoch <= args.epochs - 20
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                          mixup_alpha=mixup_alpha, ema_model=ema_model)
        val_loss, val_acc = run_epoch(ema_model, val_loader, criterion, device, None)
        scheduler.step()
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
              f"train_top1={train_acc:.4f} val_loss={val_loss:.4f} val_top1={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": ema_model.module.state_dict(),
                "arch": args.arch,
                "num_classes": len(vocab), "width": args.width,
                "vocab": vocab, "best_val_top1": best_acc,
                "input_shape": [1, 64, INPUT_FEATURES],
            }, best_path)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    export_model = create_model(checkpoint.get("arch", "cnn"), checkpoint["num_classes"],
                                width=checkpoint.get("width", 128))
    export_model.load_state_dict(checkpoint["model_state_dict"])
    export_model.eval()
    dummy = torch.zeros(1, 64, INPUT_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        export_model, dummy, args.output / "model.onnx",
        input_names=["keypoints"], output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    print(f"Best validation Top-1: {best_acc:.4%}; wrote {best_path} and {args.output / 'model.onnx'}")
    if best_acc < 0.70:
        print("Target not reached. Review data quality, class balance, preprocessing parity, and training configuration.")


if __name__ == "__main__":
    main()
