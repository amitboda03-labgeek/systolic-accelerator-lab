#!/usr/bin/env python3
"""Run a quantized MNIST-style fault-tolerance experiment.

The accelerator computes an 8x8 matrix-vector product, so the experiment uses
a tiny quantized convolutional feature extractor followed by a trained
eight-class head.  The 144 extracted features are tiled into eighteen 8x8
matrix operations. It quantizes the head to signed int8, injects permanent PE
faults, and compares unprotected inference with the one-spare-column repair
policy used by the RTL.

Use ``--synthetic`` for a deterministic smoke test.  Use ``--download`` to
fetch the official MNIST IDX files, or pass ``--images`` and ``--labels`` for
an existing local copy.  Results are JSON/CSV/SVG artifacts and are not a
substitute for a gate-level power run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np


CLASSES = 8
FEATURES = 144
TILES = 18
ROWS = 8
COLS = 8
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def downsample(images: np.ndarray) -> np.ndarray:
    """Average-pool 28x28 images to 8x8 and quantize pixels to int8."""

    result = np.zeros((len(images), 8, 8), dtype=np.uint8)
    for y in range(8):
        y0 = round(y * 28 / 8)
        y1 = round((y + 1) * 28 / 8)
        for x in range(8):
            x0 = round(x * 28 / 8)
            x1 = round((x + 1) * 28 / 8)
            result[:, y, x] = np.rint(images[:, y0:y1, x0:x1].mean(axis=(1, 2))).astype(np.uint8)
    return np.rint(result.astype(np.float32) * (127.0 / 255.0)).astype(np.int8)


def convolutional_features(images: np.ndarray) -> np.ndarray:
    """Apply four fixed 3x3 edge kernels, ReLU, and int8 requantization."""

    kernels = np.asarray(
        [
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ],
        dtype=np.int32,
    )
    maps = np.zeros((len(images), 4, 6, 6), dtype=np.int32)
    integer_images = images.astype(np.int32)
    for y in range(6):
        for x in range(6):
            window = integer_images[:, y : y + 3, x : x + 3]
            maps[:, :, y, x] = np.einsum("nrc,krc->nk", window, kernels)
    # Keep the feature dynamic range compatible with signed int8 operands.
    return np.clip(np.rint(np.maximum(maps, 0) / 24.0), 0, 127).astype(np.int8).reshape(len(images), FEATURES)


def read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic, = struct.unpack(">I", handle.read(4))
        dimensions = magic & 0xFF
        if dimensions == 1:
            count, = struct.unpack(">I", handle.read(4))
            return np.frombuffer(handle.read(), dtype=np.uint8, count=count)
        if dimensions == 3:
            count, rows, columns = struct.unpack(">III", handle.read(12))
            raw = np.frombuffer(handle.read(), dtype=np.uint8, count=count * rows * columns)
            return raw.reshape(count, rows, columns)
    raise ValueError(f"unsupported IDX file: {path}")


def download_file(directory: Path, name: str) -> Path:
    destination = directory / name
    if not destination.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}")
        urllib.request.urlretrieve(MNIST_URL + name, destination)
    return destination


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.synthetic:
        return synthetic_dataset(args.limit, args.seed)
    if args.download:
        directory = Path(args.download_dir)
        image_path = download_file(directory, "train-images-idx3-ubyte.gz")
        label_path = download_file(directory, "train-labels-idx1-ubyte.gz")
    elif args.images and args.labels:
        image_path, label_path = Path(args.images), Path(args.labels)
    else:
        raise SystemExit("choose --synthetic, --download, or both --images and --labels")
    images = read_idx(image_path)
    labels = read_idx(label_path)
    mask = labels < CLASSES
    images = images[mask][: args.limit]
    labels = labels[mask][: args.limit]
    if len(images) < CLASSES * 2:
        raise SystemExit("dataset needs at least two examples for each digit 0-7")
    return downsample(images), labels.astype(np.int64)


def synthetic_dataset(limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Compact 8x8 glyph prototypes make the smoke test deterministic while
    # retaining the same quantized feature shape as the MNIST path.
    glyphs = np.array(
        [
            ["01111110", "11000011", "11011011", "11011011", "11011011", "11000011", "01111110", "00000000"],
            ["00011000", "00111000", "01111000", "00011000", "00011000", "00011000", "01111110", "00000000"],
            ["01111110", "11000011", "00000011", "00001110", "00111000", "01100000", "11111111", "00000000"],
            ["01111110", "11000011", "00000011", "00111110", "00000011", "11000011", "01111110", "00000000"],
            ["00001110", "00011110", "00110110", "01100110", "11111111", "00000110", "00000110", "00000000"],
            ["11111111", "11000000", "11000000", "01111110", "00000011", "11000011", "01111110", "00000000"],
            ["00111110", "01100000", "11000000", "11111110", "11000011", "11000011", "01111110", "00000000"],
            ["11111111", "00000011", "00000110", "00001100", "00011000", "00011000", "00011000", "00000000"],
        ],
        dtype="U8",
    )
    per_class = max(2, limit // CLASSES)
    images = []
    labels = []
    for label, glyph in enumerate(glyphs):
        base = np.array([[int(bit) for bit in row] for row in glyph], dtype=np.uint8) * 255
        for _ in range(per_class):
            noisy = base.astype(np.int16) + rng.normal(0, 24, base.shape)
            images.append(np.clip(noisy, 0, 255).astype(np.uint8))
            labels.append(label)
    order = rng.permutation(len(images))[:limit]
    quantized = np.rint(np.stack(images)[order].astype(np.float32) * (127.0 / 255.0)).astype(np.int8)
    return quantized, np.asarray(labels, dtype=np.int64)[order]


def fit_quantized_classifier(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = features.reshape(len(features), FEATURES).astype(np.float32)
    targets = np.eye(CLASSES, dtype=np.float32)[labels]
    floating = np.linalg.pinv(x) @ targets
    peak = max(float(np.max(np.abs(floating))), 1e-9)
    return np.clip(np.rint(floating * (120.0 / peak)), -128, 127).astype(np.int8)


def make_tiles(weights: np.ndarray) -> np.ndarray:
    return weights.reshape(TILES, ROWS, COLS)


def signed_int8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def mutate(value: int, mode: str) -> int:
    if mode == "stuck0":
        return 0
    if mode == "stuck1":
        return 127
    if mode == "flip":
        return signed_int8(value ^ 0x01)
    raise ValueError(f"unknown fault mode: {mode}")


def apply_faults(weights: np.ndarray, faults: Iterable[tuple[int, int, int, str]]) -> np.ndarray:
    result = weights.copy()
    for tile, row, col, mode in faults:
        result[tile, row, col] = mutate(int(result[tile, row, col]), mode)
    return result


def score(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    tiled = features.reshape(len(features), TILES, ROWS).astype(np.int32)
    return np.einsum("ntr,trc->nc", tiled, weights.astype(np.int32))


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def protected_scores(features: np.ndarray, original: np.ndarray, faults: list[tuple[int, int, int, str]]) -> tuple[np.ndarray, bool, bool]:
    if not faults:
        return score(features, original), False, False
    # The checksum column identifies a faulty logical column.  The spare can
    # repair exactly one such column; later faults force that output into the
    # RTL's explicit degraded state rather than allowing corrupted data out.
    repaired = apply_faults(original, faults)
    first = faults[0]
    repaired[first[0], first[1], first[2]] = original[first[0], first[1], first[2]]
    result = score(features, repaired)
    degraded = len(faults) > 1
    if degraded:
        for tile, row, col, _mode in faults[1:]:
            result[:, col] = 0
    return result, True, degraded


def choose_faults(count: int, mode: str, rng: np.random.Generator) -> list[tuple[int, int, int, str]]:
    locations = [(tile, row, col) for tile in range(TILES) for row in range(ROWS) for col in range(COLS)]
    rng.shuffle(locations)
    return [(tile, row, col, mode) for tile, row, col in locations[:count]]


def svg_plot(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 920, 500
    left, top, plot_w, plot_h = 80, 60, 790, 350
    counts = sorted({int(row["fault_count"]) for row in rows})
    max_count = max(counts or [1])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<text x="32" y="32" fill="#f8fafc" font-size="20">Quantized MNIST fault-tolerance curve (mean across fault modes)</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8"/>',
        '<text x="460" y="455" fill="#cbd5e1" text-anchor="middle">permanent logical-column faults</text>',
        '<text x="20" y="240" fill="#cbd5e1" transform="rotate(-90 20 240)" text-anchor="middle">accuracy</text>',
    ]
    colors = {"unprotected": "#f97316", "protected": "#22c55e"}
    for policy, color in colors.items():
        points = []
        for count in counts:
            matches = [
                row for row in rows
                if row["policy"] == policy and int(row["fault_count"]) == count
            ]
            mean_accuracy = float(np.mean([float(row["accuracy"]) for row in matches]))
            x = left + (count / max_count if max_count else 0) * plot_w
            y = top + (1 - mean_accuracy) * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for point in points:
            x, y = point.split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    lines.extend(
        [
            '<circle cx="100" cy="475" r="5" fill="#f97316"/><text x="112" y="479" fill="#e2e8f0">unprotected</text>',
            '<circle cx="245" cy="475" r="5" fill="#22c55e"/><text x="257" y="479" fill="#e2e8f0">one-spare ABFT repair</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> int:
    images, labels = load_dataset(args)
    if len(images) < CLASSES * 2:
        raise SystemExit("not enough samples")
    features = convolutional_features(images)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(features))
    split = max(CLASSES, int(len(features) * 0.7))
    train, test = order[:split], order[split:]
    if len(test) == 0:
        test = train
    weights = make_tiles(fit_quantized_classifier(features[train], labels[train]))
    baseline = accuracy(score(features[test], weights), labels[test])
    rows: list[dict[str, object]] = []
    for mode in ("stuck0", "stuck1", "flip"):
        for fault_count in range(args.max_faults + 1):
            faults = choose_faults(fault_count, mode, rng)
            unprotected = accuracy(score(features[test], apply_faults(weights, faults)), labels[test])
            protected, detected, degraded = protected_scores(features[test], weights, faults)
            rows.extend(
                [
                    {
                        "mode": mode,
                        "fault_count": fault_count,
                        "policy": "unprotected",
                        "accuracy": unprotected,
                        "detected": False,
                        "degraded": False,
                    },
                    {
                        "mode": mode,
                        "fault_count": fault_count,
                        "policy": "protected",
                        "accuracy": accuracy(protected, labels[test]),
                        "detected": detected,
                        "degraded": degraded,
                    },
                ]
            )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "synthetic-8x8" if args.synthetic else "mnist",
        "samples": int(len(features)),
        "test_samples": int(len(test)),
        "classes": list(range(CLASSES)),
        "network": "four-filter 3x3 ReLU feature extractor plus trained int8 linear head",
        "feature_shape": [18, 8],
        "quantization": "signed int8 weights and activations",
        "baseline_accuracy": baseline,
        "fault_model": "one permanent PE weight fault per selected location",
        "repair_model": "ABFT checksum column plus one spare logical output column",
        "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "fault_count", "policy", "accuracy", "detected", "degraded"])
        writer.writeheader()
        writer.writerows(rows)
    svg_plot(rows, args.output / "fault_tolerance.svg")
    print(f"dataset={summary['dataset']} samples={len(features)} baseline_accuracy={baseline:.3f}")
    print(f"wrote {args.output / 'results.json'} and {args.output / 'fault_tolerance.svg'}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--synthetic", action="store_true")
    result.add_argument("--download", action="store_true")
    result.add_argument("--download-dir", default="datasets/mnist")
    result.add_argument("--images")
    result.add_argument("--labels")
    result.add_argument("--limit", type=int, default=512)
    result.add_argument("--max-faults", type=int, default=4)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--output", type=Path, default=Path("build/fault-experiment"))
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
