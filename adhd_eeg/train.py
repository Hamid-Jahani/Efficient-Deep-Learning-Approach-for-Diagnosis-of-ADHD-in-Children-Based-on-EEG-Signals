"""Command-line entry point for running the ADHD EEG classification pipeline
on a local machine (as an alternative to the Colab notebook).

Example
-------
    python -m adhd_eeg.train --data-dir /path/to/adhd_dataset --model resnet

If the dataset is not available yet, this fails immediately with a message
explaining how to obtain/place it (see README.md, section "Data").
"""
from __future__ import annotations

import argparse
import sys

from adhd_eeg.config import DEFAULT_SEGMENT_SIZE, IMAGE_SHAPE, resolve_data_dir

MODEL_NAMES = ("cnn", "resnet")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the EEG dataset directory (defaults to $ADHD_DATA_DIR or ./data/adhd_dataset).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_NAMES),
        default="resnet",
        help="Which model to train (default: resnet).",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds (default: 10).")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs per fold (default: 100).")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128).")
    parser.add_argument(
        "--segment-size",
        type=int,
        default=DEFAULT_SEGMENT_SIZE,
        help="Epochs per subject-level segment (default: 20, matches the paper).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        data_dir = resolve_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Imported lazily so `--help` and error paths above don't require heavy
    # deep-learning/EEG dependencies to be installed.
    from adhd_eeg import models
    from adhd_eeg.cv import cross_validate
    from adhd_eeg.features import build_dataset

    print(f"Loading and transforming EEG recordings from '{data_dir}' ...")
    images, labels, subjects, segments = build_dataset(data_dir, segment_size=args.segment_size)
    print(f"Built {len(labels)} epoch images from {len(set(subjects))} subjects.")

    model_builders = {"cnn": models.cnn_model, "resnet": models.resnet_model}
    model_fn = model_builders[args.model]
    results = cross_validate(
        images,
        labels,
        subjects,
        segments,
        model_fn,
        input_shape=IMAGE_SHAPE,
        segment_size=args.segment_size,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    accuracies = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    print(f"\n{args.model} — {args.folds}-fold subject-level results:")
    print(f"  accuracy: mean={sum(accuracies) / len(accuracies):.4f}")
    print(f"  f1:       mean={sum(f1s) / len(f1s):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
