#!/usr/bin/env python3
"""
train.py
────────
Command-line training script.

Usage
─────
  python train.py --dataset dataset_imgs/
  python train.py --dataset dataset_imgs/ --max-per-class 150 --no-loo
  python train.py --dataset dataset_imgs/ --k 7

After training, models/knn_model.pkl and models/knn_meta.json are written.
The app will automatically load the trained model on next run.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train the LeafGuard k-NN classifier."
    )
    parser.add_argument("--dataset",       required=True,
                        help="Path to dataset root folder (sub-folders = classes)")
    parser.add_argument("--max-per-class", type=int, default=200,
                        help="Max images per class to load (default: 200)")
    parser.add_argument("--k",             type=int, default=5,
                        help="k for k-NN (default: 5)")
    parser.add_argument("--no-loo",        action="store_true",
                        help="Skip Leave-One-Out cross-validation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # import after arg parsing so errors surface cleanly
    from pipeline.dataset import load_dataset
    from pipeline.classifier import train_knn, loo_evaluate
    from collections import Counter
    import numpy as np

    print(f"\n{'='*55}")
    print(f"  LeafGuard — Training Pipeline")
    print(f"{'='*55}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  Max/class: {args.max_per_class}")
    print(f"  k        : {args.k}")
    print(f"  LOO eval : {not args.no_loo}")
    print(f"{'='*55}\n")

    print("Loading dataset and extracting features…")
    X, y, paths = load_dataset(
        str(dataset_path),
        max_per_class=args.max_per_class,
        verbose=True,
    )

    dist = Counter(y)
    print(f"\nClass distribution:")
    for cls, cnt in sorted(dist.items()):
        bar = "█" * (cnt // max(1, max(dist.values()) // 30))
        print(f"  {cls:<28} {cnt:>4}  {bar}")
    print(f"\n  Total: {len(y)} samples, {X.shape[1]} features\n")

    print("Training k-NN classifier…")
    train_info = train_knn(X, y, k=args.k)
    print(f"  ✓ Training accuracy : {train_info['train_accuracy']*100:.2f}%")
    print(f"  ✓ Model saved to    : models/knn_model.pkl")

    if not args.no_loo and len(y) >= 4:
        print(f"\nRunning Leave-One-Out cross-validation ({len(y)} folds)…")
        loo = loo_evaluate(X, y)

        print(f"\n  LOO Accuracy: {loo['loo_accuracy']*100:.2f}%")
        print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
        labels = loo["labels"]
        cm     = loo["confusion_matrix"]

        col_w = 14
        header = f"{'':>28}" + "".join(f"{l[:col_w-2]:>{col_w}}" for l in labels)
        print(header)
        sep = " " * 28 + "-" * (col_w * len(labels))
        print(sep)
        for row_label, row in zip(labels, cm):
            print(f"{row_label[:28]:>28}" + "".join(f"{v:>{col_w}}" for v in row))

        print(f"\n  Per-class breakdown:")
        rep = loo["classification_report"]
        for cls in labels:
            if cls in rep:
                r = rep[cls]
                print(f"  {cls:<28}  "
                      f"P={r['precision']:.2f}  "
                      f"R={r['recall']:.2f}  "
                      f"F1={r['f1-score']:.2f}  "
                      f"n={r['support']}")

    print(f"\n{'='*55}")
    print("  Training complete. Run the app with:")
    print("    streamlit run app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
