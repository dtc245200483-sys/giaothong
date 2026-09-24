"""Split an auto-labeled dataset (images/ + labels/) into train/val for Ultralytics.

Usage:
    python split_dataset.py --data <dir> [--val-frac 0.15]

Result layout:
    <data>/images/train, <data>/images/val
    <data>/labels/train, <data>/labels/val
"""
import argparse
import os
import random
import shutil
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(args.data, "images")
    lbl_dir = os.path.join(args.data, "labels")
    imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    imgs = [f for f in imgs if os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))]
    if not imgs:
        print("no labeled images found")
        return

    random.Random(args.seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * args.val_frac))
    val = set(imgs[:n_val])
    train = [f for f in imgs if f not in val]

    for sub in ("train", "val"):
        os.makedirs(os.path.join(img_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir, sub), exist_ok=True)

    for f in imgs:
        sub = "val" if f in val else "train"
        stem = os.path.splitext(f)[0]
        shutil.move(os.path.join(img_dir, f), os.path.join(img_dir, sub, f))
        shutil.move(os.path.join(lbl_dir, stem + ".txt"), os.path.join(lbl_dir, sub, stem + ".txt"))

    freq = Counter()
    for sub in ("train", "val"):
        for f in os.listdir(os.path.join(lbl_dir, sub)):
            with open(os.path.join(lbl_dir, sub, f)) as fh:
                for line in fh:
                    freq[(sub, int(line.split()[0]))] += 1

    print(f"train={len(train)} val={len(val)}")
    print("class frequencies (train/val):")
    for sub in ("train", "val"):
        print(" ", sub, dict((k, v) for (s, k), v in freq.items() if s == sub))


if __name__ == "__main__":
    main()
