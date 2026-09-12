from __future__ import annotations

import argparse
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from config import CONFIG


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8 volleyball jersey dataset.")
    parser.add_argument("--images-dir", default=str(CONFIG.paths.raw_images_dir), help="Directory containing raw images.")
    parser.add_argument("--labels-dir", default=str(CONFIG.paths.raw_labels_dir), help="Directory containing YOLO label text files.")
    parser.add_argument("--output-dir", default=str(CONFIG.paths.data_dir), help="Base directory for prepared dataset.")
    parser.add_argument("--class-names", nargs="+", default=list(CONFIG.dataset_classes), help="Dataset class names.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-only", action="store_true", help="Create the folder structure and dataset.yaml only.")
    return parser.parse_args()


def ensure_structure(base_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (base_dir / "processed" / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "processed" / "labels" / split).mkdir(parents=True, exist_ok=True)
    (base_dir / "raw" / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "raw" / "labels").mkdir(parents=True, exist_ok=True)
    (base_dir / "splits").mkdir(parents=True, exist_ok=True)


def paired_assets(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir() if images_dir.exists() else []):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[Path, Path]]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave room for a test split.")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)
    val_cutoff = train_cutoff + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_cutoff],
        "val": shuffled[train_cutoff:val_cutoff],
        "test": shuffled[val_cutoff:],
    }


def copy_split(base_dir: Path, split_name: str, assets: Iterable[tuple[Path, Path]]) -> None:
    image_target = base_dir / "processed" / "images" / split_name
    label_target = base_dir / "processed" / "labels" / split_name
    for directory in (image_target, label_target):
        for child in directory.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    for image_path, label_path in assets:
        shutil.copy2(image_path, image_target / image_path.name)
        shutil.copy2(label_path, label_target / label_path.name)


def write_dataset_yaml(base_dir: Path, class_names: list[str]) -> Path:
    dataset_yaml = base_dir / "dataset.yaml"
    payload = {
        "path": ".",
        "train": "processed/images/train",
        "val": "processed/images/val",
        "test": "processed/images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    ensure_structure(base_dir)
    dataset_yaml = write_dataset_yaml(base_dir, args.class_names)

    if args.bootstrap_only:
        print(f"Created folder structure and dataset config at {dataset_yaml}")
        return

    pairs = paired_assets(images_dir, labels_dir)
    if not pairs:
        print(f"No image/label pairs found in {images_dir} and {labels_dir}. Dataset config is ready at {dataset_yaml}")
        return

    split_mapping = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)
    for split_name, assets in split_mapping.items():
        copy_split(base_dir, split_name, assets)
    print({split_name: len(assets) for split_name, assets in split_mapping.items()})
    print(f"Prepared dataset config at {dataset_yaml}")


if __name__ == "__main__":
    main()
