from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLOv8 model for volleyball jersey tracking.")
    parser.add_argument("--data", default=str(CONFIG.paths.dataset_yaml), help="Path to YOLO dataset YAML file.")
    parser.add_argument("--model", default=CONFIG.training.model, help="YOLOv8 model checkpoint or model name.")
    parser.add_argument("--epochs", type=int, default=CONFIG.training.epochs)
    parser.add_argument("--imgsz", type=int, default=CONFIG.training.imgsz)
    parser.add_argument("--batch", type=int, default=CONFIG.training.batch)
    parser.add_argument("--device", default=CONFIG.training.device)
    parser.add_argument("--patience", type=int, default=CONFIG.training.patience)
    parser.add_argument("--project", default=str(CONFIG.paths.runs_dir))
    parser.add_argument("--name", default=CONFIG.training.run_name)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_yaml = Path(args.data)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    print(results)


if __name__ == "__main__":
    main()
