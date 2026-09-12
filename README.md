# YOLOv8 Volleyball Jersey Tracking

A starter project for training and running a YOLOv8-based volleyball system that detects players and balls, tracks identities across frames, and classifies teams from jersey colors.

## Features

- YOLOv8 training workflow for custom volleyball datasets
- Real-time and batch video inference with Ultralytics tracking
- Ball detection alongside player detection
- Jersey color extraction and team classification utilities
- Dataset preparation helper for YOLO-format image/label pairs
- Centralized configuration for paths, thresholds, model settings, and team colors
- Example notebook covering setup, preparation, training, and inference

## Project Structure

```text
.
├── config.py
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── labels/
│   ├── processed/
│   │   ├── images/
│   │   └── labels/
│   └── splits/
├── models/
├── notebooks/
│   └── volleyball_jersey_tracking_workflow.ipynb
├── outputs/
│   ├── runs/
│   └── tracks/
├── requirements.txt
├── scripts/
│   ├── data_preparation.py
│   ├── inference.py
│   └── train.py
├── tests/
│   └── test_utils.py
└── volleyball_tracking/
    └── utils/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to change:

- dataset/model/output paths
- training defaults such as image size, batch size, and epochs
- inference confidence and IoU thresholds
- tracker selection (`bytetrack.yaml` by default)
- jersey color prototypes used for team classification

Default dataset classes are `player` and `ball`. Update `CONFIG.dataset_classes` if your dataset uses different labels.

## Data Preparation

Place YOLO-format image/label pairs in:

- `data/raw/images`
- `data/raw/labels`

Bootstrap the dataset structure and generate `data/dataset.yaml`:

```bash
python -m scripts.data_preparation --bootstrap-only
```

Prepare train/val/test splits from paired files:

```bash
python -m scripts.data_preparation --images-dir data/raw/images --labels-dir data/raw/labels
```

## Training

Fine-tune a YOLOv8 model on your volleyball dataset:

```bash
python -m scripts.train --data data/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 1280 --batch 16
```

Training outputs are written under `outputs/runs`.

## Inference and Tracking

### Webcam / Real-Time

```bash
python -m scripts.inference --source 0 --model models/best.pt --display
```

### Batch Video File

```bash
python -m scripts.inference --source path/to/match.mp4 --model models/best.pt --save-output
```

### Batch Directory Processing

```bash
python -m scripts.inference --source path/to/videos --model models/best.pt --save-output
```

The inference pipeline:

1. runs YOLOv8 detection and ByteTrack-based ID assignment
2. keeps ball and player detections in the same pass
3. crops the jersey torso region for each tracked player
4. compares dominant jersey color against configured team colors
5. writes annotated videos into `outputs/tracks` when `--save-output` is enabled

## Notebook

Open `notebooks/volleyball_jersey_tracking_workflow.ipynb` in Jupyter to walk through:

- dependency installation
- dataset setup
- training commands
- inference examples
- team-color customization

## Validation

This repository includes lightweight utility tests:

```bash
python -m unittest discover -s tests
```

You can also verify the CLI entry points without YOLO weights by using `--help`:

```bash
python -m scripts.train --help
python -m scripts.data_preparation --help
python -m scripts.inference --help
```
