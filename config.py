from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_images_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "images")
    raw_labels_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "labels")
    processed_images_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "images")
    processed_labels_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "labels")
    splits_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "splits")
    dataset_yaml: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "dataset.yaml")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "runs")
    tracks_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "tracks")


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 1280
    batch: int = 16
    device: str = "cpu"
    patience: int = 20
    project_name: str = "jersey_tracking"
    run_name: str = "volleyball_yolov8"


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    line_thickness: int = 2
    player_classes: tuple[str, ...] = ("player", "person")
    ball_classes: tuple[str, ...] = ("ball", "sports ball", "volleyball")
    team_colors_bgr: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "team_a": (220, 60, 60),
            "team_b": (60, 140, 220),
            "official": (40, 200, 240),
        }
    )


@dataclass(frozen=True)
class AppConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dataset_classes: tuple[str, ...] = ("player", "ball")


CONFIG = AppConfig()
