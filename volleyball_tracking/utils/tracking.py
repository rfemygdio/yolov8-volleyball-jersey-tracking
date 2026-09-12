from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import glob as glob_module


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class TrackAnnotation:
    track_id: int | None
    class_name: str
    confidence: float
    team_name: str | None
    box: tuple[int, int, int, int]


def clamp_box(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def resolve_video_sources(source: str) -> list[int | str]:
    candidate = source.strip()
    if candidate.isdigit():
        return [int(candidate)]
    if candidate.startswith(("rtsp://", "http://", "https://")):
        return [candidate]

    path = Path(candidate)
    if path.is_dir():
        matches = [str(file) for file in sorted(path.iterdir()) if file.suffix.lower() in VIDEO_SUFFIXES]
        return matches

    if any(token in candidate for token in "*?[]"):
        matches = [match for match in sorted(glob_module.glob(candidate)) if Path(match).suffix.lower() in VIDEO_SUFFIXES]
        return matches

    return [candidate]
