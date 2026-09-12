from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from volleyball_tracking.utils.color import dominant_color_bgr, jersey_region


def classify_team_by_color(
    color_bgr: Sequence[int],
    team_colors_bgr: Mapping[str, Sequence[int]],
) -> str:
    if not team_colors_bgr:
        return "unknown"

    sample = np.asarray(color_bgr, dtype=np.float32)
    distances = {
        team_name: float(np.linalg.norm(sample - np.asarray(reference, dtype=np.float32)))
        for team_name, reference in team_colors_bgr.items()
    }
    return min(distances, key=distances.get)


def classify_team_from_bbox(
    frame: np.ndarray,
    box: Sequence[float],
    team_colors_bgr: Mapping[str, Sequence[int]],
) -> tuple[str, tuple[int, int, int]]:
    region = jersey_region(frame, box)
    color = dominant_color_bgr(region)
    return classify_team_by_color(color, team_colors_bgr), color
