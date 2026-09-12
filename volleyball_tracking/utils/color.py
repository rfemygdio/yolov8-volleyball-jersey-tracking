from __future__ import annotations

from typing import Sequence

import numpy as np


def jersey_region(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    top = y1 + int(box_height * 0.15)
    bottom = y1 + int(box_height * 0.6)
    left = x1 + int(box_width * 0.2)
    right = x1 + int(box_width * 0.8)
    return frame[top:max(top + 1, bottom), left:max(left + 1, right)]


def dominant_color_bgr(image: np.ndarray) -> tuple[int, int, int]:
    if image.size == 0:
        return (0, 0, 0)

    pixels = image.reshape(-1, image.shape[-1]).astype(np.float32)
    brightness = pixels.mean(axis=1)
    informative = pixels[(brightness > 30) & (brightness < 245)]
    if informative.size == 0:
        informative = pixels
    color = informative.mean(axis=0)
    return tuple(int(channel) for channel in np.clip(color, 0, 255))
