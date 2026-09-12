from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data_preparation import split_pairs
from volleyball_tracking.utils.classification import classify_team_by_color, classify_team_from_bbox
from volleyball_tracking.utils.tracking import clamp_box, resolve_video_sources


class UtilityTests(unittest.TestCase):
    def test_classify_team_by_color_uses_nearest_reference(self) -> None:
        team = classify_team_by_color((210, 70, 70), {"red": (220, 60, 60), "blue": (60, 120, 220)})
        self.assertEqual(team, "red")

    def test_classify_team_from_bbox_uses_jersey_region(self) -> None:
        frame = np.zeros((120, 60, 3), dtype=np.uint8)
        frame[20:70, 15:45] = (220, 60, 60)
        team, color = classify_team_from_bbox(frame, (10, 10, 50, 100), {"red": (220, 60, 60), "blue": (60, 120, 220)})
        self.assertEqual(team, "red")
        self.assertGreater(color[0], color[1])
        self.assertGreater(color[0], color[2])

    def test_resolve_video_sources_handles_directory_and_glob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "match.mp4").write_bytes(b"0")
            (base / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual(resolve_video_sources(str(base)), [str(base / "match.mp4")])
            self.assertEqual(resolve_video_sources(str(base / "*.mp4")), [str(base / "match.mp4")])
            self.assertEqual(resolve_video_sources("0"), [0])
            self.assertEqual(resolve_video_sources("rtsp://example.test/stream"), ["rtsp://example.test/stream"])

    def test_split_pairs_validates_ratios(self) -> None:
        with self.assertRaises(ValueError):
            split_pairs([], train_ratio=0.8, val_ratio=0.2, seed=42)

    def test_split_pairs_returns_expected_partition_sizes(self) -> None:
        pairs = [(Path(f"image_{index}.jpg"), Path(f"image_{index}.txt")) for index in range(10)]
        split_mapping = split_pairs(pairs, train_ratio=0.6, val_ratio=0.2, seed=42)
        self.assertEqual(len(split_mapping["train"]), 6)
        self.assertEqual(len(split_mapping["val"]), 2)
        self.assertEqual(len(split_mapping["test"]), 2)

    def test_clamp_box_limits_out_of_bounds_fractional_values(self) -> None:
        self.assertEqual(clamp_box((-5.7, 3.2, 120.8, 50.9), width=100, height=40), (0, 3, 100, 40))


if __name__ == "__main__":
    unittest.main()
