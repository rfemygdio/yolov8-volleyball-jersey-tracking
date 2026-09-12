from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from volleyball_tracking.utils.classification import classify_team_by_color, classify_team_from_bbox
from volleyball_tracking.utils.tracking import resolve_video_sources


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


if __name__ == "__main__":
    unittest.main()
