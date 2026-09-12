from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import CONFIG
from volleyball_tracking.utils.classification import classify_team_from_bbox
from volleyball_tracking.utils.tracking import TrackAnnotation, clamp_box, resolve_video_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for volleyball jersey tracking.")
    parser.add_argument("--source", default="0", help="Webcam index, video path, directory, glob, or stream URL.")
    parser.add_argument("--model", default=CONFIG.training.model, help="Model checkpoint or Ultralytics model name.")
    parser.add_argument("--tracker", default=CONFIG.tracking.tracker, help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=CONFIG.tracking.conf_threshold)
    parser.add_argument("--iou", type=float, default=CONFIG.tracking.iou_threshold)
    parser.add_argument("--device", default=CONFIG.training.device)
    parser.add_argument("--save-output", action="store_true", help="Save annotated video(s) in outputs/tracks.")
    parser.add_argument("--display", action="store_true", help="Display live annotated frames.")
    return parser.parse_args()


def annotate_frame(frame: np.ndarray, result) -> tuple[np.ndarray, list[TrackAnnotation]]:
    annotations: list[TrackAnnotation] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return frame, annotations

    import cv2

    names = result.names or {}
    for box in boxes:
        confidence = float(box.conf.item()) if box.conf is not None else 0.0
        class_id = int(box.cls.item()) if box.cls is not None else -1
        class_name = names.get(class_id, str(class_id))
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = clamp_box(xyxy, frame.shape[1], frame.shape[0])
        track_id = int(box.id.item()) if box.id is not None else None
        team_name = None
        color = (0, 255, 0)
        if class_name in CONFIG.tracking.player_classes:
            team_name, color = classify_team_from_bbox(frame, (x1, y1, x2, y2), CONFIG.tracking.team_colors_bgr)
        elif class_name in CONFIG.tracking.ball_classes:
            color = (0, 255, 255)

        label_parts = [class_name]
        if track_id is not None:
            label_parts.append(f"#{track_id}")
        if team_name:
            label_parts.append(team_name)
        label_parts.append(f"{confidence:.2f}")
        label = " ".join(label_parts)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, CONFIG.tracking.line_thickness)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        annotations.append(TrackAnnotation(track_id, class_name, confidence, team_name, (x1, y1, x2, y2)))
    return frame, annotations


def resolve_output_fps(model) -> float:
    dataset = getattr(getattr(model, "predictor", None), "dataset", None)
    fps = getattr(dataset, "fps", None)
    if isinstance(fps, (list, tuple)):
        fps = next((value for value in fps if value and value > 0), None)
    return fps if fps and fps > 0 else 30.0


def build_writer(source_name: str, frame_shape: tuple[int, int, int], fps: float):
    import cv2

    CONFIG.paths.tracks_dir.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.paths.tracks_dir / f"{source_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (frame_shape[1], frame_shape[0])), output_path


def process_source(model, source, args: argparse.Namespace) -> None:
    import cv2

    stream = model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        persist=True,
        verbose=False,
    )

    writer = None
    output_path = None
    source_name = Path(str(source)).stem or f"camera_{source}"
    window_open = False
    output_fps = resolve_output_fps(model) if args.save_output else None
    for result in stream:
        frame = result.orig_img.copy()
        annotated_frame, annotations = annotate_frame(frame, result)
        if args.save_output and writer is None:
            writer, output_path = build_writer(source_name, annotated_frame.shape, output_fps or 30.0)
        if writer is not None:
            writer.write(annotated_frame)
        if args.display:
            cv2.imshow(source_name, annotated_frame)
            window_open = True
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(source_name)
                window_open = False
                break
        if annotations:
            summary = ", ".join(
                f"{annotation.class_name}:{annotation.track_id if annotation.track_id is not None else 'na'}"
                for annotation in annotations
            )
            print(f"{source_name}: {summary}")

    if writer is not None:
        writer.release()
        print(f"Saved annotated output to {output_path}")
    if args.display and window_open:
        cv2.destroyWindow(source_name)


def main() -> None:
    args = parse_args()
    sources = resolve_video_sources(args.source)
    if not sources:
        raise FileNotFoundError(f"No supported video sources found for: {args.source}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    for source in sources:
        process_source(model, source, args)


if __name__ == "__main__":
    main()
