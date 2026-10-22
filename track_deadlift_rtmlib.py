"""RTMLib motion tracker for a constant-frame-rate, single-lifter video.

Install in your PyCharm environment:
    python -m pip install --upgrade rtmlib onnxruntime numpy

Put this file beside deadlift.mp4. RTMLib downloads its models on first run.
The output video has no audio. Scores are not MediaPipe visibility scores.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_DIR / "deadlift.mp4"
OUTPUT_VIDEO_PATH = PROJECT_DIR / "deadlift_rtmlib.mp4"
OUTPUT_CSV_PATH = PROJECT_DIR / "landmarks_rtmlib.csv"

MODE = "lightweight"  # performance, balanced, lightweight
SMOOTHING_ALPHA = 0.35  # Lower values smooth more; 1.0 disables smoothing.
MIN_SCORE = 0.5  # RTMLib keypoint score; not equivalent to MediaPipe visibility.

# Map COCO-WholeBody landmarks into the same IDs used by our MediaPipe drawing.
# The big toe approximates MediaPipe's foot-index landmark; locations differ.
# (MediaPipe-style ID, COCO-WholeBody ID, name)
LANDMARK_MAP = [
    (11, 5, "left_shoulder"), (12, 6, "right_shoulder"),
    (13, 7, "left_elbow"), (14, 8, "right_elbow"),
    (15, 9, "left_wrist"), (16, 10, "right_wrist"),
    (23, 11, "left_hip"), (24, 12, "right_hip"),
    (25, 13, "left_knee"), (26, 14, "right_knee"),
    (27, 15, "left_ankle"), (28, 16, "right_ankle"),
    (29, 19, "left_heel"), (30, 22, "right_heel"),
    (31, 17, "left_big_toe"), (32, 20, "right_big_toe"),
]

CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (31, 27),
    (28, 30), (30, 32), (32, 28),
]

BONE_COLOR = (0, 255, 0)
SPINE_COLOR = (0, 255, 255)
JOINT_COLOR = (0, 0, 255)


def draw_skeleton(frame, points):
    def line(a, b, color=BONE_COLOR):
        cv2.line(frame, a, b, color, 3, cv2.LINE_AA)

    for start, end in CONNECTIONS:
        if start in points and end in points:
            line(points[start], points[end])

    centers = []
    for left, right in [(11, 12), (23, 24)]:
        center = None
        if left in points and right in points:
            center = tuple(round((a + b) / 2)
                           for a, b in zip(points[left], points[right]))
            line(center, points[left])
            line(center, points[right])
        centers.append(center)

    if all(center is not None for center in centers):
        line(centers[0], centers[1], SPINE_COLOR)

    for point in points.values():
        cv2.circle(frame, point, 4, JOINT_COLOR, -1, cv2.LINE_AA)


def select_person(keypoints, scores):
    """Select the largest visible body; assumes one prominent lifter.

    This is not identity tracking. Crop the video if other people compete
    for selection. Face/hand points are excluded from the size calculation.
    """
    best_index, best_area = None, -1
    for index in range(len(keypoints)):
        body = keypoints[index, :17]
        valid = (scores[index, :17] >= MIN_SCORE)
        valid &= np.isfinite(body).all(axis=1)
        if np.count_nonzero(valid) < 4:
            continue
        extent = np.ptp(body[valid], axis=0)
        area = float(extent[0] * extent[1])
        if area > best_area:
            best_index, best_area = index, area
    return best_index


def smooth_points(raw_points, previous):
    # Absent landmarks are dropped. A returning joint starts at its new position.
    return {
        joint: SMOOTHING_ALPHA * current
        + (1 - SMOOTHING_ALPHA) * previous.get(joint, current)
        for joint, current in raw_points.items()
    }


def make_writer(path, fps, size):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create video: {path}")
    return writer


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not 0 < SMOOTHING_ALPHA <= 1:
        raise ValueError("SMOOTHING_ALPHA must be greater than 0 and at most 1.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    frame_number = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {VIDEO_PATH}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Cannot read the video's frame rate.")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Video contains no readable frames.")
        height, width = frame.shape[:2]

        print("Loading RTMLib models (first run downloads model files)...")
        model = Wholebody(to_openpose=False, mode=MODE,
                          backend="onnxruntime", device="cpu")
        writer = make_writer(OUTPUT_VIDEO_PATH, fps, (width, height))
        smoothed = {}
        with OUTPUT_CSV_PATH.open("w", newline="") as csv_file:
            table = csv.writer(csv_file)
            table.writerow([
                "frame", "time_s", "pose_detected", "landmark_id",
                "rtmlib_id", "name", "x_normalized", "y_normalized",
                "score", "accepted", "smoothed_x_normalized",
                "smoothed_y_normalized",
            ])
            while ok:
                # RTMLib takes OpenCV BGR frames directly; do not convert to RGB.
                keypoints, scores = model(frame)
                keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                person = select_person(keypoints, scores)
                raw_points = {}

                if person is not None:
                    if keypoints.shape[1] < 23:
                        raise RuntimeError("Expected COCO-WholeBody foot landmarks.")
                    for joint, source, _ in LANDMARK_MAP:
                        xy = keypoints[person, source, :2]
                        score = float(scores[person, source])
                        if np.isfinite(xy).all() and score >= MIN_SCORE:
                            raw_points[joint] = xy.astype(float)

                smoothed = smooth_points(raw_points, smoothed)
                points = {joint: tuple(int(round(v)) for v in xy)
                          for joint, xy in smoothed.items()}

                if person is None:
                    table.writerow([frame_number, frame_number / fps, 0]
                                   + [""] * 9)
                    cv2.putText(frame, "Pose not detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    for joint, source, name in LANDMARK_MAP:
                        x, y = keypoints[person, source, :2]
                        filtered = smoothed.get(joint)
                        table.writerow([
                            frame_number, frame_number / fps, 1, joint,
                            source, name,
                            float(x) / width if np.isfinite(x) else "",
                            float(y) / height if np.isfinite(y) else "",
                            float(scores[person, source]), int(joint in raw_points),
                            float(filtered[0]) / width if filtered is not None else "",
                            float(filtered[1]) / height if filtered is not None else "",
                        ])
                    draw_skeleton(frame, points)

                writer.write(frame)
                frame_number += 1
                print(f"Processed frame {frame_number}", flush=True)
                ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Finished: {frame_number} frames")
    print(f"Video: {OUTPUT_VIDEO_PATH}")
    print(f"CSV: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
