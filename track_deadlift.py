import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------- Settings ----------

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = PROJECT_DIR / "deadlift.mp4"
MODEL_PATH = PROJECT_DIR / "pose_landmarker_full.task"
OUTPUT_VIDEO_PATH = PROJECT_DIR / "deadlift_tracked.mp4"
OUTPUT_CSV_PATH = PROJECT_DIR / "landmarks.csv"

# Lower values give stronger smoothing but introduce more lag.
# 1.0 disables smoothing.
SMOOTHING_ALPHA = 0.35

MIN_VISIBILITY = 0.5

BONE_COLOR = (0, 255, 0)
SPINE_COLOR = (0, 255, 255)
JOINT_COLOR = (0, 0, 255)
LINE_THICKNESS = 3
JOINT_RADIUS = 4

CONNECTIONS = [
    # Upper arms and forearms
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # Thighs and lower legs
    (23, 25), (25, 27),
    (24, 26), (26, 28),

    # Left foot: ankle, heel, toe
    (27, 29), (29, 31), (31, 27),

    # Right foot: ankle, heel, toe
    (28, 30), (30, 32), (32, 28),
]

SKELETON_JOINTS = {
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32,
}


# ---------- Drawing ----------

def midpoint(a, b):
    return (
        round((a[0] + b[0]) / 2),
        round((a[1] + b[1]) / 2),
    )


def draw_skeleton(frame, points):
    # Major limb bones and foot triangles.
    for start, end in CONNECTIONS:
        if start in points and end in points:
            cv2.line(
                frame,
                points[start],
                points[end],
                BONE_COLOR,
                LINE_THICKNESS,
                cv2.LINE_AA,
            )

    shoulder_center = None
    hip_center = None

    # Shoulder branches.
    if 11 in points and 12 in points:
        shoulder_center = midpoint(points[11], points[12])

        for joint in (11, 12):
            cv2.line(
                frame,
                shoulder_center,
                points[joint],
                BONE_COLOR,
                LINE_THICKNESS,
                cv2.LINE_AA,
            )

    # Hip branches.
    if 23 in points and 24 in points:
        hip_center = midpoint(points[23], points[24])

        for joint in (23, 24):
            cv2.line(
                frame,
                hip_center,
                points[joint],
                BONE_COLOR,
                LINE_THICKNESS,
                cv2.LINE_AA,
            )

    # Single central spine line.
    if shoulder_center is not None and hip_center is not None:
        cv2.line(
            frame,
            shoulder_center,
            hip_center,
            SPINE_COLOR,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )

    for joint in SKELETON_JOINTS:
        if joint in points:
            cv2.circle(
                frame,
                points[joint],
                JOINT_RADIUS,
                JOINT_COLOR,
                -1,
                cv2.LINE_AA,
            )


# ---------- Processing ----------

def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not 0 < SMOOTHING_ALPHA <= 1:
        raise ValueError("SMOOTHING_ALPHA must be between 0 and 1.")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    frame_number = 0

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Cannot read the video's frame rate.")

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Video contains no readable frames.")

        height, width = frame.shape[:2]

        writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Cannot create the output video.")

        with (
            vision.PoseLandmarker.create_from_options(options) as landmarker,
            OUTPUT_CSV_PATH.open("w", newline="") as csv_file,
        ):
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame",
                "time_s",
                "pose_detected",
                "landmark_id",
                "x_normalized",
                "y_normalized",
                "z_relative",
                "visibility",
            ])

            smoothed_points = {}
            previous_timestamp = -1

            while ok:
                time_s = frame_number / fps

                # MediaPipe requires strictly increasing timestamps.
                timestamp_ms = max(
                    previous_timestamp + 1,
                    round(time_s * 1000),
                )
                previous_timestamp = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    raw_points = {}

                    for joint, landmark in enumerate(landmarks):
                        # Preserve raw predictions in the CSV.
                        csv_writer.writerow([
                            frame_number,
                            time_s,
                            1,
                            joint,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                        ])

                        if landmark.visibility >= MIN_VISIBILITY:
                            raw_points[joint] = (
                                landmark.x * width,
                                landmark.y * height,
                            )

                    # Exponential smoothing in pixel coordinates.
                    next_smoothed = {}

                    for joint, current in raw_points.items():
                        previous = smoothed_points.get(
                            joint, current
                        )

                        next_smoothed[joint] = (
                            SMOOTHING_ALPHA * current[0]
                            + (1 - SMOOTHING_ALPHA) * previous[0],

                            SMOOTHING_ALPHA * current[1]
                            + (1 - SMOOTHING_ALPHA) * previous[1],
                        )

                    # Discard missing joints instead of freezing them.
                    smoothed_points = next_smoothed

                    points = {
                        joint: (
                            round(position[0]),
                            round(position[1]),
                        )
                        for joint, position
                        in smoothed_points.items()
                    }

                    draw_skeleton(frame, points)

                else:
                    # Restart smoothing when the person is reacquired.
                    smoothed_points.clear()

                    csv_writer.writerow([
                        frame_number, time_s, 0,
                        "", "", "", "", "",
                    ])

                    cv2.putText(
                        frame,
                        "Pose not detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(frame)
                frame_number += 1

                if frame_number % 100 == 0:
                    print(f"Processed {frame_number} frames")

                ok, frame = cap.read()

    finally:
        cap.release()

        if writer is not None:
            writer.release()

    print(f"Finished: {frame_number} frames")
    print(f"Video: {OUTPUT_VIDEO_PATH}")
    print(f"CSV:   {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()