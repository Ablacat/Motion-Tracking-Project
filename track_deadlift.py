import csv
import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VIDEO_PATH = "deadlift.mp4"
MODEL_PATH = "pose_landmarker_full.task"

# Selected torso, arm, leg, and foot connections.
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

writer = None

try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Cannot read the video's frame rate.")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Video contains no readable frames.")

    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        "deadlift_tracked.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create the output video.")

    with (
        vision.PoseLandmarker.create_from_options(options) as landmarker,
        open("landmarks.csv", "w", newline="") as csv_file,
    ):
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "time_s", "pose_detected", "landmark_id",
            "x_normalized", "y_normalized", "z_relative", "visibility",
        ])

        frame_number = 0
        previous_timestamp = -1

        while ok:
            time_s = frame_number / fps
            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(
                previous_timestamp + 1, round(time_s * 1000)
            )
            previous_timestamp = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                points = {}

                for i, landmark in enumerate(landmarks):
                    csv_writer.writerow([
                        frame_number, time_s, 1, i,
                        landmark.x, landmark.y, landmark.z,
                        landmark.visibility,
                    ])

                    if landmark.visibility >= 0.5:
                        points[i] = (
                            round(landmark.x * width),
                            round(landmark.y * height),
                        )

                for start, end in CONNECTIONS:
                    if start in points and end in points:
                        cv2.line(
                            frame, points[start], points[end],
                            (0, 255, 0), 2,
                        )

                for point in points.values():
                    cv2.circle(frame, point, 4, (0, 0, 255), -1)

            else:
                csv_writer.writerow([
                    frame_number, time_s, 0, "", "", "", "", ""
                ])
                cv2.putText(
                    frame, "Pose not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            writer.write(frame)
            frame_number += 1
            ok, frame = cap.read()

finally:
    cap.release()
    if writer is not None:
        writer.release()

print("Saved deadlift_tracked.mp4 and landmarks.csv")