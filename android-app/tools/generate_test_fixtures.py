"""Developer-only: regenerate parity fixtures from the desktop tracker outputs and models."""
import csv
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import BodyWithFeet

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT.parent


def main():
    points = defaultdict(dict)
    expected = defaultdict(dict)
    cap = cv2.VideoCapture(str(DESKTOP / "deadlift.mp4"))
    width, height = cap.get(3), cap.get(4)
    with (DESKTOP / "landmarks_rtmlib_halpe26_relative.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["accepted"] == "1":
                points[int(row["frame"])][int(row["rtmlib_id"])] = (
                    float(row["smoothed_x_normalized"]) * width,
                    float(row["smoothed_y_normalized"]) * height,
                )
    with (DESKTOP / "angles_rtmlib_halpe26_relative.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            expected[int(row["frame"])][row["side"]] = row
    resources = ROOT / "app/src/test/resources"
    resources.mkdir(parents=True, exist_ok=True)
    with (resources / "desktop_reference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_us"] + [f"p{j}_{axis}" for j in range(26) for axis in ("x", "y")]
                        + ["absolute", "baseline", "relative", "left_min", "right_min", "left_knee", "right_knee"])
        for number in sorted(expected):
            left, right = expected[number]["left"], expected[number]["right"]
            writer.writerow([round(float(left["time_s"]) * 1e6)]
                            + [value for joint in range(26) for value in points[number].get(joint, ("NaN", "NaN"))]
                            + [left["spine_curvature_estimate_deg"], left["spine_lift_baseline_deg"], left["spine_curvature_change_deg"],
                               left["min_hip_angle_deg"], right["min_hip_angle_deg"], left["knee_change_before_ankle_deg"], right["knee_change_before_ankle_deg"]])
    ok, image = cap.read()
    cap.release()
    assert ok
    scale = min(1., 1280 / max(image.shape[:2]))
    size = tuple((int(value * scale) // 2) * 2 for value in (image.shape[1], image.shape[0]))
    image = cv2.resize(image, size)
    assets = ROOT / "app/src/androidTest/assets"
    assets.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(assets / "pose-fixture.png"), image)
    original = ort.InferenceSession

    def session(*args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        kwargs["sess_options"] = options
        return original(*args, **kwargs)

    ort.InferenceSession = session
    model = BodyWithFeet(det=str(ROOT / "app/src/main/assets/models/detector.onnx"),
                        pose=str(ROOT / "app/src/main/assets/models/pose.onnx"),
                        to_openpose=False, mode="balanced", backend="onnxruntime", device="cpu")
    boxes = model.det_model(image)
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    xy, scores = model.pose_model(image, bboxes=[box])
    (assets / "pose-fixture.json").write_text(json.dumps({"bbox": box.tolist(), "points": xy[0].tolist(), "scores": scores[0].tolist()}, indent=2) + "\n")
    print("Wrote desktop analysis fixture and native model fixture.")


if __name__ == "__main__":
    main()
