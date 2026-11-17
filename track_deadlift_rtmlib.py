"""RTMLib motion tracker for a constant-frame-rate, single-lifter video.

Install in your PyCharm environment:
    python -m pip install --upgrade rtmlib onnxruntime numpy

Put this file beside deadlift.mp4. RTMLib downloads its models on first run.
The output video has no audio. Scores are not MediaPipe visibility scores.

Angles use smoothed 2D pixel coordinates, in degrees, for each body side:
  curvature estimate: acos(neck-to-central-hip length / longest valid length);
  hip: shoulder-hip-knee; knee: hip-knee-ankle;
  ankle: knee-ankle-big toe (the foot is approximated by the big toe).
The displayed minimum hip angle is the lowest of both sides over the whole
video, not per repetition. Parameters update only on complete frames (all
mapped landmarks and valid angles); otherwise the last complete display holds.
Lift onset is detected from the lowest setup hip position, followed by
sustained upward hand/hip movement and knee extension. Knee/ankle timing resets
at each detected lift, excluding the approach/setup. A landmark gap invalidates
timing until the next lift rather than silently creating a new baseline.
Positive ankle-minus-knee onset time means the knee changed first. Missing
measurements are blank in angles_rtmlib_halpe26_relative.csv. The overlay holds the last complete
measurement, or shows -- until the first complete frame.

Spine length is the 2D Euclidean distance from detected neck to central hip.
The longest smoothed length among COMPLETE frames over the WHOLE video is the
zero-curvature reference. Estimated curvature = degrees(acos(length / maximum)),
clamped to 0..90 degrees. A ratio sqrt(2)/2 produces 45 degrees. This is the
requested shortening heuristic, not anatomical spinal curvature: perspective,
body rotation, camera distance and detection errors can also change the length.
It cannot distinguish flexion from extension. Head angle is diagnostic only.
The displayed spine angle is the signed change in this length-based estimate
from the start of the current lift, resetting to zero at every lift start.
Positive means a larger estimated angle; negative means a smaller estimated angle.
Before the first lift it is unavailable. CSV retains the absolute estimate,
the lift-start baseline and the relative change for comparison.

Pose is estimated on every frame; a failed crop can be retried with fresh
detection. A second video-reading pass renders using the fixed global calibration. Only landmark arrays are retained, not video frames.
The detector runs periodically, or immediately after tracking loss.
The balanced pose model still processes every frame. ONNX thread pools are
limited to avoid CPU contention. The final console summary reports runtime.
"""

import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import BodyWithFeet


# IPython/interactive consoles do not define __file__. In that case, run
# from the project folder so deadlift.mp4 is found in the working directory.
PROJECT_DIR = (Path(__file__).resolve().parent
               if "__file__" in globals() else Path.cwd())
VIDEO_PATH = PROJECT_DIR / "deadlift.mp4"
OUTPUT_VIDEO_PATH = PROJECT_DIR / "deadlift_rtmlib_halpe26_relative_clean.mp4"
OUTPUT_CSV_PATH = PROJECT_DIR / "landmarks_rtmlib_halpe26_relative.csv"
OUTPUT_ANGLES_PATH = PROJECT_DIR / "angles_rtmlib_halpe26_relative.csv"

# The detector finds people; the pose model locates joints within each crop.
MODE = "balanced"  # Keep the balanced Halpe26 pose model.
DETECTOR_MODE = "balanced"  # Keep detection quality; save work by reusing crops.
DETECTION_INTERVAL = 10  # Re-detect periodically; infer all pose points every frame.
CPU_THREADS = 2  # Avoid competing ONNX Runtime thread pools.
# Filter detections by confidence before smoothing the accepted coordinates.
SMOOTHING_ALPHA = 0.35  # Lower values smooth more; 1.0 disables smoothing.
MIN_SCORE = 0.5  # RTMLib keypoint score; not equivalent to MediaPipe visibility.
ANGLE_CHANGE_THRESHOLD_DEG = 5.0  # Change from lift-start baseline that counts as onset.
LIFT_CONFIRMATION_S = 0.10  # Sustained ascent needed to confirm lift onset.
LIFT_RISE_SPINE_FRACTION = 0.02  # Hip/hand rise relative to calibrated torso length.
HEAD, NECK, PELVIS = 33, 34, 35  # Internal IDs for extra Halpe26 landmarks.

# Internal joint IDs in shoulder, hip, knee, ankle, toe order.
SIDE_JOINTS = {
    "left": (11, 23, 25, 27, 31),
    "right": (12, 24, 26, 28, 32),
}
# One CSV row per side per frame. None is written as a blank measurement.
# Whole-video hip minima and per-lift knee/ankle changes are separate quantities.
ANGLE_FIELDS = [
    "frame", "time_s", "side", "lift_number", "lift_start_time_s", "torso_tilt_deg", "torso_change_deg",
    "hip_angle_deg", "min_hip_angle_deg", "min_hip_time_s",
    "knee_angle_deg", "ankle_angle_deg", "knee_change_deg", "ankle_change_deg",
    "timing_baseline_time_s", "knee_onset_time_s", "ankle_onset_time_s",
    "ankle_minus_knee_onset_s", "knee_changed_before_ankle",
    "knee_change_before_ankle_deg",
    "neck_x_px", "neck_y_px", "head_neck_hip_angle_deg",
    "spine_length_px", "spine_reference_length_px", "spine_length_ratio",
    "spine_curvature_estimate_deg",
    "spine_lift_baseline_deg", "spine_curvature_change_deg",
    "head_neck_tilt_deg", "neck_hip_tilt_deg",
]

# Map Halpe26 landmarks into our existing drawing IDs; extra IDs are above 32.
# The big toe approximates MediaPipe's foot-index landmark; locations differ.
# (internal drawing ID, Halpe26 model ID, name)
LANDMARK_MAP = [
    (11, 5, "left_shoulder"), (12, 6, "right_shoulder"),
    (13, 7, "left_elbow"), (14, 8, "right_elbow"),
    (15, 9, "left_wrist"), (16, 10, "right_wrist"),
    (23, 11, "left_hip"), (24, 12, "right_hip"),
    (25, 13, "left_knee"), (26, 14, "right_knee"),
    (27, 15, "left_ankle"), (28, 16, "right_ankle"),
    (29, 24, "left_heel"), (30, 25, "right_heel"),
    (31, 20, "left_big_toe"), (32, 21, "right_big_toe"),
    (HEAD, 17, "head"), (NECK, 18, "neck"), (PELVIS, 19, "central_hip"),
]

# Limb edges; the head/neck/pelvis chain is drawn separately.
CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (31, 27),
    (28, 30), (30, 32), (32, 28),
]

# OpenCV colours use BGR order, unlike the RGB order common in image libraries.
BONE_COLOR = (0, 255, 0)
SPINE_COLOR = (0, 255, 255)
JOINT_COLOR = (0, 0, 255)


def joint_angle(a, vertex, b):
    """Interior angle (0..180); coincident points have no defined angle."""
    # Both vectors originate at the middle joint; their dot product gives the cosine.
    u, v = np.asarray(a) - np.asarray(vertex), np.asarray(b) - np.asarray(vertex)
    length = np.linalg.norm(u) * np.linalg.norm(v)
    if not np.isfinite(length) or length <= 1e-9:
        return None
    # Clip rounding errors so acos never receives a value outside [-1, 1].
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / length, -1, 1))))


def angle_change(current, previous):
    """Smallest signed angle difference, preserving unavailable measurements."""
    if current is None or previous is None:
        return None
    # A move from +179 to -179 degrees should count as +2 degrees, not -358.
    return (current - previous + 180) % 360 - 180


def spine_posture_angle(points):
    """Head-neck-pelvis angle, an alignment proxy rather than spine curvature."""
    if not all(joint in points for joint in (HEAD, NECK, PELVIS)):
        return None
    return joint_angle(points[HEAD], points[NECK], points[PELVIS])


def spine_length(points):
    """Straight neck-to-central-hip distance in pixels, not the back's arc length."""
    if NECK not in points or PELVIS not in points:
        return None
    length = float(np.linalg.norm(np.asarray(points[NECK]) - points[PELVIS]))
    return length if math.isfinite(length) and length > 1e-9 else None


def length_curvature(length, reference):
    """Requested cosine shortening heuristic; not an anatomical curvature angle."""
    if (length is None or reference is None or not math.isfinite(length)
            or not math.isfinite(reference) or length <= 0 or reference <= 0):
        return None
    # Ratio 1 gives 0 degrees; sqrt(2)/2 gives 45 degrees. Clamp ratios slightly
    # above 1 to handle numerical rounding at the longest observed length.
    return math.degrees(math.acos(float(np.clip(length / reference, 0, 1))))


def segment_tilt(a, b):
    """Signed tilt of the b-to-a segment from image vertical, in degrees."""
    if a is None or b is None:
        return None
    # Image y increases downward; negating y measures tilt from upward vertical.
    vector = np.asarray(a) - np.asarray(b)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-9:
        return None
    return math.degrees(math.atan2(vector[0], -vector[1]))


class AngleTracker:
    """Keep whole-video hip minima and separate knee/ankle timing for each lift.

    render_records() calls update() only on complete frames. Lift starts are
    detected beforehand, so approach/setup motion never sets the baseline.
    Without a detected lift, knee/ankle timing remains unavailable.
    """

    def __init__(self, threshold=ANGLE_CHANGE_THRESHOLD_DEG, spine_reference=None,
                 lift_start_frames=()):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Angle change threshold must be positive and finite.")
        self.threshold = threshold
        if spine_reference is not None and (
                not math.isfinite(spine_reference) or spine_reference <= 0):
            raise ValueError("Spine reference must be positive and finite.")
        self.spine_reference = spine_reference
        # Sorting/deduplication keeps repetition numbers in chronological order.
        self.lift_start_frames = sorted(set(lift_start_frames))
        # Keep the exact lift-start curvature across gaps; never rebase mid-lift.
        self.spine_lift_baseline = None
        self.spine_baseline_start = None
        # The minimum survives between reps; timing resets at each lift or data gap.
        self.state = {side: {"minimum": None, "minimum_time": None,
                             "previous_torso": None, "timing": None}
                      for side in SIDE_JOINTS}

    def update(self, points, frame_number, fps):
        """Return left/right CSV rows from smoothed floating-point pixel coordinates."""
        rows = []
        time_s = frame_number / fps
        # Future lift starts are known offline but must not count as active yet.
        starts = [start for start in self.lift_start_frames if start <= frame_number]
        lift_start = starts[-1] if starts else None
        # Head/neck alignment is diagnostic only; curvature now depends on length.
        spine_angle = spine_posture_angle(points)
        length = spine_length(points)
        curvature = length_curvature(length, self.spine_reference)
        if lift_start is not None and frame_number == lift_start:
            self.spine_lift_baseline = curvature
            self.spine_baseline_start = lift_start
        # If the start frame was unavailable, leave the change unknown for that lift.
        baseline = (self.spine_lift_baseline
                    if lift_start is not None and self.spine_baseline_start == lift_start
                    else None)
        curvature_change = (None if curvature is None or baseline is None
                            else curvature - baseline)
        for side, joints in SIDE_JOINTS.items():
            shoulder, hip, knee, ankle, toe = joints
            state = self.state[side]

            # All three joints must be available to form an individual angle.
            def angle(a, vertex, b):
                if not all(joint in points for joint in (a, vertex, b)):
                    return None
                return joint_angle(points[a], points[vertex], points[b])

            torso = segment_tilt(points.get(shoulder), points.get(hip))
            hip_angle = angle(shoulder, hip, knee)
            knee_angle = angle(hip, knee, ankle)
            ankle_angle = angle(knee, ankle, toe)
            # Update each side's smallest hip angle across the entire clip.
            if hip_angle is not None and (state["minimum"] is None
                                           or hip_angle < state["minimum"]):
                state["minimum"], state["minimum_time"] = hip_angle, time_s
            # Unset fields stay None, producing blank CSV cells rather than fake zeros.
            row = dict.fromkeys(ANGLE_FIELDS)
            row.update(frame=frame_number, time_s=time_s, side=side,
                       lift_number=len(starts) if starts else None,
                       lift_start_time_s=None if lift_start is None else lift_start / fps,
                       torso_tilt_deg=torso,
                       torso_change_deg=angle_change(torso, state["previous_torso"]),
                       hip_angle_deg=hip_angle, min_hip_angle_deg=state["minimum"],
                       min_hip_time_s=state["minimum_time"],
                       knee_angle_deg=knee_angle, ankle_angle_deg=ankle_angle,
                       neck_x_px=None if NECK not in points else float(points[NECK][0]),
                       neck_y_px=None if NECK not in points else float(points[NECK][1]),
                       head_neck_hip_angle_deg=spine_angle,
                       spine_length_px=length,
                       spine_reference_length_px=self.spine_reference,
                       spine_length_ratio=(None if length is None or self.spine_reference is None
                                           else min(1.0, length / self.spine_reference)),
                       spine_curvature_estimate_deg=curvature,
                       spine_lift_baseline_deg=baseline,
                       spine_curvature_change_deg=curvature_change,
                       head_neck_tilt_deg=segment_tilt(points.get(HEAD), points.get(NECK)),
                       neck_hip_tilt_deg=segment_tilt(points.get(NECK), points.get(PELVIS)))
            state["previous_torso"] = torso

            # Establish the baseline ONLY at the detected start of each lift.
            if knee_angle is None or ankle_angle is None or lift_start is None:
                state["timing"] = None
            elif frame_number == lift_start:
                state["timing"] = {
                    "baseline_time": time_s, "knee": knee_angle,
                    "ankle": ankle_angle, "knee_onset": None,
                    "ankle_onset": None, "previous_knee_change": 0.0,
                    "knee_before_ankle": None,
                }
            # After a gap, do not create another baseline halfway through the lift.
            # Timing remains unavailable until the next detected lift start.
            if state["timing"] is not None:
                timing = state["timing"]
                # Signed changes are measured from this repetition's starting posture.
                knee_change = knee_angle - timing["knee"]
                ankle_change = ankle_angle - timing["ankle"]
                # Record only the first threshold crossing per joint per repetition.
                # Either direction of motion can cross the absolute-change threshold.
                if timing["knee_onset"] is None and abs(knee_change) >= self.threshold:
                    timing["knee_onset"] = time_s
                if timing["ankle_onset"] is None and abs(ankle_change) >= self.threshold:
                    timing["ankle_onset"] = time_s
                    # Freeze knee change from the frame immediately BEFORE ankle onset;
                    # this is not the maximum knee movement reached later in the rep.
                    timing["knee_before_ankle"] = timing["previous_knee_change"]
                knee_onset, ankle_onset = timing["knee_onset"], timing["ankle_onset"]
                # A positive lag means ankle onset happened after knee onset.
                lag = None
                if knee_onset is not None and ankle_onset is not None:
                    lag = ankle_onset - knee_onset
                row.update(
                    knee_change_deg=knee_change, ankle_change_deg=ankle_change,
                    timing_baseline_time_s=timing["baseline_time"],
                    knee_onset_time_s=knee_onset, ankle_onset_time_s=ankle_onset,
                    ankle_minus_knee_onset_s=lag,
                    knee_changed_before_ankle=int(lag > 0) if lag is not None else None,
                    knee_change_before_ankle_deg=timing["knee_before_ankle"],
                )
                timing["previous_knee_change"] = knee_change
            rows.append(row)
        return rows


def draw_angles(frame, rows):
    """Draw three metric lines in place using the latest complete measurements."""
    def value(number, suffix=" deg"):
        return "--" if number is None else f"{number:.1f}{suffix}"

    # Reduce left/right records to one hip minimum and one higher signed knee value.
    # Higher signed value is different from largest absolute magnitude.
    hip_minima = [row['min_hip_angle_deg'] for row in rows
                  if row['min_hip_angle_deg'] is not None]
    knee_changes = [row['knee_change_before_ankle_deg'] for row in rows
                    if row['knee_change_before_ankle_deg'] is not None]
    # The angle is relative to this lift's starting angle. Length still uses the
    # fixed whole-video longest reference for the underlying absolute estimate.
    lines = [
        f"Spine angle change from lift start: {value(rows[0]['spine_curvature_change_deg'])}",
        f"Lowest hip angle: {value(min(hip_minima) if hip_minima else None)}",
        f"Knee change before ankle: {value(max(knee_changes) if knee_changes else None)}",
    ]
    # Scale the complete panel to fit smaller videos.
    panel = np.zeros((len(lines) * 26 + 16, 850, 3), dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(panel, text, (8, 26 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    scale = min(1.0, frame.shape[1] / panel.shape[1],
                frame.shape[0] * 0.4 / panel.shape[0])
    panel = cv2.resize(panel, None, fx=scale, fy=scale)
    height, width = panel.shape[:2]
    frame[-height:, :width] = panel


def draw_skeleton(frame, points):
    """Draw available joints and connections using rounded integer pixel locations."""
    def line(a, b, color=BONE_COLOR):
        cv2.line(frame, a, b, color, 3, cv2.LINE_AA)

    for start, end in CONNECTIONS:
        if start in points and end in points:
            line(points[start], points[end])

    # Draw the measured head-neck-pelvis chain, without inventing a back curve.
    for start, end in ((HEAD, NECK), (NECK, PELVIS), (NECK, 11), (NECK, 12),
                       (PELVIS, 23), (PELVIS, 24)):
        if start in points and end in points:
            line(points[start], points[end], SPINE_COLOR)

    # Joints are drawn last so their centres remain visible over the bone lines.
    for point in points.values():
        cv2.circle(frame, point, 4, JOINT_COLOR, -1, cv2.LINE_AA)


def select_person(keypoints, scores):
    """Select the largest visible body; assumes one prominent lifter.

    This is not identity tracking. Crop the video if other people compete
    for selection. Size uses the first 17 body points, including face anchors,
    and excludes the extra head/neck/pelvis, foot and hand landmarks.
    """
    best_index, best_area = None, -1
    for index in range(len(keypoints)):
        body = keypoints[index, :17]
        valid = (scores[index, :17] >= MIN_SCORE)
        valid &= np.isfinite(body).all(axis=1)
        # A few isolated detections cannot give a useful body-size estimate.
        if np.count_nonzero(valid) < 4:
            continue
        # Compare bounding-rectangle areas of confident points, not score totals.
        extent = np.ptp(body[valid], axis=0)
        area = float(extent[0] * extent[1])
        if area > best_area:
            best_index, best_area = index, area
    return best_index


def smooth_points(raw_points, previous):
    """Exponential moving average: alpha * current + (1-alpha) * previous."""
    # Only complete frames are committed to the smoothing history by main().
    # Missing landmarks stay missing; smoothing never invents a replacement point.
    # A point without history starts at its current detected position.
    return {
        joint: SMOOTHING_ALPHA * current
        + (1 - SMOOTHING_ALPHA) * previous.get(joint, current)
        for joint, current in raw_points.items()
    }


def measurements_ready(points):
    """Accept measurements atomically, only with the complete mapped skeleton."""
    if any(joint not in points or not np.isfinite(points[joint]).all()
           for joint, _, _ in LANDMARK_MAP):
        return False
    # Confidence is not enough: coincident points can still make angles undefined.
    for shoulder, hip, knee, ankle, toe in SIDE_JOINTS.values():
        for a, vertex, b in ((shoulder, hip, knee), (hip, knee, ankle),
                             (knee, ankle, toe)):
            if joint_angle(points[a], points[vertex], points[b]) is None:
                return False
    return spine_length(points) is not None and spine_posture_angle(points) is not None


def empty_angle_rows(frame_number, fps):
    """Keep frame/time/side metadata while leaving all measurements blank."""
    rows = []
    for side in SIDE_JOINTS:
        row = dict.fromkeys(ANGLE_FIELDS)
        row.update(frame=frame_number, time_s=frame_number / fps, side=side)
        rows.append(row)
    return rows


def make_writer(path, fps, size):
    """Create a silent MP4 writer; size is (width, height), not NumPy shape order."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create video: {path}")
    return writer


class FastPoseEstimator:
    """Infer pose every frame inside a tracked crop; periodically re-detect.

    Detection is expensive. Between detector runs, confident joints define the
    next crop. Missing required points trigger a same-frame detector retry;
    persistent loss also forces fresh detection on the next frame.
    """

    def __init__(self, model, detection_interval=DETECTION_INTERVAL):
        if detection_interval < 1:
            raise ValueError("Detection interval must be at least one frame.")
        self.model = model
        self.interval = detection_interval
        # The crop uses [left, top, right, bottom] in original-frame pixels.
        self.box = None
        self.frame_number = 0
        self.detection_count = 0

    def __call__(self, frame):
        # Periodic full-image detection corrects drift; no reliable crop forces it too.
        detect = self.box is None or self.frame_number % self.interval == 0
        if detect:
            boxes = self.model.det_model(frame)
            self.detection_count += 1
            # Do not treat the full image as a person when detection finds nobody.
            if len(boxes) == 0:
                self.box = None
                self.frame_number += 1
                return np.empty((0, 26, 2)), np.empty((0, 26))
        else:
            boxes = np.asarray([self.box])
        # Reusing a detector crop does not skip pose estimation on this frame.
        keypoints, scores = self.model.pose_model(frame, bboxes=boxes)
        keypoints, scores = np.asarray(keypoints), np.asarray(scores)
        person = select_person(keypoints, scores)
        required = [source for _, source, _ in LANDMARK_MAP]
        if not detect and (person is None or not (
                (scores[person, required] >= MIN_SCORE).all()
                and np.isfinite(keypoints[person, required]).all())):
            # Retry this SAME frame with a fresh detector crop before rejecting it.
            boxes = self.model.det_model(frame)
            self.detection_count += 1
            if len(boxes):
                keypoints, scores = self.model.pose_model(frame, bboxes=boxes)
                keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                person = select_person(keypoints, scores)
            else:
                keypoints, scores = np.empty((0, 26, 2)), np.empty((0, 26))
                person = None
        self.box = None
        if person is not None:
            selected, confidence = keypoints[person], scores[person]
            valid = (confidence >= MIN_SCORE) & np.isfinite(selected).all(axis=1)
            # Persistent loss forces fresh detection on the next frame as well.
            if valid[required].all():
                lower, upper = selected[valid].min(axis=0), selected[valid].max(axis=0)
                # Padding leaves room for movement; clamp the crop to image boundaries.
                margin = np.maximum((upper - lower) * 0.2, 20)
                lower = np.maximum(lower - margin, [0, 0])
                upper = np.minimum(upper + margin, [frame.shape[1], frame.shape[0]])
                if np.all(upper > lower):
                    self.box = np.r_[lower, upper]
            # Keep a leading person dimension, even though only one lifter is returned.
            keypoints, scores = keypoints[person:person + 1], scores[person:person + 1]
        self.frame_number += 1
        return keypoints, scores


def make_pose_estimator():
    """Load native Halpe26 models and configure their CPU inference sessions."""
    import onnxruntime as ort

    # Native ordering (to_openpose=False) must agree with LANDMARK_MAP's source IDs.
    detector = BodyWithFeet.MODE[DETECTOR_MODE]
    model = BodyWithFeet(det=detector['det'], det_input_size=detector['det_input_size'],
                         to_openpose=False, mode=MODE, backend="onnxruntime", device="cpu")
    # RTMLib does not expose SessionOptions; replace its default CPU sessions.
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    # Limit separate model thread pools so they do not compete for every CPU core.
    options.inter_op_num_threads = 1
    # Idle workers should sleep instead of burning CPU while the other model runs.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    for component in (model.det_model, model.pose_model):
        component.session = None  # Release the old pool before creating another.
        component.session = ort.InferenceSession(component.onnx_model, sess_options=options,
                                                 providers=["CPUExecutionProvider"])
    return FastPoseEstimator(model)


def extract_pose(keypoints, scores, previous, width, height):
    """Return (model XY, scores, raw points, smoothed points, complete-frame flag).

    Model arrays retain native Halpe26 ordering for CSV export. Point dictionaries
    use internal drawing IDs, with coordinates measured in original-image pixels.
    """
    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
    person = select_person(keypoints, scores)
    raw = {}
    selected, confidence = None, None
    if person is not None:
        if keypoints.shape[1] != 26:
            raise RuntimeError("Expected 26 Halpe26 landmarks from BodyWithFeet.")
        # Copies keep saved frame records independent of subsequent model calls.
        selected, confidence = keypoints[person].copy(), scores[person].copy()
        for joint, source, _ in LANDMARK_MAP:
            xy, score = selected[source, :2], float(confidence[source])
            # Reject low-confidence, non-finite and off-image points before smoothing.
            if (np.isfinite(xy).all() and math.isfinite(score) and score >= MIN_SCORE
                    and 0 <= xy[0] < width and 0 <= xy[1] < height):
                raw[joint] = xy.astype(float)
    smoothed = smooth_points(raw, previous)
    return selected, confidence, raw, smoothed, measurements_ready(smoothed)


def detect_lift_starts(records, fps, spine_reference):
    """Infer repetition starts from hand/hip ascent after a low setup position.

    This uses wrists as a bar-motion proxy; onset is estimated, not bar detection.
    Confirmation looks ahead, then backdates the start to the lowest setup hip.
    """
    if spine_reference is None:
        return []
    # Scale the required rise to torso size rather than using a fixed pixel distance.
    rise = spine_reference * LIFT_RISE_SPINE_FRACTION
    # Convert the confirmation duration to consecutive frames, with a minimum of two.
    confirmation = max(2, math.ceil(fps * LIFT_CONFIRMATION_S))
    # Candidate layout: (frame, hand height, hip height, mean knee angle).
    starts, candidate = [], None
    waiting_for_setup, ascending = True, 0
    for number, record in enumerate(records):
        points, complete = record[3], record[4]
        # Missing data breaks the evidence of sustained upward movement.
        if not complete:
            candidate, ascending = None, 0
            continue
        hand_y = float(np.mean([points[j][1] for j in (15, 16)]))
        ankle_y = float(np.mean([points[j][1] for j in (27, 28)]))
        hip_y = float(np.mean([points[j][1] for j in (23, 24)]))
        knee_y = float(np.mean([points[j][1] for j in (25, 26)]))
        knee_angle = float(np.mean([joint_angle(points[h], points[k], points[a])
                                   for _, h, k, a, _ in SIDE_JOINTS.values()]))
        # Height above the ankles increases upward. This reduces shared vertical
        # translation effects but does not correct for rotation or perspective.
        hand_height, hip_height = ankle_y - hand_y, ankle_y - hip_y
        # Wait for near-upright knees and raised hands before seeking the next setup,
        # so one ascent cannot be counted as several repetitions.
        if not waiting_for_setup:
            if knee_angle >= 165 and hand_y < knee_y:
                waiting_for_setup = True
            continue
        # Hands at/below the knees plus bent knees identify a possible low setup.
        if candidate is None:
            if hand_y >= knee_y and knee_angle < 165:
                candidate = (number, hand_height, hip_height, knee_angle)
            continue
        # Follow the lowest/latest hip position while the lifter is still descending.
        if hip_height <= candidate[2]:
            candidate = (number, hand_height, hip_height, knee_angle)
            ascending = 0
            continue
        # Require hand rise, hip rise AND knee extension to reject hand adjustments
        # or minor posture changes while the lifter approaches the bar.
        rising = (hand_height - candidate[1] >= rise
                  and hip_height - candidate[2] >= rise
                  and knee_angle - candidate[3] >= 2)
        ascending = ascending + 1 if rising else 0
        # Confirm later, then backdate onset to the stored setup bottom.
        if ascending >= confirmation:
            starts.append(candidate[0])
            candidate, ascending, waiting_for_setup = None, 0, False
    return starts


def render_records(records, fps, width, height, reference):
    """Render cached detections and CSVs without repeating model inference.

    records use extract_pose()'s tuple layout. reference is the final longest
    valid spine length. Decode the source again to obtain clean video frames.
    """
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen {VIDEO_PATH}")
        writer = make_writer(OUTPUT_VIDEO_PATH, fps, (width, height))
        lift_starts = detect_lift_starts(records, fps, reference)
        tracker = AngleTracker(spine_reference=reference, lift_start_frames=lift_starts)
        # Show -- until the first complete measurement is available.
        displayed_rows = empty_angle_rows(0, fps)
        with (OUTPUT_CSV_PATH.open("w", newline="") as csv_file,
              OUTPUT_ANGLES_PATH.open("w", newline="") as angles_file):
            table = csv.writer(csv_file)
            angles_table = csv.DictWriter(angles_file, fieldnames=ANGLE_FIELDS)
            angles_table.writeheader()
            table.writerow([
                "frame", "time_s", "pose_detected", "landmark_id",
                "rtmlib_id", "name", "x_normalized", "y_normalized",
                "score", "accepted", "smoothed_x_normalized", "smoothed_y_normalized",
            ])
            for frame_number, (selected, confidence, raw, smoothed, complete) in enumerate(records):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot reread frame {frame_number} for rendering.")
                points = {joint: tuple(int(round(v)) for v in xy)
                          for joint, xy in smoothed.items()}
                if selected is None:
                    table.writerow([frame_number, frame_number / fps, 0] + [""] * 9)
                    cv2.putText(frame, "Pose not detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    # Export raw detections/scores even when rejected. Smoothed positions
                    # stay blank if unavailable. Divide coordinates by image dimensions
                    # for normalised CSV positions; angle calculations still use pixels.
                    for joint, source, name in LANDMARK_MAP:
                        x, y = selected[source, :2]
                        filtered = smoothed.get(joint)
                        table.writerow([
                            frame_number, frame_number / fps, 1, joint, source, name,
                            float(x) / width if np.isfinite(x) else "",
                            float(y) / height if np.isfinite(y) else "",
                            float(confidence[source]), int(joint in raw),
                            float(filtered[0]) / width if filtered is not None else "",
                            float(filtered[1]) / height if filtered is not None else "",
                        ])
                    draw_skeleton(frame, points)
                # Update the parameters together only on a complete frame.
                if complete:
                    angle_rows = tracker.update(smoothed, frame_number, fps)
                    displayed_rows = angle_rows
                else:
                    # CSV marks missing data; the display holds the previous complete
                    # frame. Held measurements must not be exported as fresh data.
                    angle_rows = empty_angle_rows(frame_number, fps)
                    for state in tracker.state.values():
                        state['timing'] = None
                        state['previous_torso'] = None
                angles_table.writerows(angle_rows)
                draw_angles(frame, displayed_rows)
                writer.write(frame)
    finally:
        # Closing the writer finalises the MP4, including after an interrupted render.
        cap.release()
        if writer is not None:
            writer.release()
    return lift_starts


def main():
    """Validate input, infer/cache landmarks, calibrate length, then render outputs."""
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not 0 < SMOOTHING_ALPHA <= 1:
        raise ValueError("SMOOTHING_ALPHA must be greater than 0 and at most 1.")
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {VIDEO_PATH}")
        # Frame index / original FPS supplies timestamps, regardless of processing speed.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Cannot read the video's frame rate.")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Video contains no readable frames.")
        height, width = frame.shape[:2]
        print("Loading RTMLib models (first run downloads model files)...", flush=True)
        estimator = make_pose_estimator()
        # Pass 1 keeps landmark arrays, not memory-heavy video frames. Clean source
        # images are read again in pass 2 after global calibration is available.
        records, smoothing_history = [], {}
        reference = None
        inference_started = time.perf_counter()
        while ok:
            keypoints, scores = estimator(frame)
            record = extract_pose(keypoints, scores, smoothing_history, width, height)
            records.append(record)
            # Tuple entries 3/4 hold smoothed points and the completeness flag.
            # Incomplete frames cannot affect history or the zero-curvature reference.
            if record[4]:
                smoothing_history = record[3]
                length = spine_length(record[3])
                reference = length if reference is None else max(reference, length)
            if len(records) % 30 == 0:
                print(f"Inferred {len(records)} frames", flush=True)
            ok, frame = cap.read()
        inference_seconds = time.perf_counter() - inference_started
    finally:
        cap.release()
    # Pass 2 uses the final longest length on EVERY frame. A running maximum would
    # otherwise change the curvature calibration midway through the clip.
    lift_starts = render_records(records, fps, width, height, reference)
    # Total runtime includes model loading, inference, CSV output and video encoding.
    elapsed = time.perf_counter() - started
    print(f"Finished: {len(records)} frames in {elapsed:.2f}s ({len(records) / elapsed:.2f} fps)")
    print(f"Inference: {inference_seconds:.2f}s; spine reference: {reference} px")
    print(f"Detected lift starts: {[round(start / fps, 3) for start in lift_starts]} s")
    print(f"Detector runs: {estimator.detection_count}; pose frames: {len(records)}")
    print(f"Video: {OUTPUT_VIDEO_PATH}")
    print(f"CSV: {OUTPUT_CSV_PATH}")
    print(f"Angles: {OUTPUT_ANGLES_PATH}")
    return {"frames": len(records), "elapsed_s": elapsed,
            "inference_s": inference_seconds, "spine_reference_px": reference,
            "lift_start_frames": lift_starts}


# Importing helpers for tests does not load models or process a video.
if __name__ == "__main__":
    main()
