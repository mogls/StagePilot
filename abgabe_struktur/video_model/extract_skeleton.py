#!/usr/bin/env python3
"""
extract_skeleton.py

Extract per-frame human skeletons from an arbitrary video using MediaPipe
(pose + hands), and remap the results into OpenPose's BODY_25 + hand
keypoint layout so the output is compatible with pipelines built around
the youtube-gesture-dataset / OpenPose format (e.g. run_openpose.py's
save_skeleton_to_pickle output).

Why MediaPipe instead of OpenPose:
    OpenPose v1.4 (used by the original dataset) requires building a
    CUDA/Caffe/CMake C++ project from source and is effectively
    unmaintained. MediaPipe is `pip install`-able, runs on CPU or GPU,
    and is actively maintained. The tradeoff is that MediaPipe's own
    keypoint layout differs from OpenPose's, so this script remaps.

What is and isn't remapped:
    - Body (25 points, BODY_25 order): mapped from MediaPipe's 33 pose
      landmarks. Neck and MidHip are computed as midpoints (OpenPose has
      them, MediaPipe doesn't). LSmallToe/RSmallToe have no MediaPipe
      equivalent and are zero-filled with confidence 0.
    - Hands (21 points per hand): MediaPipe's hand landmark order
      (wrist, thumb x4, index x4, middle x4, ring x4, pinky x4) is
      IDENTICAL to OpenPose's hand keypoint order, so this is a direct
      1:1 copy, no approximation involved.
    - Face (70 points, OpenPose order): mapped from MediaPipe's 478-point
      face mesh (468 surface points + 10 iris points, which the Tasks
      API FaceLandmarker returns by default). The 68-point subset uses
      a community-published index mapping (independently verified
      against two production repos: Real3DPortrait and mimictalk, both
      converting MediaPipe mesh -> classic 68-point face-alignment
      layout for face-driven video generation, where wrong indices
      would visibly break the pipeline). The 2 pupil points (indices
      68, 69) are added from MediaPipe's iris center landmarks, matched
      to OpenPose's own convention confirmed directly from its source
      (poseParameters.cpp): index 68 = RPupil, index 69 = LPupil.
      Caveat: the 68-point subset is a well-tested community mapping,
      not an official Google/CMU correspondence table, so treat exact
      point-for-point placement as good-but-not-guaranteed, especially
      within the eyebrow/eye groupings.

Output format:
    A pickle containing a list of per-frame entries. Each entry is a
    list of "people" dicts (mirrors the structure read by the original
    repo's read_skeleton_json / save_skeleton_to_pickle):

        [
          {
            "pose_keypoints_2d": [x0,y0,c0, x1,y1,c1, ... 25 points],
            "hand_left_keypoints_2d": [ ... 21 points ],
            "hand_right_keypoints_2d": [ ... 21 points ],
            "face_keypoints_2d": [ ... 70 points, OpenPose order ],
          },
          ...
        ]

Usage:
    python extract_skeleton.py --video path/to/video.mp4 --output out.pickle
    python extract_skeleton.py --video video.mp4 --output out.pickle --no-face
    python extract_skeleton.py --video video.mp4 --output out.pickle --no-hands

Requirements:
    pip install mediapipe opencv-python numpy
"""

import argparse
import os
import pickle
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model download (MediaPipe Tasks API needs local .task model bundle files)
# ---------------------------------------------------------------------------

MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def ensure_model(models_dir, key):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{key}_landmarker.task")
    if not os.path.exists(path):
        url = MODEL_URLS[key]
        print(f"Downloading {key} model from {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


# ---------------------------------------------------------------------------
# BODY_25 remap: MediaPipe Pose (33 landmarks) -> OpenPose BODY_25 (25 pts)
# ---------------------------------------------------------------------------

MP_POSE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32,
}

# BODY_25 order. "MID:a,b" = midpoint of two MP_POSE landmarks (computed).
# None = no MediaPipe equivalent -> zero-filled, confidence 0.
BODY_25_LAYOUT = [
    ("Nose", "nose"),
    ("Neck", ("MID", "left_shoulder", "right_shoulder")),
    ("RShoulder", "right_shoulder"),
    ("RElbow", "right_elbow"),
    ("RWrist", "right_wrist"),
    ("LShoulder", "left_shoulder"),
    ("LElbow", "left_elbow"),
    ("LWrist", "left_wrist"),
    ("MidHip", ("MID", "left_hip", "right_hip")),
    ("RHip", "right_hip"),
    ("RKnee", "right_knee"),
    ("RAnkle", "right_ankle"),
    ("LHip", "left_hip"),
    ("LKnee", "left_knee"),
    ("LAnkle", "left_ankle"),
    ("REye", "right_eye"),
    ("LEye", "left_eye"),
    ("REar", "right_ear"),
    ("LEar", "left_ear"),
    ("LBigToe", "left_foot_index"),
    ("LSmallToe", None),
    ("LHeel", "left_heel"),
    ("RBigToe", "right_foot_index"),
    ("RSmallToe", None),
    ("RHeel", "right_heel"),
]


def remap_pose_to_body25(pose_landmarks, width, height):
    """pose_landmarks: MediaPipe NormalizedLandmark list (33 items) or None."""
    out = []
    if pose_landmarks is None:
        return [0.0, 0.0, 0.0] * 25

    def get_xy_conf(name):
        idx = MP_POSE[name]
        lm = pose_landmarks[idx]
        conf = getattr(lm, "visibility", 1.0)
        return lm.x * width, lm.y * height, conf

    for _, src in BODY_25_LAYOUT:
        if src is None:
            out.extend([0.0, 0.0, 0.0])
        elif isinstance(src, tuple):  # midpoint
            _, a, b = src
            xa, ya, ca = get_xy_conf(a)
            xb, yb, cb = get_xy_conf(b)
            out.extend([(xa + xb) / 2, (ya + yb) / 2, min(ca, cb)])
        else:
            x, y, c = get_xy_conf(src)
            out.extend([x, y, c])
    return out


def hand_landmarks_to_openpose(hand_landmarks, width, height):
    """MediaPipe hand landmark order matches OpenPose hand order exactly."""
    if hand_landmarks is None:
        return [0.0, 0.0, 0.0] * 21
    out = []
    for lm in hand_landmarks:
        out.extend([lm.x * width, lm.y * height, 1.0])
    return out



# ---------------------------------------------------------------------------
# Face remap: MediaPipe FaceMesh (478 pts, incl. iris) -> OpenPose 70-point
# face format (68-point layout + 2 pupils).
#
# The 68-point index list below is a community mapping into MediaPipe's mesh
# indices, cross-checked against two independent production repos
# (Real3DPortrait, mimictalk) that use it to drive face-alignment /
# talking-head models -- a context where a wrong index would visibly wreck
# results, which is why it's trusted here more than an unverifiable guess.
# Order is jaw(17) + eyebrows(10) + nose(9) + eyes(12) + mouth(20) = 68,
# matching the classic dlib/iBUG 68-point grouping sizes.
# ---------------------------------------------------------------------------

_JAW = [356, 454, 361, 288, 397, 379, 378, 377, 152, 148, 149, 150, 172, 58, 132, 234, 127]
_BROW = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
_NOSE = [6, 5, 1, 2, 129, 240, 2, 460, 358]
_EYE = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FACE_68_MP_INDICES = _JAW + _BROW + _NOSE + _EYE + _MOUTH  # 68 indices into the 478-point mesh

# Iris centers (present when the FaceLandmarker's 478-point output is used).
# MediaPipe convention: 468 = subject's left iris center, 473 = right.
MP_LEFT_IRIS_CENTER = 468
MP_RIGHT_IRIS_CENTER = 473
# OpenPose convention (confirmed in poseParameters.cpp): 68 = RPupil, 69 = LPupil.


def remap_face_to_openpose70(face_landmarks, width, height):
    if face_landmarks is None or len(face_landmarks) < 478:
        return [0.0, 0.0, 0.0] * 70

    out = []
    for idx in FACE_68_MP_INDICES:
        lm = face_landmarks[idx]
        out.extend([lm.x * width, lm.y * height, 1.0])

    r_pupil = face_landmarks[MP_RIGHT_IRIS_CENTER]
    l_pupil = face_landmarks[MP_LEFT_IRIS_CENTER]
    out.extend([r_pupil.x * width, r_pupil.y * height, 1.0])  # index 68: RPupil
    out.extend([l_pupil.x * width, l_pupil.y * height, 1.0])  # index 69: LPupil
    return out


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(video_path, output_path, models_dir, use_hands=True, use_face=True):
    pose_model = ensure_model(models_dir, "pose")
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )

    hand_landmarker = None
    if use_hands:
        hand_model = ensure_model(models_dir, "hand")
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_model),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )

    face_landmarker = None
    if use_face:
        face_model = ensure_model(models_dir, "face")
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=face_model),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    frames_out = []

    print(f"Video: {video_path} ({width}x{height} @ {fps:.2f}fps)")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_lm = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        left_hand_lm, right_hand_lm = None, None
        if hand_landmarker is not None:
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            for lm_list, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                if label == "Left":
                    left_hand_lm = lm_list
                else:
                    right_hand_lm = lm_list

        face_lm = None
        if face_landmarker is not None:
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_lm = face_result.face_landmarks[0] if face_result.face_landmarks else None

        person = {
            "pose_keypoints_2d": remap_pose_to_body25(pose_lm, width, height),
            "hand_left_keypoints_2d": hand_landmarks_to_openpose(left_hand_lm, width, height),
            "hand_right_keypoints_2d": hand_landmarks_to_openpose(right_hand_lm, width, height),
            "face_keypoints_2d": remap_face_to_openpose70(face_lm, width, height),
        }
        frames_out.append([person])

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx} frames...")

    cap.release()
    pose_landmarker.close()
    if hand_landmarker is not None:
        hand_landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()

    with open(output_path, "wb") as f:
        pickle.dump(frames_out, f)

    print(f"Done. {frame_idx} frames -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output .pickle file")
    parser.add_argument("--models-dir", default="./mp_models", help="Where to cache downloaded .task model files")
    parser.add_argument("--no-hands", action="store_true", help="Skip hand landmark extraction")
    parser.add_argument("--no-face", action="store_true", help="Skip face landmark extraction")
    args = parser.parse_args()

    extract(
        args.video,
        args.output,
        args.models_dir,
        use_hands=not args.no_hands,
        use_face=not args.no_face,
    )


if __name__ == "__main__":
    main()
