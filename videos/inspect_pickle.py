"""
inspect_pickle.py — read and print the contents of a .pickle file

Usage:
    python inspect_pickle.py path/to/file.pickle
    python inspect_pickle.py ./data/ --all
"""

import os
import pickle
import argparse
import numpy as np


KEYPOINT_KEYS = [
    "pose_keypoints",
    "face_keypoints",
    "hand_left_keypoints",
    "hand_right_keypoints",
]

# Body25 joint names for pose_keypoints
BODY25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]


def parse_person(person_dict):
    """
    Convert a person dict into named numpy arrays per keypoint group.
    Each flat list [x,y,c, x,y,c, ...] becomes shape (N, 3).
    """
    result = {}
    for key in KEYPOINT_KEYS:
        if key in person_dict:
            flat = np.array(person_dict[key], dtype=np.float32)
            if flat.size > 0 and flat.size % 3 == 0:
                result[key] = flat.reshape(-1, 3)  # (N_joints, 3)
            else:
                result[key] = flat
    return result


def inspect_file(path):
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print(f"Size: {os.path.getsize(path):,} bytes")
    print(f"{'='*60}")

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # ── Top-level summary ────────────────────────────────────────────
    total_frames = len(data)
    empty_frames = sum(1 for f in data if len(f) == 0)
    nonempty = [(i, f) for i, f in enumerate(data) if len(f) > 0]
    person_counts = {}
    for _, frame in nonempty:
        n = len(frame)
        person_counts[n] = person_counts.get(n, 0) + 1

    print(f"\nTotal frames:     {total_frames}")
    print(f"Empty frames:     {empty_frames}")
    print(f"Non-empty frames: {len(nonempty)}")
    print(f"People per frame: {person_counts}  (usually 1 for TED talks)")

    # ── Dict keys present ────────────────────────────────────────────
    if nonempty:
        sample_person = nonempty[0][1][0]
        print(f"\nKeys in each person dict: {list(sample_person.keys())}")
        print(f"\nKeypoint counts:")
        for key in KEYPOINT_KEYS:
            if key in sample_person:
                n = len(sample_person[key]) // 3
                print(f"  {key:<30} {n} joints  ({n*3} values)")

    # ── First person in first non-empty frame ────────────────────────
    print(f"\n--- First non-empty frame (index {nonempty[0][0]}) ---")
    person = parse_person(nonempty[0][1][0])

    for key, arr in person.items():
        if arr.ndim != 2:
            print(f"\n{key}: shape={arr.shape} (unexpected)")
            continue

        n_joints = arr.shape[0]
        names = BODY25_NAMES if key == "pose_keypoints" else [f"joint_{i}" for i in range(n_joints)]
        detected = np.sum(arr[:, 2] > 0.01)

        print(f"\n{key}  ({n_joints} joints, {detected} detected with conf > 0.01):")
        print(f"  {'name':<14} {'x':>8}  {'y':>8}  {'conf':>8}")
        print(f"  {'-'*44}")
        for j in range(min(n_joints, 10)):
            x, y, c = arr[j]
            name = names[j] if j < len(names) else f"joint_{j}"
            marker = "" if c > 0.01 else "  ← not detected"
            print(f"  {name:<14} {x:8.2f}  {y:8.2f}  {c:8.5f}{marker}")
        if n_joints > 10:
            print(f"  ... ({n_joints - 10} more joints)")

    # ── Sequence stats for pose_keypoints ────────────────────────────
    print(f"\n--- Pose sequence statistics (across all non-empty frames) ---")
    wrist_r = []  # joint 4 = right wrist in body25
    neck = []     # joint 1 = neck
    for _, frame in nonempty:
        person_dict = frame[0]  # take first person
        if "pose_keypoints" not in person_dict:
            continue
        flat = np.array(person_dict["pose_keypoints"], dtype=np.float32)
        if flat.size < 25 * 3:
            continue
        joints = flat.reshape(-1, 3)
        if joints[4, 2] > 0.01:   # right wrist detected
            wrist_r.append(joints[4, :2])
        if joints[1, 2] > 0.01:   # neck detected
            neck.append(joints[1, :2])

    if len(wrist_r) > 1:
        wr = np.stack(wrist_r)
        motion = np.diff(wr, axis=0)
        energy = float(np.sqrt((motion**2).sum(axis=1)).mean())
        print(f"  Right wrist frames with data: {len(wrist_r)}")
        print(f"  Right wrist mean motion (energy): {energy:.3f} px/frame")
        print(f"  Right wrist x range: {wr[:,0].min():.1f} – {wr[:,0].max():.1f}")
        print(f"  Right wrist y range: {wr[:,1].min():.1f} – {wr[:,1].max():.1f}")

    if len(neck) > 1:
        nk = np.stack(neck)
        print(f"  Neck position (mean): x={nk[:,0].mean():.1f}  y={nk[:,1].mean():.1f}")


def main():
    p = argparse.ArgumentParser(description="Inspect .pickle files")
    p.add_argument("path", help="Path to a .pickle file or folder")
    p.add_argument("--all", action="store_true", help="Inspect all files in folder")
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = sorted([f for f in os.listdir(args.path) if f.endswith(".pickle")])
        print(f"Found {len(files)} .pickle files in {args.path}")
        targets = files if args.all else files[:1]
        for fname in targets:
            inspect_file(os.path.join(args.path, fname))
        if not args.all and len(files) > 1:
            print(f"\nTip: use --all to inspect every file.")
    else:
        inspect_file(args.path)


if __name__ == "__main__":
    main()
