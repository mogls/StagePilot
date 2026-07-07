"""
pickle_file_extractor.py — helper to extract OpenPose JSON output into
the .pickle format consumed by the rest of the pipeline.

Expected input layout (OpenPose output directory):
    openpose_output/
        000000000000_keypoints.json
        000000000001_keypoints.json
        ...

Each JSON file contains one frame in OpenPose format:
    {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d":       [x,y,c, ...],  # 25 joints × 3
                "face_keypoints_2d":        [x,y,c, ...],  # 70 joints × 3
                "hand_left_keypoints_2d":   [x,y,c, ...],  # 21 joints × 3
                "hand_right_keypoints_2d":  [x,y,c, ...]   # 21 joints × 3
            },
            ...
        ]
    }

Output:
    A list of frames, where each frame is a list of person dicts:
    [
        [  # frame 0
            {
                "pose_keypoints":       [x,y,c, ...],
                "face_keypoints":       [x,y,c, ...],
                "hand_left_keypoints":  [x,y,c, ...],
                "hand_right_keypoints": [x,y,c, ...],
            },
            ...  # more people if detected
        ],
        [],   # frame 1 — empty (no one detected)
        ...
    ]

Usage:
    python pickle_file_extractor.py \\
        --input ./openpose_output/ \\
        --output ./data/my_talk.pickle

    # Batch-convert all sub-folders in a directory:
    python pickle_file_extractor.py \\
        --input ./openpose_runs/ \\
        --output ./data/ \\
        --batch
"""

import os
import json
import pickle
import argparse


# Mapping from OpenPose JSON key → our internal key
_KEY_MAP = {
    "pose_keypoints_2d":      "pose_keypoints",
    "face_keypoints_2d":      "face_keypoints",
    "hand_left_keypoints_2d": "hand_left_keypoints",
    "hand_right_keypoints_2d":"hand_right_keypoints",
}


def convert_person(op_person):
    """Convert one OpenPose person dict to our internal format."""
    out = {}
    for op_key, our_key in _KEY_MAP.items():
        if op_key in op_person:
            out[our_key] = op_person[op_key]
    return out


def extract_frames(json_dir):
    """
    Read all *_keypoints.json files in json_dir (sorted by name)
    and return a frame list in our pickle format.
    """
    json_files = sorted([
        f for f in os.listdir(json_dir) if f.endswith("_keypoints.json")
    ])

    if not json_files:
        raise ValueError(f"No *_keypoints.json files found in {json_dir}")

    frames = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        people = data.get("people", [])
        frame = [convert_person(p) for p in people]
        frames.append(frame)

    return frames


def convert_one(json_dir, out_path):
    """Convert a single OpenPose output directory to a .pickle file."""
    print(f"  {json_dir} → {out_path}")
    frames = extract_frames(json_dir)
    non_empty = sum(1 for f in frames if f)
    print(f"    {len(frames)} frames ({non_empty} with detected people)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(frames, f)


def main():
    p = argparse.ArgumentParser(
        description="Convert OpenPose JSON output directories to .pickle files"
    )
    p.add_argument("--input",  required=True,
                   help="OpenPose output directory (or parent directory if --batch)")
    p.add_argument("--output", required=True,
                   help="Output .pickle file path (or output folder if --batch)")
    p.add_argument("--batch", action="store_true",
                   help="Convert every sub-directory inside --input")
    args = p.parse_args()

    if args.batch:
        os.makedirs(args.output, exist_ok=True)
        subdirs = sorted([
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
        ])
        print(f"Batch mode: {len(subdirs)} sub-directories found in {args.input}")
        for subdir in subdirs:
            src = os.path.join(args.input, subdir)
            dst = os.path.join(args.output, subdir + ".pickle")
            try:
                convert_one(src, dst)
            except Exception as e:
                print(f"    ERROR: {e}")
    else:
        convert_one(args.input, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
