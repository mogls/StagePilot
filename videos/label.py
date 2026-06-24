"""
label.py — simple CLI labeling tool for skeleton clips

Helps you efficiently label 250 clips with presentation scores.
Saves progress as you go so you can stop and resume anytime.

Usage:
    python label.py --data_dir ./data --output ./data/labels.csv

Controls:
    For each dimension, enter a score 0.0–1.0 (or press Enter to skip/use previous)
    Type 's' to skip a clip entirely
    Type 'q' to quit and save progress
    Type 'r' to redo the previous clip
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "gesture_variety",
    "body_openness",
    "movement_energy",
    "head_movement",
    "expressiveness",
]

# Short descriptions shown during labeling to keep scores consistent
DIMENSION_GUIDE = {
    "gesture_variety":  "Variety of hand/arm gestures (1=none, 5=some, 10=rich variety)",
    "body_openness":    "Open vs closed posture (1=very closed/arms crossed, 10=fully open)",
    "movement_energy":  "Physical energy/animation (1=still/stiff, 10=highly animated)",
    "head_movement":    "Head nods/turns/scanning (1=rigid, 10=natural varied movement)",
    "expressiveness":   "Overall face+body expressiveness (1=blank, 10=very expressive)",
}


def peek_clip(pkl_path):
    """Print basic info about a clip without needing video."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    total = len(data)
    nonempty = [f for f in data if len(f) > 0]
    n_frames = len(nonempty)

    # Count joints from first non-empty frame
    n_joints = "?"
    energy_hint = "n/a"
    if nonempty:
        person = nonempty[0][0]
        n_joints = sum(
            len(person.get(k, [])) // 3
            for k in ["pose_keypoints", "hand_left_keypoints", "hand_right_keypoints"]
        )

        # Right wrist energy: pose joint 4
        try:
            wrists = []
            for frame in nonempty:
                p = frame[0]
                pose = p.get("pose_keypoints", [])
                if len(pose) >= 15:
                    arr = np.array(pose, dtype=np.float32).reshape(-1, 3)
                    if arr[4, 2] > 0.01:
                        wrists.append(arr[4, :2])
            if len(wrists) > 1:
                wr = np.stack(wrists)
                motion = np.diff(wr, axis=0)
                energy = float(np.sqrt((motion**2).sum(axis=1)).mean())
                energy_hint = f"~{energy:.3f} px/frame"
        except Exception:
            pass

    print(f"  Total frames: {total} | With data: {n_frames} | Joints: {n_joints} | Wrist energy: {energy_hint}")


def load_existing(output_csv):
    """Load already-labeled and already-skipped clips so neither reappears."""
    done_set = set()
    rows = []

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        rows = df.to_dict("records")
        done_set.update(df["filename"].tolist())

    skipped_csv = output_csv.replace(".csv", "_skipped.csv")
    if os.path.exists(skipped_csv):
        sk = pd.read_csv(skipped_csv)
        done_set.update(sk["filename"].tolist())
        n = len(sk)
        print(f"Skipped clips on record: {n} (won't be shown again)")

    return done_set, rows


def save_skipped(fname, output_csv):
    """Append a filename to the skipped CSV."""
    skipped_csv = output_csv.replace(".csv", "_skipped.csv")
    file_exists = os.path.exists(skipped_csv)
    with open(skipped_csv, "a", newline="") as f:
        writer = __import__("csv").writer(f)
        if not file_exists:
            writer.writerow(["filename"])
        writer.writerow([fname])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data", help="Folder with .pickle files")
    p.add_argument("--output",   default="./data/labels.csv", help="Output CSV")
    p.add_argument("--limit",    default=None, type=int, help="Max clips to label in this session")
    p.add_argument("--index",    default=None, help="Path to clip_index.csv from split_pickle.py (optional)")
    args = p.parse_args()

    # Auto-detect clip_index.csv if not specified
    index_path = args.index
    if index_path is None:
        auto = os.path.join(args.data_dir, "clip_index.csv")
        if os.path.exists(auto):
            index_path = auto

    clip_index = {}
    if index_path and os.path.exists(index_path):
        idx_df = pd.read_csv(index_path)
        idx_df.columns = idx_df.columns.str.strip()
        for _, row in idx_df.iterrows():
            clip_index[row["filename"].strip()] = row
        print(f"Loaded clip index: {index_path} ({len(clip_index)} clips)")

    pkl_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".pickle")])
    print(f"\nFound {len(pkl_files)} .pickle files in {args.data_dir}")

    done_set, rows = load_existing(args.output)
    remaining = [f for f in pkl_files if f not in done_set]
    print(f"Already labeled: {len(done_set)} | Remaining: {len(remaining)}")

    if args.limit:
        remaining = remaining[:args.limit]

    print("\nScoring guide:")
    for dim, guide in DIMENSION_GUIDE.items():
        print(f"  {dim}: {guide}")

    print("\nControls: enter score 1-10 | Enter=keep last score | 's'=skip clip | 'q'=quit\n")
    print("=" * 60)

    last_scores = {col: 0.5 for col in SCORE_COLUMNS}

    for i, fname in enumerate(remaining):
        pkl_path = os.path.join(args.data_dir, fname)

        print(f"\n[{i+1}/{len(remaining)}] {fname}")
        if fname in clip_index:
            info = clip_index[fname]
            t_start = info.get("time_start", "?")
            t_end   = info.get("time_end",   "?")
            source  = info.get("source",     "?")
            det     = info.get("detected_rate", "?")
            print(f"  Source: {source}  |  {t_start} → {t_end}  |  Detection: {det}")
        try:
            peek_clip(pkl_path)
        except Exception as e:
            print(f"  (Could not read clip info: {e})")

        scores = {}
        skip_clip = False

        for col in SCORE_COLUMNS:
            while True:
                prompt = f"  {col} (last={last_scores[col] * 10:.0f}/10): "
                try:
                    raw = input(prompt).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    raw = "q"

                if raw == "q":
                    _save(rows, args.output)
                    print(f"\nSaved {len(rows)} labels to {args.output}")
                    return

                if raw == "s":
                    skip_clip = True
                    break

                if raw == "":
                    scores[col] = last_scores[col]
                    break

                try:
                    val = float(raw)
                    if 1 <= val <= 10:
                        val = round(val / 10, 1)
                        scores[col] = val
                        last_scores[col] = val
                        break
                    else:
                        print("    Please enter a value between 1 and 10")
                except ValueError:
                    print("    Invalid input. Enter a number 1-10, 's' to skip, or 'q' to quit.")

            if skip_clip:
                break

        if skip_clip:
            print(f"  Skipped {fname}")
            save_skipped(fname, args.output)
            continue

        row = {"filename": fname, **scores}
        rows.append(row)
        done_set.add(fname)

        if len(rows) % 10 == 0:
            _save(rows, args.output)
            print(f"  [Auto-saved {len(rows)} labels]")

    _save(rows, args.output)
    print(f"\nDone! {len(rows)} clips labeled and saved to {args.output}")

    if rows:
        df = pd.DataFrame(rows)
        available = [c for c in SCORE_COLUMNS if c in df.columns]
        if available:
            print("\nScore summary:")
            print(df[available].describe().round(3))
    else:
        print("No clips labeled in this session.")


def _save(rows, path):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
