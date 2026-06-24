"""
split_pickle.py — slice TED talk .pickle files into exact 5-second clips

Usage:
    # Split from 0:30 to 15:00 at 30fps into 5-second clips:
    python split_pickle.py --input ./data/05jJodDVJRQ.pickle --output ./data/clips/ --start 0:30 --end 15:00

    # Specify a different fps (check your source video if unsure):
    python split_pickle.py --input ./data/05jJodDVJRQ.pickle --output ./data/clips/ --start 1:00 --end 20:00 --fps 25

    # Split all files in a folder (uses full duration of each):
    python split_pickle.py --input ./data/ --output ./data/clips/

    # Custom clip length (e.g. 10 seconds):
    python split_pickle.py --input ./data/05jJodDVJRQ.pickle --output ./data/clips/ --start 0:30 --end 15:00 --clip_seconds 10

Output:
    ./data/clips/05jJodDVJRQ_clip_0000.pickle   (frames for 0:30–0:35)
    ./data/clips/05jJodDVJRQ_clip_0001.pickle   (frames for 0:35–0:40)
    ...
    ./data/clips/clip_index.csv
"""

import os
import pickle
import argparse
import csv
import numpy as np


# ── Time helpers ─────────────────────────────────────────────────────────────

def parse_time(s):
    """
    Parse a time string into total seconds.
    Accepts: "1:30" (1 min 30 sec), "90" (90 seconds), "1:30:00" (1 hour 30 min)
    """
    s = str(s).strip()
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise ValueError(f"Cannot parse time '{s}'. Use format MM:SS (e.g. '1:30') or total seconds.")


def seconds_to_str(secs):
    """Format seconds as M:SS for display."""
    m = int(secs) // 60
    s = secs % 60
    return f"{m}:{s:05.2f}"


def has_pose_data(frame):
    """Return True if frame has a person with at least one confident pose joint."""
    if not frame:
        return False
    person = frame[0]
    pose = person.get("pose_keypoints", [])
    if len(pose) < 3:
        return False
    arr = np.array(pose[:75], dtype=np.float32).reshape(-1, 3)
    return bool(arr[:, 2].max() > 0.01)


# ── Core splitter ─────────────────────────────────────────────────────────────

def split_file(src_path, out_dir, fps=30.0, clip_seconds=5.0,
               start_time=None, end_time=None, min_detected=0.3):
    """
    Split one .pickle file into fixed-length clips.

    Args:
        src_path:      path to source .pickle
        out_dir:       folder to write clips into
        fps:           frames per second of the source video
        clip_seconds:  length of each clip in seconds
        start_time:    start of usable range in seconds (None = beginning)
        end_time:      end of usable range in seconds (None = end of file)
        min_detected:  min fraction of frames needing valid pose to keep clip

    Returns:
        list of clip metadata dicts
    """
    basename = os.path.splitext(os.path.basename(src_path))[0]
    clip_frames = round(fps * clip_seconds)  # exact frame count per clip

    print(f"\nLoading {os.path.basename(src_path)}...")
    with open(src_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    total_frames = len(data)
    total_seconds = total_frames / fps

    # Resolve start/end frame indices
    start_frame = round((start_time or 0.0) * fps)
    end_frame   = round((end_time   or total_seconds) * fps)
    start_frame = max(0, min(start_frame, total_frames))
    end_frame   = max(0, min(end_frame,   total_frames))

    usable_frames  = end_frame - start_frame
    usable_seconds = usable_frames / fps
    expected_clips = usable_frames // clip_frames

    print(f"  Total:   {total_frames:,} frames  ({seconds_to_str(total_seconds)})")
    print(f"  Range:   frame {start_frame:,} → {end_frame:,}  "
          f"({seconds_to_str(start_frame/fps)} → {seconds_to_str(end_frame/fps)})")
    print(f"  Usable:  {usable_frames:,} frames  ({usable_seconds:.1f}s)")
    print(f"  Clips:   {clip_seconds}s × {fps}fps = {clip_frames} frames/clip  "
          f"→ up to {expected_clips} clips")

    clips_meta = []
    clip_idx   = 0
    skipped    = 0

    pos = start_frame
    while pos + clip_frames <= end_frame:
        clip = data[pos : pos + clip_frames]

        # Quality check
        detected = sum(1 for f in clip if has_pose_data(f))
        det_rate = detected / clip_frames

        if det_rate < min_detected:
            skipped += 1
            pos += clip_frames
            continue

        # Timestamps for this clip
        t_start = pos / fps
        t_end   = (pos + clip_frames) / fps

        clip_name = f"{basename}_clip_{clip_idx:04d}.pickle"
        clip_path = os.path.join(out_dir, clip_name)
        with open(clip_path, "wb") as f:
            pickle.dump(clip, f)

        clips_meta.append({
            "filename":      clip_name,
            "source":        os.path.basename(src_path),
            "frame_start":   pos,
            "frame_end":     pos + clip_frames,
            "time_start":    seconds_to_str(t_start),
            "time_end":      seconds_to_str(t_end),
            "clip_seconds":  clip_seconds,
            "fps":           fps,
            "detected_rate": round(det_rate, 3),
        })

        clip_idx += 1
        pos += clip_frames

    print(f"  Saved:   {clip_idx} clips  |  Skipped: {skipped} (low detection)")
    return clips_meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Split TED gesture .pickle files into fixed-length clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5-second clips from 0:30 to 15:00 at 30fps
  python split_pickle.py --input talk.pickle --output ./clips/ --start 0:30 --end 15:00

  # 10-second clips, 25fps video
  python split_pickle.py --input talk.pickle --output ./clips/ --start 1:00 --end 20:00 --fps 25 --clip_seconds 10

  # Full file, default settings
  python split_pickle.py --input ./data/ --output ./clips/
        """
    )
    p.add_argument("--input",        required=True,
                   help="Source .pickle file or folder of .pickle files")
    p.add_argument("--output",       required=True,
                   help="Output folder for clip files")
    p.add_argument("--start",        default=None,
                   help="Start time MM:SS (default: beginning of file)")
    p.add_argument("--end",          default=None,
                   help="End time MM:SS (default: end of file)")
    p.add_argument("--fps",          default=30.0, type=float,
                   help="Frames per second of the source video (default: 30)")
    p.add_argument("--clip_seconds", default=5.0,  type=float,
                   help="Length of each clip in seconds (default: 5)")
    p.add_argument("--min_detected", default=0.3,  type=float,
                   help="Min fraction of frames needing pose data (default: 0.3)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    start_secs = parse_time(args.start) if args.start else None
    end_secs   = parse_time(args.end)   if args.end   else None

    # Gather source files
    if os.path.isdir(args.input):
        sources = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".pickle") and "_clip_" not in f
        ])
        if not sources:
            print(f"No .pickle files found in {args.input}")
            return
    else:
        sources = [args.input]

    print(f"Source files: {len(sources)}")
    print(f"Clip length:  {args.clip_seconds}s at {args.fps}fps "
          f"= {round(args.fps * args.clip_seconds)} frames/clip")
    if start_secs is not None or end_secs is not None:
        s = seconds_to_str(start_secs) if start_secs else "start"
        e = seconds_to_str(end_secs)   if end_secs   else "end"
        print(f"Time range:   {s} → {e}")

    all_meta = []
    for src in sources:
        meta = split_file(
            src, args.output,
            fps=args.fps,
            clip_seconds=args.clip_seconds,
            start_time=start_secs,
            end_time=end_secs,
            min_detected=args.min_detected,
        )
        all_meta.extend(meta)

    # Write index CSV
    index_path = os.path.join(args.output, "clip_index.csv")
    if all_meta:
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_meta[0].keys())
            writer.writeheader()
            writer.writerows(all_meta)

    print(f"\n{'='*50}")
    print(f"Total clips saved: {len(all_meta)}")
    print(f"Index written to:  {index_path}")
    print(f"\nNext step:")
    print(f"  python label.py --data_dir {args.output} --output {args.output}/labels.csv")


if __name__ == "__main__":
    main()
