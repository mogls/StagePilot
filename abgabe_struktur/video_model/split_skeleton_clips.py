#!/usr/bin/env python3
"""
split_skeleton_clips.py

Splits a whole-video skeleton .pickle (the output of extract_skeleton.py --
a list of per-frame "people" entries) into fixed-length clip .pickle files,
each covering a window of N seconds. Each output clip pickle has exactly the
same structure as the input (a list of per-frame people lists), so it's a
drop-in match for anything that already reads extract_skeleton.py's output
(e.g. verify_skeleton.py) -- just at clip length instead of full-video length.

Frame count per clip is derived from the video's fps, since the pickle
itself doesn't store fps. Pass --video to read fps directly from the source
video (recommended, avoids mismatches), or --fps to specify it manually if
you don't have the video handy.

By default, clips are non-overlapping (stride == clip length). Use
--stride-seconds to create overlapping clips (e.g. --clip-seconds 5
--stride-seconds 2.5 for 50%-overlapping 5s clips).

The last clip is dropped by default if it's shorter than a full clip
(use --keep-last to keep it, e.g. for labeling short trailing gestures).

A manifest.json is written alongside the clips, mapping each clip file to
its frame range and time range in the source video -- handy for going back
to label specific clips.

Usage:
    python split_skeleton_clips.py --pickle skeleton.pickle --video video.mp4 \
        --clip-seconds 5 --output-dir clips/

    python split_skeleton_clips.py --pickle skeleton.pickle --fps 29.97 \
        --clip-seconds 5 --stride-seconds 2.5 --output-dir clips/ --keep-last
"""

import argparse
import json
import os
import pickle

import cv2


def get_fps_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"Could not read a valid fps from {video_path}; pass --fps manually instead.")
    return fps


def split_clips(frames_data, fps, clip_seconds, stride_seconds, keep_last):
    frames_per_clip = round(fps * clip_seconds)
    stride_frames = round(fps * stride_seconds) if stride_seconds else frames_per_clip

    if frames_per_clip < 1:
        raise ValueError("clip_seconds is too short for the given fps (rounds to < 1 frame).")

    total_frames = len(frames_data)
    clips = []
    start = 0
    while start < total_frames:
        end = start + frames_per_clip
        clip = frames_data[start:end]
        is_full = len(clip) == frames_per_clip
        if not is_full and not keep_last:
            break
        clips.append({
            "start_frame": start,
            "end_frame": start + len(clip),
            "start_time_sec": start / fps,
            "end_time_sec": (start + len(clip)) / fps,
            "n_frames": len(clip),
            "frames": clip,
        })
        if end >= total_frames:
            break
        start += stride_frames

    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pickle", required=True, help="Input whole-video skeleton .pickle from extract_skeleton.py")
    parser.add_argument("--video", help="Source video, used to read fps (recommended)")
    parser.add_argument("--fps", type=float, help="Manually specify fps instead of --video")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=None,
                         help="Default: same as --clip-seconds (no overlap)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-last", action="store_true",
                         help="Keep the final clip even if it's shorter than a full clip")
    parser.add_argument("--prefix", default="clip", help="Filename prefix for output clip pickles")
    args = parser.parse_args()

    if not args.video and args.fps is None:
        parser.error("Provide either --video (to read fps) or --fps directly.")

    fps = get_fps_from_video(args.video) if args.video else args.fps

    with open(args.pickle, "rb") as f:
        frames_data = pickle.load(f)

    clips = split_clips(frames_data, fps, args.clip_seconds, args.stride_seconds, args.keep_last)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    for i, clip in enumerate(clips):
        filename = f"{args.prefix}_{i:04d}.pickle"
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "wb") as f:
            pickle.dump(clip["frames"], f)

        manifest.append({
            "file": filename,
            "clip_index": i,
            "start_frame": clip["start_frame"],
            "end_frame": clip["end_frame"],
            "n_frames": clip["n_frames"],
            "start_time_sec": round(clip["start_time_sec"], 3),
            "end_time_sec": round(clip["end_time_sec"], 3),
        })

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "source_pickle": os.path.abspath(args.pickle),
            "source_video": os.path.abspath(args.video) if args.video else None,
            "fps": fps,
            "clip_seconds": args.clip_seconds,
            "stride_seconds": args.stride_seconds or args.clip_seconds,
            "total_source_frames": len(frames_data),
            "n_clips": len(clips),
            "clips": manifest,
        }, f, indent=2)

    print(f"fps: {fps:.3f}, frames/clip: {round(fps*args.clip_seconds)}, "
          f"stride: {args.stride_seconds or args.clip_seconds}s")
    print(f"Source frames: {len(frames_data)} -> {len(clips)} clips written to {args.output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
