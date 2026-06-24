"""
infer.py — run the trained model on new skeleton pickle files

Usage:
    # Score a single clip:
    python infer.py --checkpoint ./checkpoints/best_model.pth --clip ./data/new_talk.pkl

    # Score a whole folder and save results to CSV:
    python infer.py --checkpoint ./checkpoints/best_model.pth --folder ./data/new_clips/
"""

import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch

from dataset import SkeletonDataset, SCORE_COLUMNS
from model import STGCN, SkeletonLSTM, body25_edges


def load_model(checkpoint_path, device):
    """Load the trained model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    if cfg["model"] == "stgcn":
        model = STGCN(
            in_channels=cfg["n_coords"],
            n_joints=cfg["n_joints"],
            n_outputs=cfg["n_outputs"],
            edges=body25_edges(),
            dropout=0.0,        # disable dropout at inference time
        )
    else:
        model = SkeletonLSTM(
            n_joints=cfg["n_joints"],
            n_coords=cfg["n_coords"],
            n_outputs=cfg["n_outputs"],
            dropout=0.0,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"Loaded {cfg['model'].upper()} model from epoch {ckpt['epoch']}")
    print(f"Val loss at save: {ckpt['val_loss']:.4f}")
    print(f"Scoring: {cfg['score_columns']}")

    return model, cfg


def score_clip(pkl_path, model, cfg, device):
    """
    Run inference on a single pickle file.
    Returns a dict of {dimension: score} values.
    """
    # Reuse the dataset's preprocessing logic
    dummy_dataset = SkeletonDataset.__new__(SkeletonDataset)
    dummy_dataset.target_frames = cfg["target_frames"]
    dummy_dataset.augment = False

    skeleton = dummy_dataset._load_pickle(pkl_path)
    skeleton = dummy_dataset._normalize(skeleton)
    skeleton = dummy_dataset._resample(skeleton)

    # (C, T, V) → add batch dim → (1, C, T, V)
    tensor = torch.FloatTensor(skeleton).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(tensor).squeeze(0).cpu().numpy()

    return {col: float(scores[i]) for i, col in enumerate(cfg["score_columns"])}


def interpret_scores(scores):
    """
    Convert raw scores to human-readable feedback.
    Simple threshold-based interpretation — you can make this more nuanced.
    """
    feedback = []
    for dim, score in scores.items():
        if score >= 0.7:
            level = "strong"
        elif score >= 0.4:
            level = "moderate"
        else:
            level = "needs work"

        tips = {
            "gesture_variety": {
                "strong":      "Great gesture variety — you use a rich range of hand movements.",
                "moderate":    "Some gesture variety. Try mixing pointing, open palm, and counting gestures.",
                "needs work":  "Limited gestures detected. Vary your hand movements to emphasize key points.",
            },
            "body_openness": {
                "strong":      "Open body language — arms away from body, facing audience well.",
                "moderate":    "Moderate openness. Avoid crossing arms and try facing forward more.",
                "needs work":  "Closed posture detected. Open your chest and keep arms uncrossed.",
            },
            "movement_energy": {
                "strong":      "Good energy in your movement — animated and engaged.",
                "moderate":    "Moderate movement. A bit more physical engagement could help.",
                "needs work":  "Low movement energy. Try using the stage and animating your delivery.",
            },
            "head_movement": {
                "strong":      "Natural head movement — good nodding and directional engagement.",
                "moderate":    "Some head movement. More variation would increase engagement.",
                "needs work":  "Little head movement. Try nodding to emphasize points and scanning the room.",
            },
            "expressiveness": {
                "strong":      "Highly expressive — face and body convey enthusiasm.",
                "moderate":    "Moderate expressiveness. More facial engagement would help.",
                "needs work":  "Low expressiveness. Let your face and body reflect your message.",
            },
        }

        tip = tips.get(dim, {}).get(level, f"{dim}: {level}")
        feedback.append({
            "dimension": dim,
            "score": round(score, 3),
            "level": level,
            "feedback": tip,
        })

    return feedback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--clip",   default=None, help="Single .pkl file to score")
    p.add_argument("--folder", default=None, help="Folder of .pkl files to score")
    p.add_argument("--output", default="results.csv", help="Output CSV for batch mode")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    # ---- Single clip ----
    if args.clip:
        print(f"\nScoring: {args.clip}")
        scores = score_clip(args.clip, model, cfg, device)
        feedback = interpret_scores(scores)

        print("\n" + "="*50)
        print("PRESENTATION ANALYSIS — SKELETON MODULE")
        print("="*50)
        for item in feedback:
            bar = "█" * int(item["score"] * 20) + "░" * (20 - int(item["score"] * 20))
            print(f"\n{item['dimension']}")
            print(f"  Score: {item['score']:.2f}  [{bar}]  ({item['level']})")
            print(f"  {item['feedback']}")

        print("\nRaw scores (for fusion module):")
        print(json.dumps(scores, indent=2))
        return scores

    # ---- Batch folder ----
    if args.folder:
        pkl_files = [f for f in os.listdir(args.folder) if f.endswith(".pickle")]
        print(f"\nScoring {len(pkl_files)} clips in {args.folder}...")

        rows = []
        for fname in pkl_files:
            path = os.path.join(args.folder, fname)
            try:
                scores = score_clip(path, model, cfg, device)
                scores["filename"] = fname
                rows.append(scores)
                print(f"  {fname}: {scores}")
            except Exception as e:
                print(f"  ERROR {fname}: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        print("\nSummary statistics:")
        print(df[SCORE_COLUMNS].describe().round(3))


if __name__ == "__main__":
    main()
