"""
dataset.py — loads labeled skeleton pickle files for ST-GCN training

Pickle format (TED gesture dataset):
    List of frames. Each frame is a list of people (usually 1 for TED talks).
    Each person is a dict:
    {
        'pose_keypoints':       [x,y,c, x,y,c, ...]  # 25 joints × 3
        'face_keypoints':       [x,y,c, ...]           # 70 joints × 3
        'hand_left_keypoints':  [x,y,c, ...]           # 21 joints × 3
        'hand_right_keypoints': [x,y,c, ...]           # 21 joints × 3
    }
    Values are flat lists; reshape to (N_joints, 3) for (x, y, confidence).

Label file format (labels.csv):
    filename,       gesture_variety, body_openness, movement_energy, head_movement, expressiveness
    clip_001.pickle, 0.7,            0.6,           0.5,             0.4,           0.8
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCORE_COLUMNS = [
    "gesture_variety",
    "body_openness",
    "movement_energy",
    "head_movement",
    "expressiveness",
]

# Number of joints per keypoint group
N_POSE  = 25
N_FACE  = 70
N_HAND  = 21  # each hand

# Total joints if using all groups: 25 + 70 + 21 + 21 = 137
# Total joints if using pose only:  25
# Change USE_GROUPS to control which keypoints to include
USE_GROUPS = ["pose_keypoints", "hand_left_keypoints", "hand_right_keypoints"]
# USE_GROUPS = ["pose_keypoints"]  # lighter option


def extract_skeleton(frame_list, use_groups=USE_GROUPS):
    """
    Convert the raw pickle frame list into a (T, V, 3) numpy array.

    - Skips empty frames (no people detected)
    - Takes the first detected person per frame
    - Concatenates selected keypoint groups along the joint axis
    - Missing keypoint groups are zero-filled

    Returns:
        numpy array of shape (T_valid, V, 3)  where V = sum of joints in use_groups
    """
    joint_counts = {
        "pose_keypoints":       N_POSE,
        "face_keypoints":       N_FACE,
        "hand_left_keypoints":  N_HAND,
        "hand_right_keypoints": N_HAND,
    }
    V = sum(joint_counts[g] for g in use_groups)

    frames_out = []
    for frame in frame_list:
        if not frame:           # empty — no person detected
            continue

        person = frame[0]       # take first person only
        parts = []

        for group in use_groups:
            n = joint_counts[group]
            if group in person and len(person[group]) == n * 3:
                flat = np.array(person[group], dtype=np.float32)
                parts.append(flat.reshape(n, 3))
            else:
                parts.append(np.zeros((n, 3), dtype=np.float32))

        joints = np.concatenate(parts, axis=0)  # (V, 3)
        frames_out.append(joints)

    if not frames_out:
        return np.zeros((1, V, 3), dtype=np.float32)

    return np.stack(frames_out, axis=0)   # (T, V, 3)


class SkeletonDataset(Dataset):
    """
    Loads skeleton pickle files and their corresponding label scores.

    Args:
        data_dir:      folder containing your .pickle files
        labels_csv:    path to CSV file with filenames + scores
        target_frames: all clips are resampled to this many frames
        augment:       whether to apply data augmentation (training only)
        use_groups:    which keypoint groups to include
    """

    def __init__(self, data_dir, labels_csv, target_frames=150,
                 augment=False, use_groups=USE_GROUPS):
        self.data_dir     = data_dir
        self.target_frames = target_frames
        self.augment      = augment
        self.use_groups   = use_groups

        self.labels = pd.read_csv(labels_csv)
        self.labels.columns = self.labels.columns.str.strip()
        self.labels["filename"] = self.labels["filename"].str.strip()

        n_joints = sum({
            "pose_keypoints": N_POSE, "face_keypoints": N_FACE,
            "hand_left_keypoints": N_HAND, "hand_right_keypoints": N_HAND,
        }[g] for g in use_groups)

        print(f"Loaded {len(self.labels)} labeled clips")
        print(f"Keypoint groups: {use_groups}  ({n_joints} joints total)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        pkl_path = os.path.join(self.data_dir, row["filename"])

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        skeleton = extract_skeleton(raw, self.use_groups)  # (T, V, 3)
        skeleton = self._normalize(skeleton)
        skeleton = self._resample(skeleton)

        if self.augment:
            skeleton = self._augment(skeleton)

        # ST-GCN expects (C, T, V)
        tensor = torch.FloatTensor(skeleton).permute(2, 0, 1)
        scores = torch.FloatTensor([row[col] for col in SCORE_COLUMNS])
        return tensor, scores

    # ── Preprocessing ────────────────────────────────────────────────

    def _normalize(self, skeleton):
        """Zero-center on neck (pose joint 1) and scale by torso height."""
        coords = skeleton[:, :, :2].copy()

        # Neck = joint 1 in pose_keypoints (first group)
        origin = coords[:, 1:2, :]
        coords -= origin

        # Scale by torso: neck(1) → mid-hip(8)
        if skeleton.shape[1] > 8:
            neck_pos  = coords[:, 1, :]
            hip_pos   = coords[:, 8, :]
            torso_len = np.linalg.norm(neck_pos - hip_pos, axis=-1).mean()
            if torso_len > 1e-6:
                coords /= torso_len

        out = skeleton.copy()
        out[:, :, :2] = coords
        return out

    def _resample(self, skeleton):
        """Resample to exactly target_frames via linear interpolation."""
        T = skeleton.shape[0]
        if T == self.target_frames:
            return skeleton
        old_t = np.linspace(0, T - 1, T)
        new_t = np.linspace(0, T - 1, self.target_frames)
        out = np.zeros((self.target_frames, skeleton.shape[1], skeleton.shape[2]),
                       dtype=np.float32)
        for v in range(skeleton.shape[1]):
            for c in range(skeleton.shape[2]):
                out[:, v, c] = np.interp(new_t, old_t, skeleton[:, v, c])
        return out

    def _augment(self, skeleton):
        """Standard skeleton augmentations — ~6x effective dataset size."""
        T = skeleton.shape[0]

        # Random temporal crop (80–100%)
        ratio = np.random.uniform(0.8, 1.0)
        crop  = int(T * ratio)
        start = np.random.randint(0, T - crop + 1)
        skeleton = self._resample(skeleton[start:start + crop])

        # Horizontal flip (mirror x)
        if np.random.random() < 0.5:
            skeleton[:, :, 0] *= -1

        # Small joint noise
        skeleton[:, :, :2] += np.random.normal(0, 0.01, skeleton[:, :, :2].shape)

        return skeleton


def make_loaders(data_dir, labels_csv, batch_size=8, val_split=0.2,
                 target_frames=150, num_workers=0, use_groups=USE_GROUPS):
    """
    Creates train and validation DataLoaders with an 80/20 split.
    Returns: (train_loader, val_loader, n_joints, n_coords=3)
    """
    from torch.utils.data import random_split, DataLoader

    joint_counts = {
        "pose_keypoints": N_POSE, "face_keypoints": N_FACE,
        "hand_left_keypoints": N_HAND, "hand_right_keypoints": N_HAND,
    }
    n_joints = sum(joint_counts[g] for g in use_groups)
    n_coords = 3

    full = SkeletonDataset(data_dir, labels_csv,
                           target_frames=target_frames,
                           augment=False,
                           use_groups=use_groups)

    val_size   = int(len(full) * val_split)
    train_size = len(full) - val_size
    train_data, val_data = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_data.dataset.augment = True

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_data,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"Train: {train_size} clips | Val: {val_size} clips")
    print(f"Joints: {n_joints} | Coords: {n_coords}")
    return train_loader, val_loader, n_joints, n_coords
