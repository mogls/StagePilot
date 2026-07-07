# Skeleton Presentation Scorer

ST-GCN based model that takes OpenPose skeleton pickle files and outputs
5 presentation quality scores per clip.

## Project structure

```
videos/
├── data/
│   ├── video_1/                    ← clips from each TED talk go here
│   │   ├── 05jJodDVJRQ_clip_0000.pickle
│   │   └── ...
│   ├── video_2/
│   ├── video_3/
│   ├── video_4/
│   ├── labels_3_4.csv              ← labeled scores for videos 3 & 4
│   └── labels_skipped_3_4.csv     ← clips marked as skipped
├── checkpoints/
│   └── best_model.pth              ← saved after training
├── dataset.py                      ← data loading + augmentation
├── model.py                        ← ST-GCN + LSTM architectures
├── train.py                        ← training loop
├── infer.py                        ← run trained model on new clips
├── label.py                        ← interactive labeling tool
├── split_pickle.py                 ← slice full talks into 5-second clips
├── inspect_pickle.py               ← diagnostic: print clip contents
├── pickle_file_extractor.py        ← convert OpenPose JSON → .pickle
└── README.md
```

## Setup

```bash
pip install torch torchvision numpy pandas
```

## Full pipeline

```
Raw .pickle files (full TED talks)
         ↓  pickle_file_extractor.py   (convert OpenPose JSON → .pickle)
         ↓  split_pickle.py            (slice into 5-second clips)
Short clips in data/video_N/
         ↓  inspect_pickle.py          (optional — diagnose clip contents)
         ↓  label.py
labels.csv  +  labels_skipped.csv
         ↓  train.py                   (uses dataset.py internally)
checkpoints/best_model.pth
         ↓  infer.py
JSON scores → fusion LLM
```

---

## Step 1 — (optional) Convert OpenPose JSON to .pickle

If you ran OpenPose and have a folder of `*_keypoints.json` files:

```bash
python pickle_file_extractor.py --input ./openpose_output/ --output ./data/my_talk.pickle
```

---

## Step 2 — Split full talks into 5-second clips

```bash
# Single file, trim intro/outro (confirmed 25fps for TED data)
python split_pickle.py \
    --input ./data/my_talk.pickle \
    --output ./data/video_5/ \
    --start 0:30 --end 16:30 --fps 25

# All .pickle files in a folder
python split_pickle.py --input ./data/ --output ./data/clips/ --fps 25
```

---

## Step 3 — (optional) Inspect a clip

```bash
python inspect_pickle.py ./data/video_1/05jJodDVJRQ_clip_0000.pickle
```

---

## Step 4 — Label clips

`label.py` scans `--data_dir` **recursively**, so you can point it at the
top-level `data/` folder even when clips are stored in sub-directories.
Pre-labeled clips in the existing CSV files are skipped automatically.

```bash
# Label everything under data/ — skips clips already in labels_3_4.csv
python label.py \
    --data_dir ./data/ \
    --output ./data/labels_3_4.csv
```

For each clip you'll be prompted for 5 dimensions on a **1–10** scale
(stored as 0.0–1.0 in the CSV):

| Dimension        | 1                          | 10                     |
|------------------|----------------------------|------------------------|
| gesture_variety  | No gestures                | Rich, varied gestures  |
| body_openness    | Arms crossed, closed       | Fully open posture     |
| movement_energy  | Completely still           | Highly animated        |
| head_movement    | Rigid, no movement         | Natural, varied        |
| expressiveness   | Blank, unexpressive        | Very expressive        |

Controls: `1–10` = score | `Enter` = reuse last score | `s` = skip | `q` = quit

Progress is auto-saved every 10 clips so you can stop and resume anytime.

---

## Step 5 — Train

`train.py` also accepts a top-level `data_dir` containing sub-directories.
Point `--labels` at the combined labels file.

**Start with the LSTM baseline** (runs in minutes, validates your data pipeline):

```bash
python train.py \
    --data_dir ./data/ \
    --labels ./data/labels_3_4.csv \
    --model lstm \
    --epochs 30
```

**Then train ST-GCN** (better results, takes longer):

```bash
python train.py \
    --data_dir ./data/ \
    --labels ./data/labels_3_4.csv \
    --model stgcn \
    --epochs 50
```

**With pretrained weights** (recommended — download from ST-GCN repo):

```bash
python train.py \
    --data_dir ./data/ \
    --labels ./data/labels_3_4.csv \
    --model stgcn \
    --pretrained stgcn_ntu_xview.pth \
    --freeze_epochs 10 \
    --epochs 50
```

Training output:
```
Epoch  1/50 | Train: 0.0842 | Val: 0.0791 | LR: 1.00e-03 | 12.3s
Epoch  5/50 | Train: 0.0634 | Val: 0.0598 | LR: 1.00e-03 | 11.8s
  Per-dimension val MSE and R²:
    gesture_variety        MSE=0.0412  R²=0.612
    body_openness          MSE=0.0318  R²=0.701
    ...
  ✓ New best model saved (val_loss=0.0598)
```

---

## Step 6 — Run inference on new clips

Score a single clip:

```bash
python infer.py \
    --checkpoint ./checkpoints/best_model.pth \
    --clip ./data/video_1/05jJodDVJRQ_clip_0042.pickle
```

Output:
```
PRESENTATION ANALYSIS — SKELETON MODULE
========================================

gesture_variety
  Score: 0.72  [██████████████░░░░░░]  (strong)
  Great gesture variety — you use a rich range of hand movements.

body_openness
  Score: 0.38  [███████░░░░░░░░░░░░░]  (needs work)
  Closed posture detected. Open your chest and keep arms uncrossed.

...

Raw scores (for fusion module):
{
  "gesture_variety": 0.72,
  "body_openness": 0.38,
  "movement_energy": 0.61,
  "head_movement": 0.54,
  "expressiveness": 0.69
}
```

Score a whole folder:

```bash
python infer.py \
    --checkpoint ./checkpoints/best_model.pth \
    --folder ./data/video_1/ \
    --output results.csv
```

---

## Downloading pretrained ST-GCN weights

The original ST-GCN pretrained on NTU RGB+D is available from the paper's GitHub:
https://github.com/yysijie/st-gcn

Or use the MMAction2 version (easier to work with):
https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/stgcn

Download the `stgcn_ntu60_xview` checkpoint and pass it as `--pretrained`.

---

## Tuning tips

- **Val loss not going down?** Try reducing `--lr` to `5e-4` or `1e-4`
- **Overfitting (train much lower than val)?** Increase `--dropout` to `0.6`, reduce `--epochs`
- **Underfitting (both losses high)?** More data needed, or reduce model complexity with LSTM
- **Low R² on a specific dimension?** That dimension may need clearer labeling criteria or more variance in your labeled examples
- **Using all 67 joints (pose + hands)?** The default `USE_GROUPS` in `dataset.py` already selects these. Do not pass `--n_joints 25` unless you want body-only mode.
