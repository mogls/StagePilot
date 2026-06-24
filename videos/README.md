# Skeleton Presentation Scorer

ST-GCN based model that takes OpenPose skeleton pickle files and outputs
5 presentation quality scores per clip.

## Project structure

```
skeleton_trainer/
├── data/
│   ├── clip_001.pkl        ← your TED gesture dataset pickles go here
│   ├── clip_002.pkl
│   └── labels.csv          ← your labeled scores go here
├── checkpoints/
│   └── best_model.pth      ← saved after training
├── dataset.py              ← data loading + augmentation
├── model.py                ← ST-GCN + LSTM architectures
├── train.py                ← training loop
├── infer.py                ← run trained model on new clips
├── label.py                ← interactive labeling tool
└── README.md
```

## Setup

```bash
pip install torch torchvision numpy pandas
```

## Step 1 — Put your pickle files in ./data/

Copy your TED gesture dataset .pkl files into the `data/` folder.

## Step 2 — Label ~250 clips

Run the interactive labeling tool:

```bash
python label.py --data_dir ./data --output ./data/labels.csv
```

For each clip you'll be asked to score 5 dimensions from 0.0 to 1.0:

| Dimension        | 0.0                        | 0.5              | 1.0                    |
|------------------|----------------------------|------------------|------------------------|
| gesture_variety  | No gestures                | Some variety     | Rich, varied gestures  |
| body_openness    | Arms crossed, closed       | Mixed            | Fully open posture     |
| movement_energy  | Completely still           | Some movement    | Highly animated        |
| head_movement    | Rigid, no movement         | Some nodding     | Natural, varied        |
| expressiveness   | Blank, unexpressive        | Moderate         | Very expressive        |

Tips:
- Label in batches of 50, take breaks to stay calibrated
- Use the wrist motion hint printed for each clip as a rough energy guide
- Press Enter to reuse last score for a dimension (speeds things up)
- Progress is auto-saved every 10 clips

## Step 3 — Train

**Start with the LSTM baseline** (runs in minutes, validates your data pipeline):

```bash
python train.py --data_dir ./data --labels ./data/labels.csv --model lstm --epochs 30
```

**Then train ST-GCN** (better results, takes longer):

```bash
python train.py --data_dir ./data --labels ./data/labels.csv --model stgcn --epochs 50
```

**With pretrained weights** (recommended — download from ST-GCN repo):

```bash
python train.py \
  --data_dir ./data \
  --labels ./data/labels.csv \
  --model stgcn \
  --pretrained stgcn_ntu_xview.pth \
  --freeze_epochs 10 \
  --epochs 50
```

Training output looks like:
```
Epoch  1/50 | Train: 0.0842 | Val: 0.0791 | LR: 1.00e-03 | 12.3s
Epoch  5/50 | Train: 0.0634 | Val: 0.0598 | LR: 1.00e-03 | 11.8s
  Per-dimension val MSE and R²:
    gesture_variety        MSE=0.0412  R²=0.612
    body_openness          MSE=0.0318  R²=0.701
    ...
✓ New best model saved (val_loss=0.0598)
```

## Step 4 — Run inference on new clips

Score a single clip:

```bash
python infer.py --checkpoint ./checkpoints/best_model.pth --clip ./data/new_talk.pkl
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
python infer.py --checkpoint ./checkpoints/best_model.pth --folder ./new_clips/ --output results.csv
```

## Downloading pretrained ST-GCN weights

The original ST-GCN pretrained on NTU RGB+D is available from the paper's GitHub:
https://github.com/yysijie/st-gcn

Or use the MMAction2 version (easier to work with):
https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/stgcn

Download the `stgcn_ntu60_xview` checkpoint and pass it as `--pretrained`.

## Tuning tips

- **Val loss not going down?** Try reducing `--lr` to `5e-4` or `1e-4`
- **Overfitting (train much lower than val)?** Increase `--dropout` to `0.6`, reduce `--epochs`
- **Underfitting (both losses high)?** More data needed, or reduce model complexity with LSTM
- **Low R² on a specific dimension?** That dimension may need clearer labeling criteria or more variance in your labeled examples
