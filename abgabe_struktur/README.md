# präsentation_ai

Analyses a presentation video and scores it across multiple dimensions — speech content,
prosody (pauses, tempo, pitch, loudness, filler words), emotional variation, and body language.
Produces a 0–100 overall score with detailed per-module reports and actionable feedback.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | |
| **ffmpeg** on PATH | Used to extract audio from video files |

**Install ffmpeg:**
```
# Windows (winget)
winget install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt install ffmpeg
```

---

## Setup

Run these commands once, in order, from inside the `abgabe_struktur/` folder.

**1. Install Python packages**
```
pip install -r requirements.txt
```

**2. Install the spaCy German language model**

This cannot go in `requirements.txt` — it is installed separately:
```
python -m spacy download de_core_news_sm
```

**3. (First run only) MediaPipe model files**

`extract_skeleton.py` automatically downloads three small model files (~18 MB total)
from Google on the first run and caches them in `video_model/mp_models/`.
No action needed — just make sure you have internet access the first time you run
the pipeline with a video file.

---

## Running the pipeline

```
python main.py <path/to/video.mp4>
```

The pipeline runs all analysis modules in order and writes reports to `reports/`.

### Common flags

| Flag | Effect |
|---|---|
| `--skip-transcribe` | Skip Whisper transcription — reuse the transcript from a previous run |
| `--transkript <path>` | Use a specific existing transcript file instead of generating one |
| `--audio <path>` | Use a pre-extracted audio file instead of running ffmpeg |
| `--skip-video` | Skip the skeleton / body language module (saves ~2 min) |
| `--skip-emotion` | Skip the emotional variation module (its model loads ~3 GB) |
| `--dry-run` | Print the pipeline steps without running anything |

### Examples

```bash
# Full analysis
python main.py presentation.mp4

# Skip re-transcribing a video you already ran before
python main.py presentation.mp4 --skip-transcribe

# Fast run — skip the two heavy ML modules
python main.py presentation.mp4 --skip-video --skip-emotion

# Use an existing transcript (no video needed)
python main.py --transkript Transkripte/presentation_transkript.txt

# Preview pipeline steps without running
python main.py --dry-run
```

---

## Outputs

All outputs are created automatically inside `abgabe_struktur/`:

| Path | Contents |
|---|---|
| `Transkripte/` | Word-level transcript `.txt` files (one per video) |
| `zwischen_output/` | Intermediate JSON files — one per analysis module |
| `reports/` | Human-readable `.txt` reports, one sub-folder per module |
| `reports/gesamt/` | **Final overall report** with score, consistency hints, top-3 improvements |
| `zwischen_output/video_work/` | Skeleton pickle and 5-second clips (reused on re-runs) |

The most useful file after a run:
```
reports/gesamt/gesamt_report_<TIMESTAMP>.txt
```

---

## Modules

| File | What it does |
|---|---|
| `main.py` | Orchestrates the full pipeline |
| `transcribe.py` | Whisper speech-to-text with word-level timestamps |
| `inhalt_analyse.py` | spaCy + Zero-Shot AI: sentence structure, key messages, rhetorical moments, audience address, emotional tone |
| `pausen_analyse.py` | Detects and classifies pauses from timestamp gaps (8 categories) |
| `sprechfluss_analyse.py` | Detects word repetitions and sentence breaks (disfluencies) |
| `sprechtempo_analyse.py` | Measures syllables/second — overall tempo, variation, key message slowdown |
| `fuellwoerter_analyse_v2.py` | Counts 4 filler-word categories (hesitation sounds, hedges, modal particles, intensifiers) |
| `lautstaerke_analyse.py` | Loudness variation and key-message emphasis via librosa RMS |
| `pitch_variation_analyse.py` | Pitch (F0) variation in semitones and sentence-final contours |
| `emotionale_variation_analyse.py` | Emotion classification per audio segment via wav2vec2 |
| `gesamtscore.py` | Aggregates all module scores into a weighted 0–100 result with 5 consistency checks |
| `video_model/extract_skeleton.py` | MediaPipe pose + hand + face extraction → skeleton pickle |
| `video_model/split_skeleton_clips.py` | Splits skeleton pickle into 5-second clips |
| `video_model/infer.py` | ST-GCN inference: scores gesture variety, body openness, movement energy, head movement, expressiveness |

---

## Score weights

```
Inhalt / Sprache  25 %
  └─ Filler words        15 %
  └─ Speech fluency      10 %

Prosody           45 %
  └─ Pauses              11.25 %
  └─ Speech tempo         9 %
  └─ Pitch variation      9 %
  └─ Emotional variation  9 %
  └─ Loudness             6.75 %

Video / Body      30 %
  └─ Skeleton model      30 %
```

If a module fails or is skipped, its weight is redistributed to the remaining modules —
a partial score is always produced.
