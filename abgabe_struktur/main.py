"""main.py
==============================================================================
Orchestriert die komplette präsentation_ai-Pipeline (Planungs-Abschnitt 13).

Ablauf:
    1. transcribe.py                (Whisper)
    2. inhalt_analyse.py            (spaCy + Zero-Shot)
    3. Parallel-Gruppe A (Text-basiert, in fester Reihenfolge):
         a. pausen_analyse.py       — MUSS zuerst laufen (v2, 13.2)
         b. sprechfluss_analyse.py  — konsumiert Stocker informativ aus (a)
         c. sprechtempo_analyse.py
         d. füllwörter_analyse_v2.py
    4. Parallel-Gruppe B (Audio-basiert, teilen Audio im Speicher):
         a. lautstaerke_analyse.py
         b. pitch_variation_analyse.py
         c. emotionale_variation_analyse.py
    5. video_analyse.py             (MediaPipe + DeepFace) — falls vorhanden
    6. gesamtscore.py               (Aggregation + 5 Konsistenz-Checks)

Fehlerbehandlung:
    Wenn ein Modul crasht, stoppt die Pipeline NICHT. Der Fehler wird
    geloggt, das Modul in gesamtscore.py als "nicht bewertet" markiert.

Aufruf:
    python main.py <video_pfad>
    python main.py <video_pfad> --skip-transcribe   # wenn Transkript existiert
    python main.py <video_pfad> --skip-video         # kein Video-Modul
    python main.py --dry-run                         # nur die Pipeline zeigen
=============================================================================="""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ============================================================================
# PFADE
# ============================================================================

PROJEKT_ROOT = Path(__file__).resolve().parent
ZWISCHEN_OUTPUT = PROJEKT_ROOT / "zwischen_output"
REPORTS_ROOT = PROJEKT_ROOT / "reports"
TRANSKRIPTE_DIR = PROJEKT_ROOT / "Transkripte"

# Standard-Dateinamen
TRANSKRIPT_SUFFIX = "_transkript.txt"

MODUL_OUTPUTS = {
    "inhalt":           ZWISCHEN_OUTPUT / "inhalt_analyse_output.json",
    "pausen":           ZWISCHEN_OUTPUT / "pausen_analyse_output.json",
    "sprechfluss":      ZWISCHEN_OUTPUT / "sprechfluss_analyse_output.json",
    "sprechtempo":      ZWISCHEN_OUTPUT / "sprechtempo_analyse_output.json",
    "fuellwoerter":     ZWISCHEN_OUTPUT / "fuellwoerter_analyse_output.json",
    "lautstaerke":      ZWISCHEN_OUTPUT / "lautstaerke_analyse_output.json",
    "pitch_variation":  ZWISCHEN_OUTPUT / "pitch_variation_analyse_output.json",
    "emotion":          ZWISCHEN_OUTPUT / "emotionale_variation_analyse_output.json",
    "video":            ZWISCHEN_OUTPUT / "video_analyse_output.json",
    "gesamt":           ZWISCHEN_OUTPUT / "gesamtscore_output.json",
}


# ============================================================================
# PIPELINE-INFRASTRUKTUR
# ============================================================================

class PipelineErgebnis:
    """Fasst zusammen, was gelaufen ist und was nicht."""
    def __init__(self):
        self.erfolgreich = []
        self.gescheitert = {}
        self.uebersprungen = []
        self.startzeit = time.time()

    def ok(self, name: str, dauer: float):
        self.erfolgreich.append((name, dauer))

    def fail(self, name: str, err: Exception):
        self.gescheitert[name] = err

    def skip(self, name: str, grund: str):
        self.uebersprungen.append((name, grund))

    def zusammenfassung(self) -> str:
        total = time.time() - self.startzeit
        lines = [
            "",
            "=" * 70,
            f"PIPELINE-ZUSAMMENFASSUNG  ({total:.1f}s gesamt)",
            "=" * 70,
        ]
        for name, dauer in self.erfolgreich:
            lines.append(f"  [OK]   {name:30s} {dauer:6.1f}s")
        for name, grund in self.uebersprungen:
            lines.append(f"  [SKIP] {name:30s} ({grund})")
        for name, err in self.gescheitert.items():
            lines.append(f"  [FAIL] {name:30s} {type(err).__name__}: {err}")
        lines.append("=" * 70)
        return "\n".join(lines)


def schritt(name: str, ergebnis: PipelineErgebnis, fn: Callable[[], None]) -> bool:
    """
    Führt einen Pipeline-Schritt aus mit Timing, Logging, Fehler-Isolation.
    Returns True wenn erfolgreich.
    """
    print(f"\n{'=' * 70}\n[main] SCHRITT: {name}\n{'=' * 70}")
    t0 = time.time()
    try:
        fn()
        dauer = time.time() - t0
        ergebnis.ok(name, dauer)
        print(f"[main] {name} fertig in {dauer:.1f}s")
        return True
    except Exception as e:
        ergebnis.fail(name, e)
        print(f"[main][FAIL] {name} crashed:\n{traceback.format_exc()}")
        return False


# ============================================================================
# WRAPPER FÜR JEDES MODUL
# ============================================================================

def run_transcribe(video_pfad: Path) -> Path:
    """Ruft transcribe.py auf und liefert den Transkript-Pfad."""
    import transcribe  # lokales Modul
    words = transcribe.transcribe(str(video_pfad), model_name="small")
    TRANSKRIPTE_DIR.mkdir(parents=True, exist_ok=True)
    transkript_pfad = TRANSKRIPTE_DIR / (video_pfad.stem + TRANSKRIPT_SUFFIX)
    transcribe.schreibe_transkript(words, str(transkript_pfad))
    return transkript_pfad


def run_inhalt(transkript_pfad: Path) -> None:
    """
    inhalt_analyse.py hat i.d.R. eine main()-Funktion mit File-Dialog.
    Wir setzen die Umgebungsvariable und rufen die Hauptfunktion direkt auf.
    Fallback: subprocess-Aufruf.
    """
    _run_module_subprocess(
        "inhalt_analyse.py",
        # nimmt den Pfad als arg — inhalt_analyse akzeptiert sys.argv[1]
        args=[str(transkript_pfad)],
    )


def run_pausen(transkript_pfad: Path) -> None:
    # pausen_analyse.py hat eine haupt-Funktion mit einer Path-Signatur
    import importlib.util
    spec = importlib.util.spec_from_file_location("pausen", PROJEKT_ROOT / "pausen_analyse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.analyse_pausen(
        transkript_pfad=transkript_pfad,
        inhalt_pfad=MODUL_OUTPUTS["inhalt"],
        output_json_pfad=MODUL_OUTPUTS["pausen"],
        output_txt_pfad=REPORTS_ROOT / "pausen" / f"pausen_report_{ts()}.txt",
    )


def run_sprechfluss(transkript_pfad: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sprechfluss", PROJEKT_ROOT / "sprechfluss_analyse.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.analyse_sprechfluss(
        transkript_pfad=transkript_pfad,
        inhalt_pfad=MODUL_OUTPUTS["inhalt"],
        pausen_pfad=MODUL_OUTPUTS["pausen"],  # v2: Stocker informativ
        output_json_pfad=MODUL_OUTPUTS["sprechfluss"],
        output_txt_pfad=REPORTS_ROOT / "sprechfluss" / f"sprechfluss_report_{ts()}.txt",
    )


def run_sprechtempo(transkript_pfad: Path) -> None:
    _run_module_subprocess("sprechtempo_analyse.py", args=[str(transkript_pfad)])


def run_fuellwoerter(transkript_pfad: Path) -> None:
    # v2: lokalen Konsistenz-Check unterdruecken, läuft zentral in gesamtscore
    os.environ["PAI_SKIP_LOCAL_CONSISTENCY"] = "1"
    _run_module_subprocess("fuellwoerter_analyse_v2.py", args=[str(transkript_pfad)])


def run_lautstaerke(audio_pfad: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lautstaerke", PROJEKT_ROOT / "lautstaerke_analyse.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.analyse_lautstaerke(
        audio_pfad=audio_pfad,
        inhalt_pfad=MODUL_OUTPUTS["inhalt"],
        output_json_pfad=MODUL_OUTPUTS["lautstaerke"],
        output_txt_pfad=REPORTS_ROOT / "lautstaerke" / f"lautstaerke_report_{ts()}.txt",
    )


def run_pitch(audio_pfad: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pitch", PROJEKT_ROOT / "pitch_variation_analyse.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.analyse_pitch_variation(
        audio_pfad=audio_pfad,
        inhalt_pfad=MODUL_OUTPUTS["inhalt"],
        output_json_pfad=MODUL_OUTPUTS["pitch_variation"],
        output_txt_pfad=REPORTS_ROOT / "pitch" / f"pitch_report_{ts()}.txt",
    )


def run_emotion(audio_pfad: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "emotion", PROJEKT_ROOT / "emotionale_variation_analyse.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.analyse_emotionale_variation(
        audio_pfad=audio_pfad,
        inhalt_pfad=MODUL_OUTPUTS["inhalt"],
        output_json_pfad=MODUL_OUTPUTS["emotion"],
        output_txt_pfad=REPORTS_ROOT / "emotion" / f"emotion_report_{ts()}.txt",
    )


def run_video(video_pfad: Path) -> None:
    """
    Video-Analyse-Pipeline:
      1. extract_skeleton.py       — MediaPipe pose/hand/face -> .pickle
      2. split_skeleton_clips.py   — 5-Sekunden-Clips        -> clip_*.pickle
      3. infer.py                  — ST-GCN Scoring          -> scores pro Clip
      4. Aggregation               -> zwischen_output/video_analyse_output.json
    """
    import sys as _sys
    import json as _json
    import pickle as _pickle
    import importlib.util as _ilu

    # ── Pfade ────────────────────────────────────────────────────────────────
    VIDEO_MODEL_DIR = PROJEKT_ROOT / "video_model"
    CHECKPOINT      = VIDEO_MODEL_DIR / "model.pth"
    MP_MODELS_DIR   = VIDEO_MODEL_DIR / "mp_models"
    WORK_DIR        = ZWISCHEN_OUTPUT / "video_work"
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    skeleton_pkl = WORK_DIR / (video_pfad.stem + "_skeleton.pickle")
    clips_dir    = WORK_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Kein Modell-Checkpoint gefunden: {CHECKPOINT}\n"
            "Lege model.pth nach abgabe_struktur/video_model/model.pth."
        )

    def _load(name):
        path = VIDEO_MODEL_DIR / f"{name}.py"
        spec = _ilu.spec_from_file_location(name, path)
        mod  = _ilu.module_from_spec(spec)
        if str(VIDEO_MODEL_DIR) not in _sys.path:
            _sys.path.insert(0, str(VIDEO_MODEL_DIR))
        spec.loader.exec_module(mod)
        return mod

    # ── Schritt 1: Skeleton extrahieren ──────────────────────────────────────
    print(f"[video] Schritt 1/3: Skeleton extrahieren → {skeleton_pkl.name}")
    if skeleton_pkl.exists():
        print(f"[video] Skeleton-Pickle existiert, überspringe Extraktion.")
    else:
        extract_mod = _load("extract_skeleton")
        extract_mod.extract(
            video_path=str(video_pfad),
            output_path=str(skeleton_pkl),
            models_dir=str(MP_MODELS_DIR),
        )

    # ── Schritt 2: Clips splitten ─────────────────────────────────────────────
    print(f"[video] Schritt 2/3: Skeleton in 5-Sekunden-Clips aufteilen")
    split_mod = _load("split_skeleton_clips")

    # split_skeleton_clips.py kann fps direkt aus dem Video lesen
    fps = split_mod.get_fps_from_video(str(video_pfad))

    with open(skeleton_pkl, "rb") as _f:
        frames_data = _pickle.load(_f)

    clips = split_mod.split_clips(
        frames_data=frames_data,
        fps=fps,
        clip_seconds=5.0,
        stride_seconds=None,   # non-overlapping
        keep_last=False,
    )
    if not clips:
        raise RuntimeError(
            "[video] Keine Clips erzeugt — zu wenig Pose-Erkennung im Video."
        )

    # Clips auf Disk schreiben (split_clips() gibt frame-Listen zurück, kein I/O)
    clips_meta = []
    for i, clip in enumerate(clips):
        filename = f"clip_{i:04d}.pickle"
        out_path = clips_dir / filename
        with open(out_path, "wb") as _f:
            _pickle.dump(clip["frames"], _f)
        clips_meta.append({
            "file":           filename,
            "start_time_sec": clip["start_time_sec"],
            "end_time_sec":   clip["end_time_sec"],
        })
    print(f"[video] {len(clips_meta)} Clips erzeugt.")

    # ── Schritt 3: Inference ─────────────────────────────────────────────────
    print(f"[video] Schritt 3/3: ST-GCN Inference ({len(clips_meta)} Clips)")
    import torch as _torch
    infer_mod = _load("infer")

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model, cfg = infer_mod.load_model(str(CHECKPOINT), device)

    # extract_skeleton.py schreibt '*_keypoints_2d'-Schlüssel;
    # dataset.py::extract_skeleton() liest '*_keypoints' (ohne '_2d').
    # Wir normalisieren jeden Clip vor der Inference.
    import pickle as _pickle

    KEY_MAP = {
        "pose_keypoints_2d":       "pose_keypoints",
        "hand_left_keypoints_2d":  "hand_left_keypoints",
        "hand_right_keypoints_2d": "hand_right_keypoints",
        "face_keypoints_2d":       "face_keypoints",
    }

    def _normalise_keys(pkl_path):
        """Liest ein Clip-Pickle, normalisiert die Schlüsselnamen in-place."""
        with open(pkl_path, "rb") as _f:
            frames = _pickle.load(_f)
        changed = False
        for frame in frames:
            for person in frame:
                for old, new in KEY_MAP.items():
                    if old in person and new not in person:
                        person[new] = person.pop(old)
                        changed = True
        if changed:
            with open(pkl_path, "wb") as _f:
                _pickle.dump(frames, _f)

    clip_results = []
    for meta in clips_meta:
        pkl_path = clips_dir / meta["file"]
        _normalise_keys(str(pkl_path))
        try:
            scores   = infer_mod.score_clip(str(pkl_path), model, cfg, device)
            feedback = infer_mod.interpret_scores(scores)
            clip_results.append({
                "filename":   meta["file"],
                "time_start": meta["start_time_sec"],
                "time_end":   meta["end_time_sec"],
                "scores":     scores,
                "feedback":   feedback,
            })
        except Exception as e:
            print(f"[video] WARN: Clip {meta['filename']} fehlgeschlagen: {e}")

    if not clip_results:
        raise RuntimeError("[video] Kein Clip konnte gescort werden.")

    # ── Aggregation: Mittelwert über alle Clips ───────────────────────────────
    import statistics as _stats
    dims = list(clip_results[0]["scores"].keys())
    mean_scores = {
        dim: round(_stats.mean(c["scores"][dim] for c in clip_results), 4)
        for dim in dims
    }
    # Gesamtscore: Mittelwert der 5 Dimensionen, skaliert auf 0–100
    gesamtscore = round(sum(mean_scores.values()) / len(mean_scores) * 100, 2)

    output = {
        "modul":        "video_analyse",
        "version":      "1.0",
        "video":        str(video_pfad),
        "n_clips":      len(clip_results),
        "fps":          fps,
        "mean_scores":  mean_scores,
        "clip_results": clip_results,
        "scoring": {
            "gesamtscore": gesamtscore,
            "dimension_scores": {dim: round(v * 100, 2) for dim, v in mean_scores.items()},
        },
    }

    output_path = MODUL_OUTPUTS["video"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, ensure_ascii=False, indent=2)

    # Report
    report_path = REPORTS_ROOT / "video" / f"video_report_{ts()}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("VIDEO-ANALYSE — Skeleton + ST-GCN\n")
        f.write("=" * 60 + "\n")
        f.write(f"Video:         {video_pfad.name}\n")
        f.write(f"Clips:         {len(clip_results)}\n")
        f.write(f"Gesamtscore:   {gesamtscore:.1f} / 100\n\n")
        f.write("Dimensions (Mittelwert über alle Clips):\n")
        for dim, val in mean_scores.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            f.write(f"  {dim:20s}  {val:.3f}  [{bar}]\n")
        f.write("\nFeedback pro Clip:\n")
        for cr in clip_results:
            f.write(f"\n  [{cr['time_start']} – {cr['time_end']}] {cr['filename']}\n")
            for fb in cr["feedback"]:
                f.write(f"    {fb['dimension']:20s} {fb['score']:.2f}  ({fb['level']})\n")
                f.write(f"    → {fb['feedback']}\n")

    print(f"[video] Gesamtscore: {gesamtscore:.1f}/100")
    print(f"[video] JSON:   {output_path}")
    print(f"[video] Report: {report_path}")


def run_gesamtscore() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gesamtscore", PROJEKT_ROOT / "gesamtscore.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.aggregiere(
        input_dir=ZWISCHEN_OUTPUT,
        output_json=MODUL_OUTPUTS["gesamt"],
        output_txt=REPORTS_ROOT / "gesamt" / f"gesamt_report_{ts()}.txt",
    )


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_module_subprocess(script_name: str, args: list = None) -> None:
    """
    Ruft ein Modul als Subprocess auf. Für Module mit main()+File-Dialog
    einfacher als Import, weil sie sys.argv, tkinter etc. verwenden.
    """
    import subprocess
    cmd = [sys.executable, str(PROJEKT_ROOT / script_name)]
    if args:
        cmd += args
    print(f"[main] Subprocess: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJEKT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} exit code {result.returncode}")


def extrahiere_audio(video_pfad: Path) -> Path:
    """Zieht die Audio-Spur aus dem Video mit ffmpeg. Skippt wenn schon da."""
    audio_pfad = TRANSKRIPTE_DIR / (video_pfad.stem + ".wav")
    audio_pfad.parent.mkdir(parents=True, exist_ok=True)
    if audio_pfad.exists():
        print(f"[main] Audio existiert: {audio_pfad}")
        return audio_pfad
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", str(video_pfad),
        "-ac", "1", "-ar", "16000",
        "-vn", str(audio_pfad),
    ]
    print(f"[main] Extrahiere Audio: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg fehlgeschlagen:\n{result.stderr[:500]}")
    return audio_pfad


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="präsentation_ai Pipeline")
    parser.add_argument("video", nargs="?",
                        help="Pfad zum Video (oder Audio) der Präsentation")
    parser.add_argument("--transkript", help="Vorhandenes Transkript verwenden")
    parser.add_argument("--audio", help="Vorhandene Audio-Datei verwenden")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Transkription auslassen (--transkript setzen)")
    parser.add_argument("--skip-video", action="store_true",
                        help="Video-Analyse auslassen")
    parser.add_argument("--skip-emotion", action="store_true",
                        help="Emotion-Analyse auslassen (ML-Modell laedt ~3 GB)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur die Pipeline-Schritte anzeigen, nichts ausführen")
    args = parser.parse_args()

    if args.dry_run:
        pipeline_dry_run()
        return

    if not args.video and not args.transkript:
        print("[main] Bitte Video oder --transkript angeben.")
        parser.print_help()
        sys.exit(1)

    ZWISCHEN_OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    video_pfad = Path(args.video) if args.video else None
    ergebnis = PipelineErgebnis()

    # ------- Schritt 1: Transkription -------
    if args.transkript:
        transkript_pfad = Path(args.transkript)
        ergebnis.skip("transkription", "--transkript gesetzt")
    elif args.skip_transcribe:
        transkript_pfad = TRANSKRIPTE_DIR / (video_pfad.stem + TRANSKRIPT_SUFFIX)
        if not transkript_pfad.exists():
            print(f"[main][FAIL] --skip-transcribe aber Transkript fehlt: {transkript_pfad}")
            sys.exit(2)
        ergebnis.skip("transkription", "--skip-transcribe")
    else:
        schritt("transkription", ergebnis, lambda: run_transcribe(video_pfad))
        transkript_pfad = TRANSKRIPTE_DIR / (video_pfad.stem + TRANSKRIPT_SUFFIX)

    # ------- Schritt 2: Inhaltsanalyse -------
    schritt("inhalt_analyse", ergebnis, lambda: run_inhalt(transkript_pfad))

    # ------- Schritt 3: Gruppe A (Transkript-basiert) -------
    # Reihenfolge: pausen -> sprechfluss -> sprechtempo -> füllwörter
    # Für Iris Xe: sequentiell laufen lassen. Parallelisierung mit
    # concurrent.futures.ProcessPoolExecutor optional (siehe Kommentar unten).
    schritt("pausen_analyse", ergebnis, lambda: run_pausen(transkript_pfad))
    schritt("sprechfluss_analyse", ergebnis, lambda: run_sprechfluss(transkript_pfad))
    schritt("sprechtempo_analyse", ergebnis, lambda: run_sprechtempo(transkript_pfad))
    schritt("fuellwoerter_analyse", ergebnis, lambda: run_fuellwoerter(transkript_pfad))

    # ------- Schritt 4: Gruppe B (Audio) -------
    audio_pfad: Optional[Path] = None
    if args.audio:
        audio_pfad = Path(args.audio)
    elif video_pfad:
        schritt("audio_extraktion", ergebnis, lambda: setattr(main, "_audio", extrahiere_audio(video_pfad)))
        audio_pfad = getattr(main, "_audio", None)

    if audio_pfad and audio_pfad.exists():
        schritt("lautstaerke_analyse", ergebnis, lambda: run_lautstaerke(audio_pfad))
        schritt("pitch_variation_analyse", ergebnis, lambda: run_pitch(audio_pfad))
        if args.skip_emotion:
            ergebnis.skip("emotionale_variation", "--skip-emotion")
        else:
            schritt("emotionale_variation", ergebnis, lambda: run_emotion(audio_pfad))
    else:
        ergebnis.skip("audio_gruppe", "keine Audio-Datei verfügbar")

    # ------- Schritt 5: Video -------
    if args.skip_video:
        ergebnis.skip("video_analyse", "--skip-video")
    elif not video_pfad:
        ergebnis.skip("video_analyse", "kein Video angegeben")
    else:
        schritt("video_analyse", ergebnis, lambda: run_video(video_pfad))

    # ------- Schritt 6: Gesamt-Aggregation -------
    schritt("gesamtscore", ergebnis, run_gesamtscore)

    # ------- Abschluss -------
    print(ergebnis.zusammenfassung())


def pipeline_dry_run():
    print("PIPELINE (kein Lauf, nur Ablauf):\n")
    print("  1. transcribe.py                    -> Transkripte/<name>_transkript.txt")
    print("  2. inhalt_analyse.py                -> zwischen_output/inhalt_analyse_output.json")
    print("  3. Gruppe A (Text):")
    print("       a. pausen_analyse.py           -> zwischen_output/pausen_analyse_output.json")
    print("       b. sprechfluss_analyse.py      -> zwischen_output/sprechfluss_analyse_output.json")
    print("       c. sprechtempo_analyse.py      -> zwischen_output/sprechtempo_analyse_output.json")
    print("       d. fuellwoerter_analyse_v2.py  -> zwischen_output/fuellwoerter_analyse_output.json")
    print("  4. Gruppe B (Audio):")
    print("       a. lautstaerke_analyse.py")
    print("       b. pitch_variation_analyse.py")
    print("       c. emotionale_variation_analyse.py")
    print("  5. video_analyse (run_video):")
    print("       a. extract_skeleton.py         -> zwischen_output/video_work/<name>_skeleton.pickle")
    print("       b. split_pickle.py             -> zwischen_output/video_work/clips/")
    print("       c. infer.py (best_model.pth)   -> zwischen_output/video_analyse_output.json")
    print("  6. gesamtscore.py                   -> reports/gesamt/gesamt_report_*.txt")


# ----------------------------------------------------------------------------
# OPTIONAL: Parallelisierung von Gruppe A
# ----------------------------------------------------------------------------
# Auf Iris Xe bringt es ca. 60 % Zeitersparnis. WICHTIG:
# pausen MUSS zuerst fertig sein bevor sprechfluss startet
# (sprechfluss liest pausen_analyse_output.json). Daher:
#
#   from concurrent.futures import ProcessPoolExecutor
#   schritt("pausen_analyse", ergebnis, lambda: run_pausen(transkript_pfad))
#   with ProcessPoolExecutor(max_workers=3) as ex:
#       f1 = ex.submit(run_sprechfluss,  transkript_pfad)
#       f2 = ex.submit(run_sprechtempo,  transkript_pfad)
#       f3 = ex.submit(run_fuellwoerter, transkript_pfad)
#       for name, fut in [("sprechfluss", f1), ("sprechtempo", f2),
#                         ("füllwörter", f3)]:
#           try:  fut.result();  ergebnis.ok(name, 0)
#           except Exception as e: ergebnis.fail(name, e)
#
# Gruppe B sollte NICHT parallelisiert werden (Audio-Daten teilen im
# Speicher, dreifache Ladezeit sonst).


if __name__ == "__main__":
    main()
