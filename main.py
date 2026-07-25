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
    # Sobald video_analyse.py existiert, hier einhaengen (analog zu run_pitch).
    raise NotImplementedError("video_analyse.py ist noch nicht implementiert.")


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
        # Wenn video_analyse.py fehlt, wird schritt() den Fehler abfangen.
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
    print("  5. video_analyse.py                 (noch nicht implementiert)")
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
