#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lautstaerke_analyse.py
======================
Bewertet Lautstärke-Variation und ob Kernbotschaften bewusst lauter gesprochen werden.

Input:
  - Audio-Datei (wav, mp3, etc. — alles was librosa lädt)
  - inhalt_analyse_output.json (für Segmente, Kernbotschaften, Struktur)

Output:
  - zwischen_output/lautstaerke_analyse_output.json
  - reports/lautstaerke/lautstaerke_report_[TIMESTAMP].txt"""

import json
import re
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

# Librosa wird dynamisch importiert (für bessere Fehlermeldung falls nicht installiert)
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa nicht installiert. Bitte installieren: pip install librosa soundfile")


# =============================================================================
# KONSTANTEN
# =============================================================================

SAMPLE_RATE = 16000          # 16 kHz Mono
HOP_LENGTH_MS = 10           # 10 ms Hop-Length
FRAME_LENGTH_MS = 20         # 20 ms Fenster

ROLLING_MEDIAN_S = 30        # 30 Sekunden Rolling Median für Baseline

# dB-Referenz (willkürlich, da wir nur relative Differenzen brauchen)
DB_REF = 1.0

# Scoring-Gewichtung
GEWICHT_D1 = 0.40
GEWICHT_D2 = 0.30
GEWICHT_D3 = 0.30


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class Segment:
    """Ein Zeitsegment mit Label (z.B. Kernbotschaft, Einleitung, etc.)."""
    label: str           # z.B. "kernbotschaft", "einleitung", "hauptteil", "schluss", "übergang"
    start_s: float
    end_s: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
        }


@dataclass
class LautstaerkeErgebnis:
    """Aggregiertes Ergebnis der Lautstärke-Analyse."""
    rms_db: np.ndarray           # RMS-Werte in dB pro Frame
    times_s: np.ndarray          # Zeitstempel pro Frame
    baseline_db: np.ndarray      # Rolling Median Baseline
    segments: List[Segment]

    d1_score: int
    d1_db_diff: float
    d1_bewertung: str

    d2_score: int
    d2_std_db: float
    d2_bewertung: str

    d3_score: int
    d3_anteil_im_bereich: float
    d3_bewertung: str
    d3_segment_details: List[Dict]

    gesamtscore: int


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def zeitstr_to_s(zeit_str: str) -> float:
    """Parst Zeitstempel zu Sekunden."""
    zeit_str = zeit_str.strip()
    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", zeit_str):
        h, m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    if re.match(r"^\d{2}:\d{2}\.\d{3}$", zeit_str):
        m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(m) * 60 + int(s) + int(ms) / 1000.0
    # Fallback: direkt als Sekunden oder Millisekunden
    try:
        val = float(zeit_str)
        if val > 10000:  # Wahrscheinlich ms
            return val / 1000.0
        return val
    except ValueError:
        raise ValueError(f"Unbekanntes Zeitformat: {zeit_str}")


def s_to_zeitstr(sekunden: float) -> str:
    """Sekunden zu MM:SS.mmm."""
    sekunden = max(0, sekunden)
    m = int(sekunden // 60)
    s = int(sekunden % 60)
    ms = int((sekunden % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def lade_json(pfad: Path) -> Optional[Dict]:
    if not pfad.exists():
        return None
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Konnte {pfad} nicht laden: {e}")
        return None


# =============================================================================
# AUDIO-VERARBEITUNG
# =============================================================================

def lade_audio(audio_pfad: Path) -> Tuple[np.ndarray, int]:
    """Lädt Audio mit librosa, konvertiert zu 16kHz Mono."""
    if not HAS_LIBROSA:
        raise ImportError("librosa ist nicht installiert. Installieren mit: pip install librosa soundfile")

    print(f"[lautstaerke] Lade Audio: {audio_pfad.name}")
    y, sr = librosa.load(str(audio_pfad), sr=SAMPLE_RATE, mono=True)
    print(f"[lautstaerke] Audio geladen: {len(y)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz, Mono")
    return y, sr


def berechne_rms_db(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet RMS in dB.

    Returns:
        rms_db: RMS-Werte in dB pro Frame
        times_s: Zeitstempel in Sekunden pro Frame (Frame-Center)
    """
    hop_length = int(sr * HOP_LENGTH_MS / 1000)   # 10 ms in Samples
    frame_length = int(sr * FRAME_LENGTH_MS / 1000)  # 20 ms in Samples

    # RMS berechnen
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True
    )[0]  # Shape: (n_frames,)

    # In dB umrechnen
    # Vermeide log(0) durch kleinen Offset
    rms_safe = np.maximum(rms, 1e-10)
    rms_db = 20.0 * np.log10(rms_safe / DB_REF)

    # Zeitstempel (Frame-Center)
    times_s = librosa.frames_to_time(
        np.arange(len(rms_db)),
        sr=sr,
        hop_length=hop_length
    )

    return rms_db, times_s


def berechne_rolling_median(rms_db: np.ndarray, sr: int) -> np.ndarray:
    """
    Berechnet Rolling Median über 30 Sekunden als Baseline.
    Kompensiert Mikrofonabstand-Schwankungen.
    """
    hop_length = int(sr * HOP_LENGTH_MS / 1000)
    frames_pro_30s = int(30.0 * sr / hop_length)

    # Mindestens 1 Frame
    frames_pro_30s = max(frames_pro_30s, 1)

    # Rolling Median mit gleitendem Fenster
    baseline = np.zeros_like(rms_db)
    half_window = frames_pro_30s // 2

    for i in range(len(rms_db)):
        start = max(0, i - half_window)
        end = min(len(rms_db), i + half_window + 1)
        baseline[i] = np.median(rms_db[start:end])

    return baseline


# =============================================================================
# SEGMENT-EXTRAKTION
# =============================================================================

def extrahiere_segmente(inhalt_data: Optional[Dict]) -> List[Segment]:
    """
    Extrahiert Segmente aus der Inhaltsanalyse:
    - Kernbotschaften
    - Struktur-Segmente (Einleitung, Hauptteil, Schluss)
    - Übergänge zwischen Segmenten
    """
    segments = []

    if not inhalt_data:
        return segments

    # 1. Kernbotschaften
    for kb in inhalt_data.get("kernbotschaften", []):
        start = kb.get("start_ms", kb.get("start"))
        end = kb.get("end_ms", kb.get("end"))
        if isinstance(start, str):
            start = zeitstr_to_s(start)
        if isinstance(end, str):
            end = zeitstr_to_s(end)
        if start is not None and end is not None:
            segments.append(Segment("kernbotschaft", float(start) / 1000.0, float(end) / 1000.0))

    # 2. Struktur-Segmente
    struktur = inhalt_data.get("struktur", {})
    if isinstance(struktur, dict):
        for key in ["einleitung", "hauptteil", "schluss"]:
            if key in struktur:
                seg = struktur[key]
                start = seg.get("start_ms", seg.get("start"))
                end = seg.get("end_ms", seg.get("end"))
                if isinstance(start, str):
                    start = zeitstr_to_s(start)
                if isinstance(end, str):
                    end = zeitstr_to_s(end)
                if start is not None and end is not None:
                    segments.append(Segment(key, float(start) / 1000.0, float(end) / 1000.0))

        # 3. Übergänge (zwischen Segmenten)
        segment_list = []
        for key in ["einleitung", "hauptteil", "schluss"]:
            if key in struktur:
                seg = struktur[key]
                start = seg.get("start_ms", seg.get("start"))
                end = seg.get("end_ms", seg.get("end"))
                if isinstance(start, str):
                    start = zeitstr_to_s(start)
                if isinstance(end, str):
                    end = zeitstr_to_s(end)
                if start is not None and end is not None:
                    segment_list.append((float(start) / 1000.0, float(end) / 1000.0, key))

        segment_list.sort()
        for i in range(1, len(segment_list)):
            prev_end = segment_list[i - 1][1]
            curr_start = segment_list[i][0]
            if curr_start > prev_end:
                segments.append(Segment("uebergang", prev_end, curr_start))

    return segments


def extrahiere_nebensaetze(inhalt_data: Optional[Dict]) -> List[Segment]:
    """Extrahiert Nicht-Kernbotschaft-Segmente als 'Nebensätze' für die Baseline."""
    segments = []

    if not inhalt_data or "satzgrenzen" not in inhalt_data:
        return segments

    # Alle Sätze minus Kernbotschaften = Nebensätze
    saetze = inhalt_data["satzgrenzen"]
    kern_starts = set()
    kern_ends = set()

    for kb in inhalt_data.get("kernbotschaften", []):
        s = kb.get("start_ms", kb.get("start"))
        e = kb.get("end_ms", kb.get("end"))
        if isinstance(s, str):
            s = zeitstr_to_s(s)
        if isinstance(e, str):
            e = zeitstr_to_s(e)
        if s is not None and e is not None:
            kern_starts.add(float(s) / 1000.0)
            kern_ends.add(float(e) / 1000.0)

    for satz in saetze:
        start = satz.get("start_ms", satz.get("start"))
        end = satz.get("end_ms", satz.get("end"))
        if isinstance(start, str):
            start = zeitstr_to_s(start)
        if isinstance(end, str):
            end = zeitstr_to_s(end)
        if start is None or end is None:
            continue

        start_s = float(start) / 1000.0
        end_s = float(end) / 1000.0

        # Prüfe ob dieser Satz eine Kernbotschaft ist
        ist_kb = False
        for ks, ke in zip(kern_starts, kern_ends):
            if start_s <= ke and end_s >= ks:
                ist_kb = True
                break

        if not ist_kb:
            segments.append(Segment("nebensatz", start_s, end_s))

    return segments


# =============================================================================
# SEGMENT-ANALYSE
# =============================================================================

def maske_fuer_segment(rms_db: np.ndarray, times_s: np.ndarray, segment: Segment) -> np.ndarray:
    """Erzeugt Boolean-Maske für Frames innerhalb eines Segments."""
    return (times_s >= segment.start_s) & (times_s <= segment.end_s)


def mittlerer_db(rms_db: np.ndarray, maske: np.ndarray) -> Optional[float]:
    """Mittlerer dB-Wert für maskierte Frames."""
    werte = rms_db[maske]
    if len(werte) == 0:
        return None
    return float(np.mean(werte))


def std_db(rms_db: np.ndarray, maske: np.ndarray) -> Optional[float]:
    """Standardabweichung in dB für maskierte Frames."""
    werte = rms_db[maske]
    if len(werte) < 2:
        return None
    return float(np.std(werte, ddof=1))


# =============================================================================
# SCORING
# =============================================================================

def score_d1_kernbotschaftsbetonung(
    rms_db: np.ndarray,
    times_s: np.ndarray,
    kernbotschaften: List[Segment],
    nebensaetze: List[Segment]
) -> Tuple[int, float, str]:
    """
    D1: Kernbotschafts-Betonung (40%)
    dB-Differenz = RMS(Kernbotschaften) − RMS(Nebensätze)
    """
    # Mittlerer dB der Kernbotschaften
    kb_mask = np.zeros(len(rms_db), dtype=bool)
    for kb in kernbotschaften:
        kb_mask |= maske_fuer_segment(rms_db, times_s, kb)

    kb_mean = mittlerer_db(rms_db, kb_mask)

    # Mittlerer dB der Nebensätze
    ns_mask = np.zeros(len(rms_db), dtype=bool)
    for ns in nebensaetze:
        ns_mask |= maske_fuer_segment(rms_db, times_s, ns)

    # Fallback: Wenn keine Nebensätze definiert, nimm alles außer Kernbotschaften
    if not np.any(ns_mask):
        ns_mask = ~kb_mask

    ns_mean = mittlerer_db(rms_db, ns_mask)

    if kb_mean is None or ns_mean is None:
        return 50, 0.0, "Nicht genug Daten"

    db_diff = kb_mean - ns_mean

    if db_diff >= 3.0:
        punkte = 100
        bewertung = "Gut betont"
    elif db_diff >= 1.0:
        punkte = 75
        bewertung = "Leicht betont"
    elif db_diff >= -1.0:
        punkte = 50
        bewertung = "Nicht betont"
    else:
        punkte = 20
        bewertung = "Kernbotschaften leiser"

    return punkte, db_diff, bewertung


def score_d2_gesamtvariation(rms_db: np.ndarray) -> Tuple[int, float, str]:
    """
    D2: Gesamt-Variation (30%)
    Standardabweichung in dB über die gesamte Präsentation.
    """
    if len(rms_db) < 2:
        return 40, 0.0, "Nicht genug Daten"

    std = float(np.std(rms_db, ddof=1))

    if 3.0 <= std <= 6.0:
        punkte = 100
        bewertung = "Optimal"
    elif 6.0 < std <= 9.0:
        punkte = 85
        bewertung = "Etwas viel"
    elif 1.5 <= std < 3.0:
        punkte = 65
        bewertung = "Gering"
    elif std < 1.5:
        punkte = 40
        bewertung = "Monoton"
    else:  # > 9.0
        punkte = 40
        bewertung = "Chaotisch"

    return punkte, std, bewertung


def score_d3_strukturkonsistenz(
    rms_db: np.ndarray,
    times_s: np.ndarray,
    baseline_db: np.ndarray,
    segments: List[Segment],
    nebensaetze: Optional[List[Segment]] = None,
) -> Tuple[int, float, str, List[Dict]]:
    """
    D3: Struktur-Konsistenz (30%)
    Vergleicht Ist-Mittelwert pro Struktur-Segment mit erwartetem Bereich.

    Erwartete Bereiche (v2-konform, relativ zur Baseline):
    - Einleitung:    Baseline ± 2 dB,    Toleranz 1 dB
    - Hauptteil:     Baseline ± 1 dB,    Toleranz 0.5 dB (Fix: nicht 0)
    - Kernbotschaft: Baseline +3 bis +6 dB, Toleranz 1.5 dB
    - Übergang:      Baseline −1 bis +1 dB, Toleranz 1 dB
    - Schluss:       Baseline ± 3 dB,    Toleranz 1.5 dB

    Baseline (v2-Fix): Median aller Frames im Nebensatz-Zeitbereich.
    Nur wenn keine Nebensätze aus inhalt_analyse verfügbar sind, wird auf
    den globalen Median zurückgefallen.

    Scoring: >=80% -> 100, 60-80% -> 75, 40-60% -> 50, <40% -> 25
    """

    # v2-Fix: Baseline aus Nebensatz-Frames statt Global-Median
    baseline_source = "global_median"
    if nebensaetze:
        ns_mask = np.zeros(len(rms_db), dtype=bool)
        for ns in nebensaetze:
            ns_mask |= maske_fuer_segment(rms_db, times_s, ns)
        if np.any(ns_mask):
            baseline_median = float(np.median(rms_db[ns_mask]))
            baseline_source = "nebensaetze"
        else:
            baseline_median = float(np.median(rms_db))
    else:
        baseline_median = float(np.median(rms_db))

    # Erwartete Bereiche definieren (v2-Fix: hauptteil Toleranz 0.5 statt 0)
    erwartet = {
        "einleitung":     (baseline_median - 2.0, baseline_median + 2.0, 1.0),
        "hauptteil":      (baseline_median - 1.0, baseline_median + 1.0, 0.5),
        "nebensatz":      (baseline_median - 1.0, baseline_median + 1.0, 0.5),
        "kernbotschaft":  (baseline_median + 3.0, baseline_median + 6.0, 1.5),
        "uebergang":      (baseline_median - 1.0, baseline_median + 1.0, 1.0),
        "schluss":        (baseline_median - 3.0, baseline_median + 3.0, 1.5),
    }

    details = []
    im_bereich_count = 0
    gesamt_count = 0

    for seg in segments:
        if seg.label not in erwartet:
            continue

        maske = maske_fuer_segment(rms_db, times_s, seg)
        mean_val = mittlerer_db(rms_db, maske)

        if mean_val is None:
            continue

        min_exp, max_exp, toleranz = erwartet[seg.label]

        # Mit Toleranz prüfen
        in_range = (mean_val >= min_exp - toleranz) and (mean_val <= max_exp + toleranz)

        details.append({
            "label": seg.label,
            "start_s": round(seg.start_s, 3),
            "end_s": round(seg.end_s, 3),
            "ist_db": round(mean_val, 2),
            "erwartet_min": round(min_exp, 2),
            "erwartet_max": round(max_exp, 2),
            "toleranz": toleranz,
            "im_bereich": in_range,
        })

        gesamt_count += 1
        if in_range:
            im_bereich_count += 1

    if gesamt_count == 0:
        return 50, 0.0, "Keine Struktur-Segmente", details

    anteil = im_bereich_count / gesamt_count

    if anteil >= 0.80:
        punkte = 100
        bewertung = "Sehr konsistent"
    elif anteil >= 0.60:
        punkte = 75
        bewertung = "Gut"
    elif anteil >= 0.40:
        punkte = 50
        bewertung = "Verbesserbar"
    else:
        punkte = 25
        bewertung = "Inkonsistent"

    return punkte, anteil, bewertung, details


def berechne_gesamtscore(d1: int, d2: int, d3: int) -> int:
    score = d1 * GEWICHT_D1 + d2 * GEWICHT_D2 + d3 * GEWICHT_D3
    return int(round(score))


# =============================================================================
# REPORT
# =============================================================================

def generiere_report(
    ergebnis: LautstaerkeErgebnis,
    audio_name: str,
    audio_dauer_s: float
) -> str:

    lines = []
    lines.append("=" * 70)
    lines.append("LAUTSTÄRKE-ANALYSE REPORT")
    lines.append("=" * 70)
    lines.append(f"Quelle: {audio_name}")
    lines.append(f"Dauer: {audio_dauer_s:.2f}s ({audio_dauer_s/60:.2f} Min)")
    lines.append(f"Abtastrate: {SAMPLE_RATE} Hz, Mono")
    lines.append(f"Analyse-Fenster: {FRAME_LENGTH_MS} ms, Hop: {HOP_LENGTH_MS} ms")
    lines.append("")

    lines.append("-" * 70)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("-" * 70)
    lines.append(f"Gesamt-Score: {ergebnis.gesamtscore}/100")
    lines.append("")
    lines.append(f"  D1 Kernbotschafts-Betonung (40%): {ergebnis.d1_score}/100 — {ergebnis.d1_bewertung}")
    lines.append(f"      (ΔdB = {ergebnis.d1_db_diff:+.2f} dB)")
    lines.append(f"  D2 Gesamt-Variation        (30%): {ergebnis.d2_score}/100 — {ergebnis.d2_bewertung}")
    lines.append(f"      (Std = {ergebnis.d2_std_db:.2f} dB)")
    lines.append(f"  D3 Struktur-Konsistenz     (30%): {ergebnis.d3_score}/100 — {ergebnis.d3_bewertung}")
    lines.append(f"      ({ergebnis.d3_anteil_im_bereich:.1%} der Segmente im erwarteten Bereich)")
    lines.append("")

    # D3 Details
    if ergebnis.d3_segment_details:
        lines.append("-" * 70)
        lines.append("D3 STRUKTUR-SEGMENTE (Detail)")
        lines.append("-" * 70)
        lines.append(f"{'Segment':<18} {'Start':<10} {'Ende':<10} {'Ist dB':<10} {'Erwartet':<18} {'Status':<10}")
        lines.append("-" * 70)
        for d in ergebnis.d3_segment_details:
            status = "✅ OK" if d["im_bereich"] else "❌ Abweichung"
            erwartet = f"{d['erwartet_min']:+.1f} bis {d['erwartet_max']:+.1f}"
            lines.append(
                f"{d['label']:<18} {s_to_zeitstr(d['start_s']):<10} "
                f"{s_to_zeitstr(d['end_s']):<10} {d['ist_db']:+.2f} dB   "
                f"{erwartet:<18} {status}"
            )
        lines.append("")

    lines.append("-" * 70)
    lines.append("STUDIEN-REFERENZEN")
    lines.append("-" * 70)
    lines.append("• McAllister, Sundberg: Dynamikbereich guter Vortragender 6--12 dB")
    lines.append("• Hincks & Edlund (2009): Monoton-Schwelle Std < 3 dB")
    lines.append("• Toastmasters: Kernbotschafts-Betonung +3 bis +6 dB über Baseline")
    lines.append("• Empirisch: Chaotisch bei Std > 9 dB")
    lines.append("")
    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def analyse_lautstaerke(
    audio_pfad: Path,
    inhalt_pfad: Optional[Path] = None,
    output_json_pfad: Optional[Path] = None,
    output_txt_pfad: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Haupt-Einstiegspunkt für die Lautstärke-Analyse.

    Args:
        audio_pfad: Pfad zur Audio-Datei
        inhalt_pfad: Pfad zu inhalt_analyse_output.json
        output_json_pfad: Zielpfad für JSON
        output_txt_pfad: Zielpfad für TXT-Report
    """

    if not HAS_LIBROSA:
        raise ImportError(
            "librosa ist nicht installiert.\n"
            "Installieren mit: pip install librosa soundfile\n"
            "Oder: conda install -c conda-forge librosa"
        )

    print(f"[lautstaerke] Starte Analyse: {audio_pfad.name}")

    # 1. Audio laden
    y, sr = lade_audio(audio_pfad)
    audio_dauer_s = len(y) / sr

    # 2. RMS in dB berechnen
    rms_db, times_s = berechne_rms_db(y, sr)
    print(f"[lautstaerke] {len(rms_db)} Frames berechnet.")

    # 3. Rolling Median Baseline
    baseline_db = berechne_rolling_median(rms_db, sr)
    print(f"[lautstaerke] Rolling Median (30s) berechnet.")

    # 4. Inhaltsanalyse laden
    inhalt_data = lade_json(inhalt_pfad) if inhalt_pfad else None

    # 5. Segmente extrahieren
    segments = extrahiere_segmente(inhalt_data)
    nebensaetze = extrahiere_nebensaetze(inhalt_data)
    kernbotschaften = [s for s in segments if s.label == "kernbotschaft"]

    print(f"[lautstaerke] {len(segments)} Segmente extrahiert "
          f"({len(kernbotschaften)} KB, {len(nebensaetze)} Nebensätze).")

    # 6. Scoring
    d1_score, d1_diff, d1_text = score_d1_kernbotschaftsbetonung(
        rms_db, times_s, kernbotschaften, nebensaetze
    )
    d2_score, d2_std, d2_text = score_d2_gesamtvariation(rms_db)
    d3_score, d3_anteil, d3_text, d3_details = score_d3_strukturkonsistenz(
        rms_db, times_s, baseline_db, segments, nebensaetze=nebensaetze
    )
    gesamt_score = berechne_gesamtscore(d1_score, d2_score, d3_score)

    print(f"[lautstaerke] Scoring: D1={d1_score}, D2={d2_score}, D3={d3_score}, Gesamt={gesamt_score}")

    # 7. Ergebnis bauen
    ergebnis = LautstaerkeErgebnis(
        rms_db=rms_db,
        times_s=times_s,
        baseline_db=baseline_db,
        segments=segments,
        d1_score=d1_score,
        d1_db_diff=d1_diff,
        d1_bewertung=d1_text,
        d2_score=d2_score,
        d2_std_db=d2_std,
        d2_bewertung=d2_text,
        d3_score=d3_score,
        d3_anteil_im_bereich=d3_anteil,
        d3_bewertung=d3_text,
        d3_segment_details=d3_details,
        gesamtscore=gesamt_score
    )

    # 8. Output
    output_data = {
        "modul": "lautstaerke_analyse",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input": str(audio_pfad),
        "meta": {
            "audio_dauer_s": round(audio_dauer_s, 3),
            "sample_rate": SAMPLE_RATE,
            "frame_length_ms": FRAME_LENGTH_MS,
            "hop_length_ms": HOP_LENGTH_MS,
            "frames_anzahl": len(rms_db),
        },
        "statistiken": {
            "rms_db_mean": round(float(np.mean(rms_db)), 2),
            "rms_db_std": round(float(np.std(rms_db, ddof=1)), 2),
            "rms_db_min": round(float(np.min(rms_db)), 2),
            "rms_db_max": round(float(np.max(rms_db)), 2),
            "baseline_db_mean": round(float(np.mean(baseline_db)), 2),
        },
        "segmente": [s.to_dict() for s in segments],
        "scoring": {
            "d1_kernbotschaftsbetonung": {
                "gewichtung": GEWICHT_D1,
                "punkte": d1_score,
                "bewertung": d1_text,
                "db_differenz": round(d1_diff, 2)
            },
            "d2_gesamtvariation": {
                "gewichtung": GEWICHT_D2,
                "punkte": d2_score,
                "bewertung": d2_text,
                "std_db": round(d2_std, 2)
            },
            "d3_strukturkonsistenz": {
                "gewichtung": GEWICHT_D3,
                "punkte": d3_score,
                "bewertung": d3_text,
                "anteil_im_bereich": round(d3_anteil, 4),
                "segment_details": d3_details
            },
            "gesamtscore": gesamt_score
        }
    }

    if output_json_pfad:
        output_json_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_pfad, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"[lautstaerke] JSON gespeichert: {output_json_pfad}")

    if output_txt_pfad:
        output_txt_pfad.parent.mkdir(parents=True, exist_ok=True)
        report = generiere_report(ergebnis, audio_pfad.name, audio_dauer_s)
        with open(output_txt_pfad, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[lautstaerke] Report gespeichert: {output_txt_pfad}")

    print(f"[lautstaerke] Fertig. Gesamt-Score: {gesamt_score}/100")
    return output_data


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lautstärke-Analyse für Präsentationsbewertungs-AI")
    parser.add_argument("audio", type=str, help="Pfad zur Audio-Datei")
    parser.add_argument("--inhalt", type=str, default=None, help="Pfad zu inhalt_analyse_output.json")
    parser.add_argument("--output-json", type=str, default="zwischen_output/lautstaerke_analyse_output.json")
    parser.add_argument("--output-txt", type=str, default=None)

    args = parser.parse_args()

    audio = Path(args.audio)
    inhalt = Path(args.inhalt) if args.inhalt else None
    out_json = Path(args.output_json)

    if args.output_txt:
        out_txt = Path(args.output_txt)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt = Path("reports/lautstaerke") / f"lautstaerke_report_{ts}.txt"

    if not audio.exists():
        print(f"[FEHLER] Audio nicht gefunden: {audio}")
        exit(1)

    try:
        analyse_lautstaerke(audio, inhalt, out_json, out_txt)
    except Exception as e:
        print(f"[FEHLER] {e}")
        raise
