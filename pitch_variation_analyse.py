#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pitch_variation_analyse.py
==========================
Misst Tonhöhen-Variation (F0) in Semitones.
Bewertet Endkonturen (Fragen steigend, Aussagen fallend) und ob
Kernbotschaften stärker moduliert sind.

Input:
  - Audio-Datei
  - inhalt_analyse_output.json (für Sätze, Kernbotschaften, Satzzeichen)

Output:
  - zwischen_output/pitch_variation_analyse_output.json
  - reports/pitch/pitch_report_[TIMESTAMP].txt"""

import json
import re
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa nicht installiert. pip install librosa soundfile")


# =============================================================================
# KONSTANTEN
# =============================================================================

SAMPLE_RATE = 16000
HOP_LENGTH_MS = 20           # 20 ms Hop-Length (wie im Dokument)

# Semitone-Schwellen
ST_MONOTON = 1.5             # SD < 1.5 ST = monoton
ST_OPTIMAL_MIN = 3.0
ST_OPTIMAL_MAX = 6.0
ST_AUFFAELLIG_MIN = 2.0
ST_AUFFAELLIG_MAX = 8.0
ST_CHAOTISCH_MIN = 10.0

# Endkontur
ENDKONTUR_DAUER_MS = 300     # Letzte 300 ms
ENDKONTUR_STEIGEND = 2.0     # ≥ +2 ST
ENDKONTUR_FALLEND = -2.0     # ≤ −2 ST

# Monoton-Passagen
MONOTON_PASSAGE_S = 8.0      # > 8 Sekunden
MONOTON_PASSAGE_ST = 1.5     # Variation < 1.5 ST

# D2 Edge Case
D2_MIN_SAETZE = 5            # Mindestens 5 Sätze nach 800ms-Filter
D2_INSUFFICIENT_SCORE = 70

# Scoring
GEWICHT_D1 = 0.40
GEWICHT_D2 = 0.30
GEWICHT_D3 = 0.30


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class SatzPitch:
    """Ein Satz mit zugeordneten Pitch-Daten."""
    index: int
    text: str
    start_s: float
    end_s: float
    ist_kernbotschaft: bool = False
    ist_frage: bool = False
    ist_aussage: bool = False
    dauer_ms: float = 0.0

    # Pitch-Daten (nur voiced Frames)
    f0_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    f0_semitones: np.ndarray = field(default_factory=lambda: np.array([]))
    times_s: np.ndarray = field(default_factory=lambda: np.array([]))

    # Endkontur
    endkontur_st: float = 0.0
    endkontur_korrekt: bool = False

    # Statistik
    mean_st: float = 0.0
    std_st: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text[:60] + "..." if len(self.text) > 60 else self.text,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "dauer_ms": round(self.dauer_ms, 1),
            "ist_kernbotschaft": self.ist_kernbotschaft,
            "ist_frage": self.ist_frage,
            "ist_aussage": self.ist_aussage,
            "endkontur_st": round(self.endkontur_st, 2),
            "endkontur_korrekt": self.endkontur_korrekt,
            "mean_st": round(self.mean_st, 2),
            "std_st": round(self.std_st, 2),
            "voiced_frames": len(self.f0_hz),
        }


@dataclass
class MonotonPassage:
    """Eine zusammenhängende monotone Passage > 8s."""
    start_s: float
    end_s: float
    dauer_s: float
    std_st: float

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "dauer_s": round(self.dauer_s, 2),
            "std_st": round(self.std_st, 2),
        }


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def zeitstr_to_s(zeit_str: str) -> float:
    zeit_str = zeit_str.strip()
    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", zeit_str):
        h, m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    if re.match(r"^\d{2}:\d{2}\.\d{3}$", zeit_str):
        m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(m) * 60 + int(s) + int(ms) / 1000.0
    try:
        val = float(zeit_str)
        return val / 1000.0 if val > 10000 else val
    except ValueError:
        raise ValueError(f"Unbekanntes Zeitformat: {zeit_str}")


def s_to_zeitstr(sekunden: float) -> str:
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
# AUDIO & PITCH
# =============================================================================

def lade_audio(audio_pfad: Path) -> Tuple[np.ndarray, int]:
    if not HAS_LIBROSA:
        raise ImportError("librosa nicht installiert. pip install librosa soundfile")
    print(f"[pitch] Lade Audio: {audio_pfad.name}")
    y, sr = librosa.load(str(audio_pfad), sr=SAMPLE_RATE, mono=True)
    print(f"[pitch] Audio: {len(y)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz")
    return y, sr


def berechne_f0(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F0-Extraktion mit librosa.pyin.

    Returns:
        f0: F0-Werte in Hz (0.0 für unvoiced)
        voiced_flag: Boolean-Array
        times_s: Zeitstempel pro Frame
    """
    hop_length = int(sr * HOP_LENGTH_MS / 1000)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),   # ~65 Hz
        fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
        sr=sr,
        hop_length=hop_length,
        frame_length=hop_length * 2      # Standard: 2× Hop
    )

    times_s = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    return f0, voiced_flag, times_s


def hz_to_semitones(f0_hz: np.ndarray, baseline_hz: float) -> np.ndarray:
    """Hz → Semitones relativ zur Baseline."""
    # Nur voiced Frames (f0 > 0)
    semitones = np.zeros_like(f0_hz)
    mask = f0_hz > 0
    semitones[mask] = 12.0 * np.log2(f0_hz[mask] / baseline_hz)
    return semitones


# =============================================================================
# SATZ-EXTRAKTION & ZUORDNUNG
# =============================================================================

def extrahiere_saetze(inhalt_data: Optional[Dict]) -> List[SatzPitch]:
    """Extrahiert Sätze aus Inhaltsanalyse mit Satzzeichen-Erkennung."""
    saetze = []
    if not inhalt_data or "satzgrenzen" not in inhalt_data:
        return saetze

    for i, raw in enumerate(inhalt_data["satzgrenzen"]):
        start = raw.get("start_ms", raw.get("start"))
        end = raw.get("end_ms", raw.get("end"))
        text = raw.get("text", "")

        if isinstance(start, str):
            start = zeitstr_to_s(start)
        if isinstance(end, str):
            end = zeitstr_to_s(end)
        if start is None or end is None:
            continue

        start_s = float(start) / 1000.0 if float(start) > 1000 else float(start)
        end_s = float(end) / 1000.0 if float(end) > 1000 else float(end)

        # Satzzeichen erkennen
        text_stripped = text.strip()
        ist_frage = text_stripped.endswith("?")
        ist_aussage = text_stripped.endswith((".", "!"))

        saetze.append(SatzPitch(
            index=i,
            text=text,
            start_s=start_s,
            end_s=end_s,
            ist_frage=ist_frage,
            ist_aussage=ist_aussage,
            dauer_ms=(end_s - start_s) * 1000.0
        ))

    return saetze


def markiere_kernbotschaften(saetze: List[SatzPitch], inhalt_data: Optional[Dict]) -> None:
    if not inhalt_data or "kernbotschaften" not in inhalt_data:
        return

    for kb in inhalt_data["kernbotschaften"]:
        kb_start = kb.get("start_ms", kb.get("start"))
        kb_end = kb.get("end_ms", kb.get("end"))
        if isinstance(kb_start, str):
            kb_start = zeitstr_to_s(kb_start)
        if isinstance(kb_end, str):
            kb_end = zeitstr_to_s(kb_end)
        if kb_start is None or kb_end is None:
            continue

        kb_start_s = float(kb_start) / 1000.0 if float(kb_start) > 1000 else float(kb_start)
        kb_end_s = float(kb_end) / 1000.0 if float(kb_end) > 1000 else float(kb_end)

        for s in saetze:
            if s.start_s <= kb_end_s and s.end_s >= kb_start_s:
                s.ist_kernbotschaft = True


def ordne_pitch_zu_saetzen(
    saetze: List[SatzPitch],
    f0_hz: np.ndarray,
    f0_st: np.ndarray,
    times_s: np.ndarray
) -> None:
    """Ordnet jedem Satz die Pitch-Frames zu, die in seinen Zeitbereich fallen."""
    for s in saetze:
        maske = (times_s >= s.start_s) & (times_s <= s.end_s) & (f0_hz > 0)
        s.f0_hz = f0_hz[maske].copy()
        s.f0_semitones = f0_st[maske].copy()
        s.times_s = times_s[maske].copy()

        if len(s.f0_semitones) > 0:
            s.mean_st = float(np.mean(s.f0_semitones))
            s.std_st = float(np.std(s.f0_semitones, ddof=1)) if len(s.f0_semitones) > 1 else 0.0


def berechne_endkontur(s: SatzPitch, f0_st: np.ndarray, times_s: np.ndarray, f0_hz: np.ndarray) -> None:
    """
    Berechnet die Endkontur eines Satzes (letzte 300 ms).
    Fix v2: Nur für Sätze ≥ 800 ms.
    """
    if s.dauer_ms < 800:
        s.endkontur_st = 0.0
        s.endkontur_korrekt = False
        return

    # Letzte 300 ms
    end_start = s.end_s - (ENDKONTUR_DAUER_MS / 1000.0)
    maske = (times_s >= end_start) & (times_s <= s.end_s) & (f0_hz > 0)

    end_st = f0_st[maske]
    if len(end_st) < 2:
        s.endkontur_st = 0.0
        s.endkontur_korrekt = False
        return

    # Lineare Regression über die letzten Frames
    x = np.arange(len(end_st))
    if len(x) < 2:
        s.endkontur_st = 0.0
        s.endkontur_korrekt = False
        return

    # Steigung berechnen (einfache Differenz erster/letzter Wert)
    # Alternative: lineare Regression
    slope = (end_st[-1] - end_st[0])  # Delta in ST über 300ms
    s.endkontur_st = float(slope)

    # Prüfe Korrektheit
    if s.ist_frage and slope >= ENDKONTUR_STEIGEND:
        s.endkontur_korrekt = True
    elif s.ist_aussage and slope <= ENDKONTUR_FALLEND:
        s.endkontur_korrekt = True
    else:
        s.endkontur_korrekt = False


# =============================================================================
# MONOTON-PASSAGEN WARNUNG
# =============================================================================

def finde_monoton_passagen(
    f0_st: np.ndarray,
    times_s: np.ndarray,
    f0_hz: np.ndarray
) -> List[MonotonPassage]:
    """
    Findet zusammenhängende Passagen > 8 Sekunden mit Variation < 1.5 ST.
    Gleitendes Fenster über die voiced Frames.
    """
    passagen = []

    # Nur voiced Frames
    voiced_mask = f0_hz > 0
    voiced_times = times_s[voiced_mask]
    voiced_st = f0_st[voiced_mask]

    if len(voiced_st) < 2:
        return passagen

    # Gleitendes Fenster: 8 Sekunden
    # Wir verwenden einen Index-basierten Ansatz
    i = 0
    while i < len(voiced_times):
        # Finde alle Frames innerhalb von 8s ab diesem Startpunkt
        start_t = voiced_times[i]
        end_t = start_t + MONOTON_PASSAGE_S

        j = i
        while j < len(voiced_times) and voiced_times[j] <= end_t:
            j += 1

        window_st = voiced_st[i:j]
        if len(window_st) > 1:
            std = float(np.std(window_st, ddof=1))
            if std < MONOTON_PASSAGE_ST:
                # Prüfe ob wir diese Passage verlängern können
                actual_end = voiced_times[j - 1] if j > 0 else start_t
                dauer = actual_end - start_t
                if dauer >= MONOTON_PASSAGE_S:
                    passagen.append(MonotonPassage(
                        start_s=start_t,
                        end_s=actual_end,
                        dauer_s=dauer,
                        std_st=std
                    ))

        i += 1

    # Überlappende Passagen zusammenfassen (nur die längste pro Region behalten)
    if not passagen:
        return passagen

    # Sortieren und deduplizieren
    passagen.sort(key=lambda p: p.start_s)
    bereinigt = [passagen[0]]
    for p in passagen[1:]:
        last = bereinigt[-1]
        if p.start_s <= last.end_s:
            # Überlappend: verlängere falls nötig
            if p.end_s > last.end_s:
                last.end_s = p.end_s
                last.dauer_s = last.end_s - last.start_s
        else:
            bereinigt.append(p)

    return bereinigt


# =============================================================================
# SCORING
# =============================================================================

def score_d1_gesamtvariation(f0_st: np.ndarray, f0_hz: np.ndarray) -> Tuple[int, float, str]:
    """
    D1: Gesamtvariation in Semitones (40%)
    Standardabweichung aller voiced Frames.
    """
    voiced_st = f0_st[f0_hz > 0]
    if len(voiced_st) < 2:
        return 20, 0.0, "Nicht genug voiced Frames"

    std = float(np.std(voiced_st, ddof=1))

    if ST_OPTIMAL_MIN <= std <= ST_OPTIMAL_MAX:
        punkte = 100
        bewertung = "Optimal (Vortrag)"
    elif (ST_AUFFAELLIG_MIN <= std < ST_OPTIMAL_MIN) or (ST_OPTIMAL_MAX < std <= ST_AUFFAELLIG_MAX):
        punkte = 75
        bewertung = "Akzeptabel"
    elif (ST_MONOTON <= std < ST_AUFFAELLIG_MIN) or (ST_AUFFAELLIG_MAX < std <= ST_CHAOTISCH_MIN):
        punkte = 40
        bewertung = "Auffällig"
    elif std < ST_MONOTON:
        punkte = 20
        bewertung = "Monoton"
    else:  # > 10
        punkte = 20
        bewertung = "Chaotisch"

    return punkte, std, bewertung


def score_d2_endkontur(saetze: List[SatzPitch]) -> Tuple[int, float, str, int, int]:
    """
    D2: End-Kontur-Korrektheit (30%)
    Fix v2: Nur Sätze ≥ 800 ms. Edge Case: < 5 Sätze → 70 Punkte (neutral).
    """
    # Filter: nur Sätze ≥ 800 ms
    gueltige = [s for s in saetze if s.dauer_ms >= 800 and (s.ist_frage or s.ist_aussage)]

    if len(gueltige) < D2_MIN_SAETZE:
        return D2_INSUFFICIENT_SCORE, 0.0, "Insufficient data (weniger als 5 Sätze >= 800ms)", len(gueltige), 0

    korrekt = sum(1 for s in gueltige if s.endkontur_korrekt)
    anteil = korrekt / len(gueltige) if gueltige else 0.0

    if anteil >= 0.80:
        punkte = 100
        bewertung = "Sehr gut"
    elif anteil >= 0.60:
        punkte = 75
        bewertung = "Gut"
    elif anteil >= 0.40:
        punkte = 50
        bewertung = "Verbesserbar"
    else:
        punkte = 25
        bewertung = "Problematisch"

    return punkte, anteil, bewertung, len(gueltige), korrekt


def score_d3_kernbotschafts_variation(saetze: List[SatzPitch], gesamt_std: float) -> Tuple[int, float, str]:
    """
    D3: Kernbotschafts-Variation (30%)
    Verhältnis: Kern-SD / Gesamt-SD
    """
    kern_saetze = [s for s in saetze if s.ist_kernbotschaft and len(s.f0_semitones) > 1]

    if not kern_saetze:
        return 75, 1.0, "Keine Kernbotschaften gefunden"

    # Gesamt-SD der Kernbotschaften (gewichtet nach Frame-Anzahl)
    kern_frames = np.concatenate([s.f0_semitones for s in kern_saetze])
    if len(kern_frames) < 2:
        return 75, 1.0, "Nicht genug Frames in Kernbotschaften"

    kern_std = float(np.std(kern_frames, ddof=1))

    if gesamt_std <= 0:
        return 75, 1.0, "Gesamt-Std = 0"

    verhaeltnis = kern_std / gesamt_std

    if verhaeltnis >= 1.2:
        punkte = 100
        bewertung = "Kernbotschaften stärker moduliert"
    elif verhaeltnis >= 0.9:
        punkte = 75
        bewertung = "Ähnlich wie Rest"
    else:
        punkte = 40
        bewertung = "Kernbotschaften unter-moduliert"

    return punkte, verhaeltnis, bewertung


def berechne_gesamtscore(d1: int, d2: int, d3: int) -> int:
    score = d1 * GEWICHT_D1 + d2 * GEWICHT_D2 + d3 * GEWICHT_D3
    return int(round(score))


# =============================================================================
# REPORT
# =============================================================================

def generiere_report(
    d1_score: int, d1_std: float, d1_text: str,
    d2_score: int, d2_anteil: float, d2_text: str,
    d2_gesamt: int, d2_korrekt: int,
    d3_score: int, d3_ratio: float, d3_text: str,
    gesamt_score: int,
    saetze: List[SatzPitch],
    monoton_passagen: List[MonotonPassage],
    audio_name: str,
    audio_dauer_s: float,
    insufficient_data: bool
) -> str:

    lines = []
    lines.append("=" * 70)
    lines.append("PITCH-VARIATION ANALYSE REPORT")
    lines.append("=" * 70)
    lines.append(f"Quelle: {audio_name}")
    lines.append(f"Dauer: {audio_dauer_s:.2f}s")
    lines.append("")

    lines.append("-" * 70)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("-" * 70)
    lines.append(f"Gesamt-Score: {gesamt_score}/100")
    lines.append("")
    lines.append(f"  D1 Gesamtvariation (40%): {d1_score}/100 — {d1_text}")
    lines.append(f"      (Std = {d1_std:.2f} Semitones)")
    lines.append(f"  D2 End-Kontur      (30%): {d2_score}/100 — {d2_text}")
    if insufficient_data:
        lines.append(f"      (Fix v2: Weniger als {D2_MIN_SAETZE} Sätze >= 800ms)")
    else:
        lines.append(f"      ({d2_korrekt}/{d2_gesamt} korrekt = {d2_anteil:.1%})")
    lines.append(f"  D3 KB-Variation    (30%): {d3_score}/100 — {d3_text}")
    lines.append(f"      (Verhältnis Kern-Std / Gesamt-Std = {d3_ratio:.2f})")
    lines.append("")

    # Sätze-Detail
    lines.append("-" * 70)
    lines.append("SÄTZE & ENDKONTUREN")
    lines.append("-" * 70)
    lines.append(f"{'Idx':>4} {'Start':<10} {'Ende':<10} {'Dauer':<8} {'Typ':<10} {'Std':<6} {'End-ST':<8} {'Status':<12}")
    lines.append("-" * 70)

    for s in saetze:
        if s.dauer_ms < 800:
            status = "(zu kurz)"
        elif s.ist_frage or s.ist_aussage:
            status = "✅ OK" if s.endkontur_korrekt else "❌ Abweichung"
        else:
            status = "(n/a)"

        typ = "Frage" if s.ist_frage else ("Aussage" if s.ist_aussage else "Andere")
        kb_mark = " [KB]" if s.ist_kernbotschaft else ""

        lines.append(
            f"{s.index:>4} {s_to_zeitstr(s.start_s):<10} {s_to_zeitstr(s.end_s):<10} "
            f"{s.dauer_ms/1000:.1f}s    {typ:<10} {s.std_st:<6.2f} {s.endkontur_st:+.2f}     {status}{kb_mark}"
        )
    lines.append("")

    # Monoton-Passagen
    if monoton_passagen:
        lines.append("-" * 70)
        lines.append("⚠️  MONOTON-PASSAGEN (> 8 Sekunden, Variation < 1.5 ST)")
        lines.append("-" * 70)
        lines.append("Grundlage: The Learning Hall (2025) — nach ~8 Sekunden Monotonie")
        lines.append("schaltet das Gehirn auf 'Dulled Input'.")
        lines.append("")
        for p in monoton_passagen:
            lines.append(
                f"  [{s_to_zeitstr(p.start_s)}] — [{s_to_zeitstr(p.end_s)}] "
                f"({p.dauer_s:.1f}s, Std = {p.std_st:.2f} ST)"
            )
        lines.append("")
    else:
        lines.append("-" * 70)
        lines.append("✅ Keine Monoton-Passagen > 8 Sekunden erkannt.")
        lines.append("-" * 70)
        lines.append("")

    # Studien
    lines.append("-" * 70)
    lines.append("STUDIEN-REFERENZEN")
    lines.append("-" * 70)
    lines.append("• Hincks & Edlund (2009): Pitch Variation Quotient (PVQ)")
    lines.append("• Hahn (2004): Monotone Delivery reduziert Erinnerungsleistung")
    lines.append("• Johns-Lewis (1986): Große Auditorien brauchen viel Pitch-Variation")
    lines.append("• Journal of Voice: Monotone Sprecher = 'weniger kompetent'")
    lines.append("• The Learning Hall (2025): Attention-Cutoff nach ~8s Monotonie")
    lines.append("")
    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def analyse_pitch_variation(
    audio_pfad: Path,
    inhalt_pfad: Optional[Path] = None,
    output_json_pfad: Optional[Path] = None,
    output_txt_pfad: Optional[Path] = None
) -> Dict[str, Any]:

    if not HAS_LIBROSA:
        raise ImportError("librosa nicht installiert. pip install librosa soundfile")

    print(f"[pitch] Starte Analyse: {audio_pfad.name}")

    # 1. Audio laden
    y, sr = lade_audio(audio_pfad)
    audio_dauer_s = len(y) / sr

    # 2. F0 berechnen
    f0_hz, voiced_flag, times_s = berechne_f0(y, sr)
    print(f"[pitch] {len(f0_hz)} F0-Frames berechnet.")

    # 3. Baseline & Semitones
    voiced_f0 = f0_hz[f0_hz > 0]
    if len(voiced_f0) == 0:
        raise ValueError("Keine voiced Frames gefunden — Audio möglicherweise stumm.")

    baseline_hz = float(np.median(voiced_f0))
    f0_st = hz_to_semitones(f0_hz, baseline_hz)
    print(f"[pitch] Baseline F0 = {baseline_hz:.2f} Hz")

    # 4. Sätze laden
    inhalt_data = lade_json(inhalt_pfad) if inhalt_pfad else None
    saetze = extrahiere_saetze(inhalt_data)
    markiere_kernbotschaften(saetze, inhalt_data)
    print(f"[pitch] {len(saetze)} Sätze geladen.")

    # 5. Pitch zu Sätzen zuordnen
    ordne_pitch_zu_saetzen(saetze, f0_hz, f0_st, times_s)

    # 6. Endkonturen berechnen
    for s in saetze:
        berechne_endkontur(s, f0_st, times_s, f0_hz)

    # 7. Monoton-Passagen finden
    monoton_passagen = finde_monoton_passagen(f0_st, times_s, f0_hz)
    if monoton_passagen:
        print(f"[pitch] {len(monoton_passagen)} Monoton-Passage(n) gefunden.")

    # 8. Scoring
    d1_score, d1_std, d1_text = score_d1_gesamtvariation(f0_st, f0_hz)
    d2_score, d2_anteil, d2_text, d2_gesamt, d2_korrekt = score_d2_endkontur(saetze)
    d3_score, d3_ratio, d3_text = score_d3_kernbotschafts_variation(saetze, d1_std)
    gesamt_score = berechne_gesamtscore(d1_score, d2_score, d3_score)

    insufficient_data = (d2_gesamt < D2_MIN_SAETZE)

    print(f"[pitch] Scoring: D1={d1_score}, D2={d2_score}, D3={d3_score}, Gesamt={gesamt_score}")

    # 9. Output
    output_data = {
        "modul": "pitch_variation_analyse",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input": str(audio_pfad),
        "meta": {
            "audio_dauer_s": round(audio_dauer_s, 3),
            "sample_rate": SAMPLE_RATE,
            "hop_length_ms": HOP_LENGTH_MS,
            "baseline_hz": round(baseline_hz, 2),
            "voiced_frames": int(np.sum(f0_hz > 0)),
            "unvoiced_frames": int(np.sum(f0_hz <= 0)),
        },
        "saetze": [s.to_dict() for s in saetze],
        "monoton_passagen": [p.to_dict() for p in monoton_passagen],
        "scoring": {
            "d1_gesamtvariation": {
                "gewichtung": GEWICHT_D1,
                "punkte": d1_score,
                "bewertung": d1_text,
                "std_semitones": round(d1_std, 2)
            },
            "d2_endkontur": {
                "gewichtung": GEWICHT_D2,
                "punkte": d2_score,
                "bewertung": d2_text,
                "anteil_korrekt": round(d2_anteil, 4),
                "saetze_gesamt": d2_gesamt,
                "saetze_korrekt": d2_korrekt,
                "insufficient_data": insufficient_data
            },
            "d3_kernbotschafts_variation": {
                "gewichtung": GEWICHT_D3,
                "punkte": d3_score,
                "bewertung": d3_text,
                "verhaeltnis": round(d3_ratio, 4)
            },
            "gesamtscore": gesamt_score
        }
    }

    if output_json_pfad:
        output_json_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_pfad, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"[pitch] JSON gespeichert: {output_json_pfad}")

    if output_txt_pfad:
        output_txt_pfad.parent.mkdir(parents=True, exist_ok=True)
        report = generiere_report(
            d1_score, d1_std, d1_text,
            d2_score, d2_anteil, d2_text,
            d2_gesamt, d2_korrekt,
            d3_score, d3_ratio, d3_text,
            gesamt_score, saetze, monoton_passagen,
            audio_pfad.name, audio_dauer_s,
            insufficient_data
        )
        with open(output_txt_pfad, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[pitch] Report gespeichert: {output_txt_pfad}")

    print(f"[pitch] Fertig. Gesamt-Score: {gesamt_score}/100")
    return output_data


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pitch-Variation Analyse für Präsentationsbewertungs-AI")
    parser.add_argument("audio", type=str, help="Pfad zur Audio-Datei")
    parser.add_argument("--inhalt", type=str, default=None, help="Pfad zu inhalt_analyse_output.json")
    parser.add_argument("--output-json", type=str, default="zwischen_output/pitch_variation_analyse_output.json")
    parser.add_argument("--output-txt", type=str, default=None)

    args = parser.parse_args()

    audio = Path(args.audio)
    inhalt = Path(args.inhalt) if args.inhalt else None
    out_json = Path(args.output_json)

    if args.output_txt:
        out_txt = Path(args.output_txt)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt = Path("reports/pitch") / f"pitch_report_{ts}.txt"

    if not audio.exists():
        print(f"[FEHLER] Audio nicht gefunden: {audio}")
        exit(1)

    try:
        analyse_pitch_variation(audio, inhalt, out_json, out_txt)
    except Exception as e:
        print(f"[FEHLER] {e}")
        raise
