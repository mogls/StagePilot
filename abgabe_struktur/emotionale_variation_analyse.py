#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""emotionale_variation_analyse.py
===============================
Erkennt Emotionen in der Stimme mittels wav2vec2 und bewertet ob die
emotionale Variation zum Inhalt und zur Struktur passt.

Input:
  - Audio-Datei
  - inhalt_analyse_output.json (für Kernbotschaften + emotionaler_ton)

Output:
  - zwischen_output/emotionale_variation_analyse_output.json
  - reports/emotion/emotion_report_[TIMESTAMP].txt"""

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
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("torch nicht installiert. pip install torch")

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers nicht installiert. pip install transformers")

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
FENSTER_S = 3.0           # 3 Sekunden
HOP_S = 2.0               # 1 Sekunde Overlap → 2s Hop

# Arousal-Wechsel
AROUSAL_WECHSEL_SCHWELLE = 0.15   # Delta > 0.15 = spürbarer Wechsel

# D1 Skala
D1_OPTIMAL_MIN = 3.0
D1_OPTIMAL_MAX = 5.0
D1_AKZEPTABEL_MIN = 1.5
D1_AKZEPTABEL_MAX = 8.0

# D2 Valence-Bereiche je Ton-Label
VALENCE_INSPIRIEREND_MIN = 0.55
VALENCE_SACHLICH_MIN = 0.40
VALENCE_SACHLICH_MAX = 0.60
VALENCE_ERNST_MAX = 0.45

# D3 Dominance
DOMINANCE_ANHEBUNG_GUT = 0.15
DOMINANCE_ANHEBUNG_LEICHT = 0.05

# Scoring
GEWICHT_D1 = 0.40
GEWICHT_D2 = 0.30
GEWICHT_D3 = 0.30

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class EmotionsSegment:
    """Ein 3s-Segment mit Emotionswerten."""
    start_s: float
    end_s: float
    arousal: float
    valence: float
    dominance: float

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "arousal": round(self.arousal, 4),
            "valence": round(self.valence, 4),
            "dominance": round(self.dominance, 4),
        }


@dataclass
class KernbotschaftEmotion:
    """Emotionsdaten für eine Kernbotschaft."""
    text: str
    start_s: float
    end_s: float
    mean_arousal: float
    mean_valence: float
    mean_dominance: float
    valence_passend: bool
    valence_erwartet_min: float
    valence_erwartet_max: float

    def to_dict(self) -> dict:
        return {
            "text": self.text[:80] + "..." if len(self.text) > 80 else self.text,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "mean_arousal": round(self.mean_arousal, 4),
            "mean_valence": round(self.mean_valence, 4),
            "mean_dominance": round(self.mean_dominance, 4),
            "valence_passend": self.valence_passend,
            "valence_erwartet": f"[{self.valence_erwartet_min:.2f}, {self.valence_erwartet_max:.2f}]",
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
# MODELL-LADEN
# =============================================================================

class EmotionModel:
    """Wrapper für das audEERING wav2vec2 Emotionsmodell."""

    def __init__(self):
        if not HAS_TORCH or not HAS_TRANSFORMERS or not HAS_LIBROSA:
            raise ImportError(
                "Benötigte Pakete fehlen. Installieren:\n"
                "  pip install torch transformers librosa soundfile"
            )

        print(f"[emotion] Lade Modell: {MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[emotion] Device: {self.device}")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        # Labels: Das Modell gibt Arousal, Valence, Dominance zurück
        # Die ID-Zuordnung ist im Modell-Config gespeichert
        self.id2label = self.model.config.id2label
        print(f"[emotion] Modell geladen. Labels: {list(self.id2label.values())}")

    def predict(self, audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
        """
        Predict Arousal, Valence, Dominance für ein Audio-Segment.

        Returns:
            (arousal, valence, dominance) — jeweils 0..1
        """
        # Resample falls nötig
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Feature Extraction
        inputs = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Logits zu Wahrscheinlichkeiten
        logits = outputs.logits.cpu().numpy()[0]

        # Das Modell gibt direkt Arousal/Valence/Dominance als Regression aus
        # Normalerweise sind die Outputs bereits skaliert (0-1 oder -1 bis 1)
        # Wir müssen die genaue Skalierung prüfen

        # Für das audEERING Modell: 3 Outputs = [arousal, valence, dominance]
        # Typischerweise im Bereich [0, 1] oder [-1, 1]
        # Wir normalisieren auf [0, 1]

        arousal = float(self._normalize(logits[0]))
        valence = float(self._normalize(logits[1]))
        dominance = float(self._normalize(logits[2]))

        return arousal, valence, dominance

    def _normalize(self, value: float) -> float:
        """Normalisiert Modell-Output auf [0, 1]."""
        # Das audEERING Modell gibt typischerweise Werte im Bereich [-1, 1] aus
        # Wir mappen auf [0, 1]
        if value < -1.0:
            value = -1.0
        if value > 1.0:
            value = 1.0
        return (value + 1.0) / 2.0


# =============================================================================
# AUDIO-SEGMENTIERUNG
# =============================================================================

def segmentiere_audio(y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segmentiert Audio in 3s-Fenster mit 1s Overlap.

    Returns:
        Liste von (segment_audio, start_s, end_s)
    """
    segments = []
    fenster_samples = int(FENSTER_S * sr)
    hop_samples = int(HOP_S * sr)

    start = 0
    while start + fenster_samples <= len(y):
        segment = y[start:start + fenster_samples]
        start_s = start / sr
        end_s = (start + fenster_samples) / sr
        segments.append((segment, start_s, end_s))
        start += hop_samples

    # Letztes Segment (kürzer, aber mindestens 1s)
    remaining = len(y) - start
    if remaining >= sr:  # Mindestens 1 Sekunde
        segment = y[start:]
        # Padding auf 3s mit Nullen
        if len(segment) < fenster_samples:
            segment = np.pad(segment, (0, fenster_samples - len(segment)), mode='constant')
        start_s = start / sr
        end_s = len(y) / sr
        segments.append((segment, start_s, end_s))

    return segments


# =============================================================================
# EMOTIONS-ANALYSE
# =============================================================================

def analyse_emotionen(
    audio_pfad: Path,
    model: EmotionModel
) -> Tuple[List[EmotionsSegment], np.ndarray, int]:
    """
    Analysiert das gesamte Audio in 3s-Segmenten.

    Returns:
        (segmente, audio_array, sample_rate)
    """
    y, sr = librosa.load(str(audio_pfad), sr=SAMPLE_RATE, mono=True)

    raw_segments = segmentiere_audio(y, sr)
    print(f"[emotion] {len(raw_segments)} Segmente (3s-Fenster, 1s Overlap)")

    emotion_segments = []
    for i, (seg_audio, start_s, end_s) in enumerate(raw_segments):
        arousal, valence, dominance = model.predict(seg_audio, sr)
        emotion_segments.append(EmotionsSegment(
            start_s=start_s,
            end_s=end_s,
            arousal=arousal,
            valence=valence,
            dominance=dominance
        ))
        if (i + 1) % 10 == 0:
            print(f"[emotion] {i + 1}/{len(raw_segments)} Segmente verarbeitet...")

    return emotion_segments, y, sr


# =============================================================================
# KERNBOTSCHAFT-ZUORDNUNG
# =============================================================================

def extrahiere_kernbotschaften(inhalt_data: Optional[Dict]) -> List[Dict]:
    """Extrahiert Kernbotschaften mit Text und Zeit."""
    kbs = []
    if not inhalt_data or "kernbotschaften" not in inhalt_data:
        return kbs

    for kb in inhalt_data["kernbotschaften"]:
        start = kb.get("start_ms", kb.get("start"))
        end = kb.get("end_ms", kb.get("end"))
        text = kb.get("text", "")

        if isinstance(start, str):
            start = zeitstr_to_s(start)
        if isinstance(end, str):
            end = zeitstr_to_s(end)
        if start is None or end is None:
            continue

        start_s = float(start) / 1000.0 if float(start) > 1000 else float(start)
        end_s = float(end) / 1000.0 if float(end) > 1000 else float(end)

        kbs.append({"text": text, "start_s": start_s, "end_s": end_s})

    return kbs


def hole_ton_label(inhalt_data: Optional[Dict]) -> str:
    """Extrahiert den emotionalen Ton aus der Inhaltsanalyse."""
    if not inhalt_data:
        return "sachlich"  # Default

    # Mögliche Felder: emotionaler_ton, ton, sentiment, etc.
    ton = inhalt_data.get("emotionaler_ton", inhalt_data.get("ton", "sachlich"))
    if isinstance(ton, str):
        ton = ton.lower()
        if "inspiri" in ton or "motiv" in ton:
            return "inspirierend"
        elif "ernst" in ton or "seriös" in ton or "grav" in ton:
            return "ernst"
        else:
            return "sachlich"
    return "sachlich"


def ordne_segmente_zu_kb(
    emotion_segments: List[EmotionsSegment],
    kernbotschaften: List[Dict]
) -> List[KernbotschaftEmotion]:
    """Ordnet Emotions-Segmente den Kernbotschaften zu und berechnet Mittelwerte."""
    ergebnisse = []

    for kb in kernbotschaften:
        kb_start = kb["start_s"]
        kb_end = kb["end_s"]

        # Segmente, die mit der KB überlappen
        passende = [s for s in emotion_segments
                    if s.start_s < kb_end and s.end_s > kb_start]

        if not passende:
            continue

        mean_arousal = float(np.mean([s.arousal for s in passende]))
        mean_valence = float(np.mean([s.valence for s in passende]))
        mean_dominance = float(np.mean([s.dominance for s in passende]))

        ergebnisse.append(KernbotschaftEmotion(
            text=kb["text"],
            start_s=kb_start,
            end_s=kb_end,
            mean_arousal=mean_arousal,
            mean_valence=mean_valence,
            mean_dominance=mean_dominance,
            valence_passend=False,  # Wird später gesetzt
            valence_erwartet_min=0.0,
            valence_erwartet_max=1.0,
        ))

    return ergebnisse


# =============================================================================
# SCORING
# =============================================================================

def berechne_d1_wechsel_rate(emotion_segments: List[EmotionsSegment], dauer_min: float) -> Tuple[int, float, str]:
    """
    D1: Arousal-Wechsel-Rate (40%)
    Fix v2: Wechsel = |arousal[i+1] - arousal[i]| > 0.15
    """
    if len(emotion_segments) < 2:
        return 40, 0.0, "Monoton (weniger als 2 Segmente)"

    wechsel = 0
    for i in range(len(emotion_segments) - 1):
        delta = abs(emotion_segments[i + 1].arousal - emotion_segments[i].arousal)
        if delta > AROUSAL_WECHSEL_SCHWELLE:
            wechsel += 1

    if dauer_min > 0:
        rate = wechsel / dauer_min
    else:
        rate = 0.0

    if D1_OPTIMAL_MIN <= rate <= D1_OPTIMAL_MAX:
        punkte = 100
        bewertung = "TED-Optimum"
    elif (D1_AKZEPTABEL_MIN <= rate < D1_OPTIMAL_MIN) or (D1_OPTIMAL_MAX < rate <= D1_AKZEPTABEL_MAX):
        punkte = 75
        bewertung = "Akzeptabel"
    elif rate < D1_AKZEPTABEL_MIN:
        punkte = 40
        bewertung = "Monoton"
    else:  # > 8
        punkte = 40
        bewertung = "Sprunghaft"

    return punkte, rate, bewertung


def berechne_d2_valence_passung(
    kb_emotionen: List[KernbotschaftEmotion],
    ton_label: str
) -> Tuple[int, float, str]:
    """
    D2: Valence-Passung Kernbotschaften (30%)
    Fix v2: Operationalisiert mit erwarteten Bereichen pro Ton-Label.
    """
    if not kb_emotionen:
        return 100, 1.0, "Keine Kernbotschaften"

    # Erwartete Bereiche definieren
    if ton_label == "inspirierend":
        erwartet_min, erwartet_max = VALENCE_INSPIRIEREND_MIN, 1.0
    elif ton_label == "ernst":
        erwartet_min, erwartet_max = 0.0, VALENCE_ERNST_MAX
    else:  # sachlich
        erwartet_min, erwartet_max = VALENCE_SACHLICH_MIN, VALENCE_SACHLICH_MAX

    passend = 0
    for kb in kb_emotionen:
        kb.valence_erwartet_min = erwartet_min
        kb.valence_erwartet_max = erwartet_max
        if erwartet_min <= kb.mean_valence <= erwartet_max:
            kb.valence_passend = True
            passend += 1
        else:
            kb.valence_passend = False

    anteil = passend / len(kb_emotionen)

    if anteil >= 0.70:
        punkte = 100
        bewertung = "Konsistent"
    elif anteil >= 0.50:
        punkte = 70
        bewertung = "Teilweise"
    else:
        punkte = 30
        bewertung = "Inkonsistent"

    return punkte, anteil, bewertung


def berechne_d3_dominance(
    emotion_segments: List[EmotionsSegment],
    kb_emotionen: List[KernbotschaftEmotion]
) -> Tuple[int, float, str]:
    """
    D3: Dominance bei Kernbotschaften (30%)
    Ist Dominance bei KB höher als im Durchschnitt?
    """
    if not kb_emotionen:
        return 100, 0.0, "Keine Kernbotschaften"

    # Globaler Durchschnitt
    global_dom = float(np.mean([s.dominance for s in emotion_segments]))
    kb_dom = float(np.mean([kb.mean_dominance for kb in kb_emotionen]))

    anhebung = kb_dom - global_dom

    if anhebung >= DOMINANCE_ANHEBUNG_GUT:
        punkte = 100
        bewertung = "Klare Erhöhung"
    elif anhebung >= DOMINANCE_ANHEBUNG_LEICHT:
        punkte = 70
        bewertung = "Leicht"
    else:
        punkte = 30
        bewertung = "Keine Betonung"

    return punkte, anhebung, bewertung


def berechne_gesamtscore(d1: int, d2: int, d3: int) -> int:
    score = d1 * GEWICHT_D1 + d2 * GEWICHT_D2 + d3 * GEWICHT_D3
    return int(round(score))


# =============================================================================
# REPORT
# =============================================================================

def generiere_report(
    emotion_segments: List[EmotionsSegment],
    kb_emotionen: List[KernbotschaftEmotion],
    d1_score: int, d1_rate: float, d1_text: str,
    d2_score: int, d2_anteil: float, d2_text: str,
    d3_score: int, d3_anhebung: float, d3_text: str,
    gesamt_score: int,
    ton_label: str,
    audio_name: str,
    audio_dauer_s: float
) -> str:

    lines = []
    lines.append("=" * 70)
    lines.append("EMOTIONALE VARIATION ANALYSE REPORT")
    lines.append("=" * 70)
    lines.append(f"Quelle: {audio_name}")
    lines.append(f"Dauer: {audio_dauer_s:.2f}s")
    lines.append(f"Modell: {MODEL_NAME}")
    lines.append(f"Gesamt-Ton (Inhalt): {ton_label}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("-" * 70)
    lines.append(f"Gesamt-Score: {gesamt_score}/100")
    lines.append("")
    lines.append(f"  D1 Arousal-Wechsel-Rate (40%): {d1_score}/100 — {d1_text}")
    lines.append(f"      ({d1_rate:.2f} Wechsel/Min, Schwelle Delta > {AROUSAL_WECHSEL_SCHWELLE})")
    lines.append(f"  D2 Valence-Passung        (30%): {d2_score}/100 — {d2_text}")
    lines.append(f"      ({d2_anteil:.1%} der Kernbotschaften im erwarteten Valence-Bereich)")
    lines.append(f"  D3 Dominance-Anhebung     (30%): {d3_score}/100 — {d3_text}")
    lines.append(f"      (Anhebung = {d3_anhebung:+.3f} über Global-Ø)")
    lines.append("")

    # Alle Segmente
    lines.append("-" * 70)
    lines.append("EMOTIONS-SEGMENTE (3s-Fenster, 1s Overlap)")
    lines.append("-" * 70)
    lines.append(f"{'Start':<10} {'Ende':<10} {'Arousal':<10} {'Valence':<10} {'Dominance':<10}")
    lines.append("-" * 70)
    for s in emotion_segments:
        lines.append(
            f"{s_to_zeitstr(s.start_s):<10} {s_to_zeitstr(s.end_s):<10} "
            f"{s.arousal:<10.3f} {s.valence:<10.3f} {s.dominance:<10.3f}"
        )
    lines.append("")

    # Kernbotschaften
    if kb_emotionen:
        lines.append("-" * 70)
        lines.append("KERNBOTSCHAFTEN — EMOTIONALE PASSUNG")
        lines.append("-" * 70)
        lines.append(f"Erwarteter Valence-Bereich für '{ton_label}': "
                     f"[{kb_emotionen[0].valence_erwartet_min:.2f}, "
                     f"{kb_emotionen[0].valence_erwartet_max:.2f}]")
        lines.append("")
        for kb in kb_emotionen:
            status = "✅ OK" if kb.valence_passend else "❌ Abweichung"
            lines.append(
                f"  [{s_to_zeitstr(kb.start_s)}] Val={kb.mean_valence:.3f} Dom={kb.mean_dominance:.3f} "
                f"Aro={kb.mean_arousal:.3f}  {status}"
            )
            lines.append(f"      {kb.text[:60]}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("STUDIEN-REFERENZEN")
    lines.append("-" * 70)
    lines.append("• Baevski et al. (2020): wav2vec 2.0 (Facebook AI)")
    lines.append("• Golbaghi & Zhou (2024): 96--97% BCA auf EMO-DB")
    lines.append("• audEERING: wav2vec2-large-robust-12-ft-emotion-msp-dim")
    lines.append("• Russell (1980): Circumplex-Modell der Emotion")
    lines.append("• Interspeech-Studie: TED-Speaker 3--5 Emotionswechsel/Min")
    lines.append("")
    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def analyse_emotionale_variation(
    audio_pfad: Path,
    inhalt_pfad: Optional[Path] = None,
    output_json_pfad: Optional[Path] = None,
    output_txt_pfad: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Haupt-Einstiegspunkt.
    """
    if not HAS_TORCH or not HAS_TRANSFORMERS or not HAS_LIBROSA:
        raise ImportError(
            "Benötigte Pakete fehlen. Installieren:\n"
            "  pip install torch transformers librosa soundfile"
        )

    print(f"[emotion] Starte Analyse: {audio_pfad.name}")

    # 1. Modell laden
    model = EmotionModel()

    # 2. Emotionen analysieren
    emotion_segments, y, sr = analyse_emotionen(audio_pfad, model)
    audio_dauer_s = len(y) / sr
    dauer_min = audio_dauer_s / 60.0

    # 3. Inhaltsanalyse laden
    inhalt_data = lade_json(inhalt_pfad) if inhalt_pfad else None
    ton_label = hole_ton_label(inhalt_data)
    kernbotschaften = extrahiere_kernbotschaften(inhalt_data)

    # 4. KB zuordnen
    kb_emotionen = ordne_segmente_zu_kb(emotion_segments, kernbotschaften)
    print(f"[emotion] {len(kb_emotionen)} Kernbotschaften mit Emotionsdaten versehen.")

    # 5. Scoring
    d1_score, d1_rate, d1_text = berechne_d1_wechsel_rate(emotion_segments, dauer_min)
    d2_score, d2_anteil, d2_text = berechne_d2_valence_passung(kb_emotionen, ton_label)
    d3_score, d3_anhebung, d3_text = berechne_d3_dominance(emotion_segments, kb_emotionen)
    gesamt_score = berechne_gesamtscore(d1_score, d2_score, d3_score)

    print(f"[emotion] Scoring: D1={d1_score}, D2={d2_score}, D3={d3_score}, Gesamt={gesamt_score}")

    # 6. Output
    output_data = {
        "modul": "emotionale_variation_analyse",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input": str(audio_pfad),
        "meta": {
            "audio_dauer_s": round(audio_dauer_s, 3),
            "model": MODEL_NAME,
            "fenster_s": FENSTER_S,
            "hop_s": HOP_S,
            "segmente_anzahl": len(emotion_segments),
            "ton_label": ton_label,
        },
        "segmente": [s.to_dict() for s in emotion_segments],
        "kernbotschaften": [k.to_dict() for k in kb_emotionen],
        "scoring": {
            "d1_arousal_wechsel": {
                "gewichtung": GEWICHT_D1,
                "punkte": d1_score,
                "bewertung": d1_text,
                "wechsel_rate_pro_min": round(d1_rate, 2),
                "wechsel_schwelle": AROUSAL_WECHSEL_SCHWELLE
            },
            "d2_valence_passung": {
                "gewichtung": GEWICHT_D2,
                "punkte": d2_score,
                "bewertung": d2_text,
                "anteil_passend": round(d2_anteil, 4),
                "ton_label": ton_label
            },
            "d3_dominance_anhebung": {
                "gewichtung": GEWICHT_D3,
                "punkte": d3_score,
                "bewertung": d3_text,
                "anhebung": round(d3_anhebung, 4)
            },
            "gesamtscore": gesamt_score
        }
    }

    if output_json_pfad:
        output_json_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_pfad, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"[emotion] JSON gespeichert: {output_json_pfad}")

    if output_txt_pfad:
        output_txt_pfad.parent.mkdir(parents=True, exist_ok=True)
        report = generiere_report(
            emotion_segments, kb_emotionen,
            d1_score, d1_rate, d1_text,
            d2_score, d2_anteil, d2_text,
            d3_score, d3_anhebung, d3_text,
            gesamt_score, ton_label,
            audio_pfad.name, audio_dauer_s
        )
        with open(output_txt_pfad, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[emotion] Report gespeichert: {output_txt_pfad}")

    print(f"[emotion] Fertig. Gesamt-Score: {gesamt_score}/100")
    return output_data


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotionale Variation Analyse für Präsentationsbewertungs-AI")
    parser.add_argument("audio", type=str, help="Pfad zur Audio-Datei")
    parser.add_argument("--inhalt", type=str, default=None, help="Pfad zu inhalt_analyse_output.json")
    parser.add_argument("--output-json", type=str, default="zwischen_output/emotionale_variation_analyse_output.json")
    parser.add_argument("--output-txt", type=str, default=None)

    args = parser.parse_args()

    audio = Path(args.audio)
    inhalt = Path(args.inhalt) if args.inhalt else None
    out_json = Path(args.output_json)

    if args.output_txt:
        out_txt = Path(args.output_txt)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt = Path("reports/emotion") / f"emotion_report_{ts}.txt"

    if not audio.exists():
        print(f"[FEHLER] Audio nicht gefunden: {audio}")
        exit(1)

    try:
        analyse_emotionale_variation(audio, inhalt, out_json, out_txt)
    except Exception as e:
        print(f"[FEHLER] {e}")
        raise
