#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pausen_analyse.py
=================
Erkennt Pausen aus Zeitstempel-Lücken zwischen Wörtern.
Klassifiziert in 8 Kategorien mit kontextueller Bewertung.
Einzige Quelle für Stocker-Erkennung (Fix v2: keine Doppelzählung)."""

import json
import os
import re
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict


# =============================================================================
# KONSTANTEN — studien-basiert, siehe Dokument Abschnitt 6.2
# =============================================================================

# Schwellenwerte in Millisekunden
MS_IGNORIEREN = 150              # < 150 ms: unter Wahrnehmungsschwelle
MS_SEGMENT_GRENZE = 10000        # > 10 s: Applaus, Cuts, Fragen — ausschließen
MS_KLEINER_STOCKER = 300         # 150--300 ms: kurzer Stocker
MS_STOCKER = 800                 # 300--800 ms: Stocker
MS_STOCKER_LANG = 2000           # 800--2000 ms: langer Stocker innerhalb Satz
MS_ZU_LANG = 4000                # > 4000 ms: zu lang
MS_ATEM_MIN = 500                # 500 ms
MS_ATEM_MAX = 1500               # 1500 ms
MS_RHETORISCH = 800              # ≥ 800 ms gilt als rhetorisch relevant
MS_WIRKUNG_MIN = 2000            # 2000 ms
MS_WIRKUNG_MAX = 4000            # 4000 ms
MS_SINNPAUSE_MAX = 2000          # 2000 ms (Obergrenze Sinnpause)

SATZ_LAENGE_LANG = 15            # > 15 Wörter = langer Satz (Atempause-Trigger)

# Scoring-Gewichtung 40/30/30
GEWICHT_D1 = 0.40
GEWICHT_D2 = 0.30
GEWICHT_D3 = 0.30

# Pfade (relativ zum Projekt-Root)
DEFAULT_INPUT_DIR = Path("zwischen_output")
DEFAULT_OUTPUT_DIR = Path("zwischen_output")
DEFAULT_REPORT_DIR = Path("reports/pausen")


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class Wort:
    """Ein Wort mit Start-/End-Zeitstempel aus dem Transkript."""
    text: str
    start_ms: float
    end_ms: float
    index: int = 0

    @property
    def dauer_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Satz:
    """Ein Satz, abgeleitet aus inhalt_analyse_output.json."""
    index: int
    text: str
    start_ms: float
    end_ms: float
    woerter: List[Wort] = field(default_factory=list)
    wortanzahl: int = 0
    ist_kernbotschaft: bool = False
    ist_struktur_uebergang: bool = False
    ist_lang: bool = False  # > 15 Wörter


@dataclass
class Pause:
    """Eine erkannte Pause zwischen zwei Wörtern."""
    start_ms: float          # Endzeit des vorherigen Worts
    end_ms: float            # Startzeit des nächsten Worts
    dauer_ms: float
    typ: str                 # Klassifikation

    # Kontext-Informationen
    vorheriges_wort: str = ""
    naechstes_wort: str = ""
    innerhalb_satz: bool = False
    nach_langem_satz: bool = False
    an_struktur_uebergang: bool = False
    vor_kernbotschaft: bool = False
    nach_kernbotschaft: bool = False
    satz_index: int = -1

    @property
    def ist_stocker(self) -> bool:
        return self.typ in ("kleiner_stocker", "stocker", "stocker_lang", "zu_lang")

    @property
    def ist_rhetorisch(self) -> bool:
        return self.typ in ("sinnpause", "wirkungspause", "strukturpause")

    @property
    def ist_zaehlbar(self) -> bool:
        """Wird in D3 (Pausen/Min) gezählt."""
        return self.typ not in ("ignorieren", "segment_grenze")

    def to_dict(self) -> dict:
        return {
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "dauer_ms": round(self.dauer_ms, 3),
            "typ": self.typ,
            "kontext": {
                "vorheriges_wort": self.vorheriges_wort,
                "naechstes_wort": self.naechstes_wort,
                "innerhalb_satz": self.innerhalb_satz,
                "nach_langem_satz": self.nach_langem_satz,
                "an_struktur_uebergang": self.an_struktur_uebergang,
                "vor_kernbotschaft": self.vor_kernbotschaft,
                "nach_kernbotschaft": self.nach_kernbotschaft,
            }
        }


@dataclass
class KernbotschaftCheck:
    """Ergebnis des Kernbotschaft-Pausen-Checks (Abschnitt 6.5)."""
    kernbotschaft_text: str
    start_ms: float
    end_ms: float
    pause_davor: Optional[Pause] = None
    pause_danach: Optional[Pause] = None
    hat_rhetorische_pause: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.kernbotschaft_text,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "hat_rhetorische_pause": self.hat_rhetorische_pause,
            "pause_davor_typ": self.pause_davor.typ if self.pause_davor else None,
            "pause_danach_typ": self.pause_danach.typ if self.pause_danach else None,
        }


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def zeitstr_to_ms(zeit_str: str) -> float:
    """
    Parst "HH:MM:SS.mmm" oder "MM:SS.mmm" zu Millisekunden.
    Robust gegen führende Nullen und verschiedene Formate.
    """
    zeit_str = zeit_str.strip()

    # Versuche HH:MM:SS.mmm
    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", zeit_str):
        h, m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    # Versuche MM:SS.mmm
    if re.match(r"^\d{2}:\d{2}\.\d{3}$", zeit_str):
        m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(m) * 60000 + int(s) * 1000 + int(ms)

    # Fallback: Versuche mit datetime
    for fmt in ("%H:%M:%S.%f", "%M:%S.%f", "%H:%M:%S", "%M:%S"):
        try:
            dt = datetime.strptime(zeit_str, fmt)
            return (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
        except ValueError:
            continue

    raise ValueError(f"Unbekanntes Zeitformat: {zeit_str}")


def ms_to_zeitstr(ms: float) -> str:
    """Millisekunden zu "MM:SS.mmm" oder "HH:MM:SS.mmm"."""
    ms = max(0, ms)
    total_sec = int(ms // 1000)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    millis = int(ms % 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"
    return f"{m:02d}:{s:02d}.{millis:03d}"


def parse_transkript(transkript_pfad: Path) -> List[Wort]:
    """
    Parst Transkript im Format:
        Wort HH:MM:SS.mmm HH:MM:SS.mmm

    Returns:
        Liste von Wort-Objekten, sortiert nach Startzeit.
    """
    woerter = []
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)$")

    with open(transkript_pfad, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            m = pattern.match(line)
            if not m:
                # Versuche es mit variablen Leerzeichen/Tabs
                teile = line.split()
                if len(teile) >= 3:
                    wort_text = teile[0]
                    start_str = teile[-2]
                    end_str = teile[-1]
                else:
                    print(f"[WARN] Zeile {i} übersprungen (Format): {line[:60]}")
                    continue
            else:
                wort_text, start_str, end_str = m.groups()

            try:
                start_ms = zeitstr_to_ms(start_str)
                end_ms = zeitstr_to_ms(end_str)
            except ValueError as e:
                print(f"[WARN] Zeile {i} übersprungen (Zeit): {e}")
                continue

            woerter.append(Wort(
                text=wort_text,
                start_ms=start_ms,
                end_ms=end_ms,
                index=len(woerter)
            ))

    # Sicherstellen, dass sortiert ist
    woerter.sort(key=lambda w: w.start_ms)
    for i, w in enumerate(woerter):
        w.index = i

    return woerter


def lade_inhalt_analyse(pfad: Path) -> Optional[Dict]:
    """Lädt inhalt_analyse_output.json, falls vorhanden."""
    if not pfad.exists():
        return None
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] Konnte inhalt_analyse nicht laden: {e}")
        return None


def extrahiere_saetze(
    woerter: List[Wort],
    inhalt_data: Optional[Dict]
) -> List[Satz]:
    """
    Extrahiert Sätze aus inhalt_analyse_output.json.
    Fallback: Wenn keine Inhaltsanalyse vorhanden, heuristisch nach Satzzeichen.
    """
    saetze = []

    if inhalt_data and "satzgrenzen" in inhalt_data:
        # Erwarte Format: [{"index": 0, "text": "...", "start_ms": 123, "end_ms": 456}, ...]
        # oder mit Zeitstempel-Strings
        raw_saetze = inhalt_data["satzgrenzen"]

        for i, raw in enumerate(raw_saetze):
            start_ms = raw.get("start_ms")
            end_ms = raw.get("end_ms")

            # Fallback: Konvertiere Strings
            if isinstance(start_ms, str):
                start_ms = zeitstr_to_ms(start_ms)
            if isinstance(end_ms, str):
                end_ms = zeitstr_to_ms(end_ms)

            if start_ms is None or end_ms is None:
                continue

            # Wörter diesem Satz zuordnen
            satz_woerter = [w for w in woerter if start_ms <= w.start_ms < end_ms]

            saetze.append(Satz(
                index=i,
                text=raw.get("text", ""),
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                woerter=satz_woerter,
                wortanzahl=len(satz_woerter),
                ist_lang=len(satz_woerter) > SATZ_LAENGE_LANG
            ))

    if not saetze:
        # Fallback: Heuristisch nach .!? aufteilen
        print("[INFO] Keine Satzgrenzen aus Inhaltsanalyse — verwende Heuristik.")
        satz_endzeichen = re.compile(r"[.!?]+$")
        aktuelle_woerter = []
        satz_idx = 0

        for w in woerter:
            aktuelle_woerter.append(w)
            if satz_endzeichen.search(w.text) or w.text.endswith((".", "!", "?")):
                start_ms = aktuelle_woerter[0].start_ms
                end_ms = aktuelle_woerter[-1].end_ms
                saetze.append(Satz(
                    index=satz_idx,
                    text=" ".join(x.text for x in aktuelle_woerter),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    woerter=list(aktuelle_woerter),
                    wortanzahl=len(aktuelle_woerter),
                    ist_lang=len(aktuelle_woerter) > SATZ_LAENGE_LANG
                ))
                aktuelle_woerter = []
                satz_idx += 1

        # Restwörter als letzten Satz
        if aktuelle_woerter:
            saetze.append(Satz(
                index=satz_idx,
                text=" ".join(x.text for x in aktuelle_woerter),
                start_ms=aktuelle_woerter[0].start_ms,
                end_ms=aktuelle_woerter[-1].end_ms,
                woerter=aktuelle_woerter,
                wortanzahl=len(aktuelle_woerter),
                ist_lang=len(aktuelle_woerter) > SATZ_LAENGE_LANG
            ))

    return saetze


def markiere_kernbotschaften(
    saetze: List[Satz],
    inhalt_data: Optional[Dict]
) -> None:
    """Markiert Sätze als Kernbotschaften, falls im Inhalts-Output vorhanden."""
    if not inhalt_data or "kernbotschaften" not in inhalt_data:
        return

    kernbotschaften = inhalt_data["kernbotschaften"]

    for kb in kernbotschaften:
        kb_start = kb.get("start_ms")
        kb_end = kb.get("end_ms")

        if isinstance(kb_start, str):
            kb_start = zeitstr_to_ms(kb_start)
        if isinstance(kb_end, str):
            kb_end = zeitstr_to_ms(kb_end)

        if kb_start is None or kb_end is None:
            continue

        for s in saetze:
            # Überschneidung prüfen
            if s.start_ms <= float(kb_end) and s.end_ms >= float(kb_start):
                s.ist_kernbotschaft = True


def markiere_struktur_uebergaenge(
    saetze: List[Satz],
    inhalt_data: Optional[Dict]
) -> None:
    """Markiert Sätze an Struktur-Übergängen (Einleitung→Hauptteil, Hauptteil→Schluss)."""
    if not inhalt_data or "struktur" not in inhalt_data:
        return

    struktur = inhalt_data["struktur"]

    # Erwarte Format: {"einleitung": {"start_ms": 0, "end_ms": 12000}, "hauptteil": {...}, "schluss": {...}}
    segmente = []
    for key in ["einleitung", "hauptteil", "schluss"]:
        if key in struktur:
            seg = struktur[key]
            s_ms = seg.get("start_ms", seg.get("start"))
            e_ms = seg.get("end_ms", seg.get("end"))

            if isinstance(s_ms, str):
                s_ms = zeitstr_to_ms(s_ms)
            if isinstance(e_ms, str):
                e_ms = zeitstr_to_ms(e_ms)

            if s_ms is not None and e_ms is not None:
                segmente.append((float(s_ms), float(e_ms), key))

    # Markiere den ersten Satz nach jedem Segment-Wechsel
    segmente.sort()
    for i in range(1, len(segmente)):
        uebergang_ms = segmente[i][0]  # Start des neuen Segments
        for s in saetze:
            # Satz liegt direkt am Übergang oder kurz danach
            if abs(s.start_ms - uebergang_ms) < 500 or (s.start_ms >= uebergang_ms and s.start_ms < uebergang_ms + 1000):
                s.ist_struktur_uebergang = True


# =============================================================================
# KERN-LOGIK: PAUSEN-ERKENNUNG & KLASSIFIKATION
# =============================================================================

def finde_pausen(
    woerter: List[Wort],
    saetze: List[Satz]
) -> List[Pause]:
    """
    Findet alle Pausen zwischen aufeinanderfolgenden Wörtern.
    Berechnet die Lücke: start_ms(Wort_N+1) - end_ms(Wort_N).
    """
    pausen = []

    # Schneller Lookup: Welcher Satz enthält welches Wort?
    wort_zu_satz = {}
    for s in saetze:
        for w in s.woerter:
            wort_zu_satz[w.index] = s

    for i in range(len(woerter) - 1):
        w1 = woerter[i]
        w2 = woerter[i + 1]

        luecke = w2.start_ms - w1.end_ms

        # Negative Lücken (Überlappungen) ignorieren wir als Artefakt
        if luecke < 0:
            continue

        s1 = wort_zu_satz.get(w1.index)
        s2 = wort_zu_satz.get(w2.index)

        innerhalb_satz = (s1 is not None and s2 is not None and s1.index == s2.index)
        nach_langem_satz = (s1 is not None and s1.ist_lang) if not innerhalb_satz else False
        an_struktur = (s2 is not None and s2.ist_struktur_uebergang) if s2 else False
        vor_kb = (s2 is not None and s2.ist_kernbotschaft) if s2 else False
        nach_kb = (s1 is not None and s1.ist_kernbotschaft) if s1 else False

        pause = Pause(
            start_ms=w1.end_ms,
            end_ms=w2.start_ms,
            dauer_ms=luecke,
            typ="unklassifiziert",
            vorheriges_wort=w1.text,
            naechstes_wort=w2.text,
            innerhalb_satz=innerhalb_satz,
            nach_langem_satz=nach_langem_satz,
            an_struktur_uebergang=an_struktur,
            vor_kernbotschaft=vor_kb,
            nach_kernbotschaft=nach_kb,
            satz_index=s1.index if s1 else -1
        )

        pausen.append(pause)

    return pausen


def klassifiziere_pause(p: Pause) -> str:
    """
    Klassifiziert eine Pause nach der Prioritäts-Matrix aus Abschnitt 6.3.
    ERSTE PASSENDE REGEL GEWINNT.

    Reihenfolge (exakt wie im Dokument):
    1. < 150 ms → ignorieren
    2. > 10 s → segment_grenze
    3. innerhalb Satz + 150--300 ms → kleiner_stocker
    4. innerhalb Satz + 300--800 ms → stocker
    5. innerhalb Satz + 800--2000 ms → stocker_lang
    6. innerhalb Satz + > 2000 ms → zu_lang
    7. nach langem Satz (>15 W.) + 500--1500 ms → atempause (VORRANG)
    8. an Struktur-Übergang + ≥ 800 ms → strukturpause
    9. vor/nach Kernbotschaft + 800--2000 ms → sinnpause
    10. vor/nach Kernbotschaft + 2000--4000 ms → wirkungspause
    11. zwischen Sätzen + 150--300 ms → natürlich_kurz
    12. zwischen Sätzen + 300--2000 ms → natürlich
    13. zwischen Sätzen + > 4000 ms → zu_lang
    14. Fallback → natürlich (oder stocker wenn innerhalb)
    """
    d = p.dauer_ms

    # 1. Unter Wahrnehmungsschwelle
    if d < MS_IGNORIEREN:
        return "ignorieren"

    # 2. Segment-Grenze (Applaus, Cut, Frage)
    if d > MS_SEGMENT_GRENZE:
        return "segment_grenze"

    # === INNERHALB SATZ ===
    if p.innerhalb_satz:
        if MS_IGNORIEREN <= d < MS_KLEINER_STOCKER:
            return "kleiner_stocker"
        if MS_KLEINER_STOCKER <= d < MS_STOCKER:
            return "stocker"
        if MS_STOCKER <= d < MS_STOCKER_LANG:
            return "stocker_lang"
        if d >= MS_STOCKER_LANG:
            return "zu_lang"
        return "stocker"  # Fallback

    # === ZWISCHEN SÄTZEN ===
    # 7. Atempause hat VORRANG nach langem Satz
    if p.nach_langem_satz and MS_ATEM_MIN <= d <= MS_ATEM_MAX:
        return "atempause"

    # 8. Struktur-Übergang
    if p.an_struktur_uebergang and d >= MS_RHETORISCH:
        return "strukturpause"

    # 9. Sinnpause vor/nach Kernbotschaft
    if (p.vor_kernbotschaft or p.nach_kernbotschaft) and MS_RHETORISCH <= d <= MS_SINNPAUSE_MAX:
        return "sinnpause"

    # 10. Wirkungspause vor/nach Kernbotschaft
    if (p.vor_kernbotschaft or p.nach_kernbotschaft) and MS_WIRKUNG_MIN <= d <= MS_WIRKUNG_MAX:
        return "wirkungspause"

    # 11. Natürlich kurz
    if MS_IGNORIEREN <= d < MS_KLEINER_STOCKER:
        return "natuerlich_kurz"

    # 12. Natürlich
    if MS_KLEINER_STOCKER <= d <= MS_STOCKER_LANG:
        return "natuerlich"

    # 13. Zu lang zwischen Sätzen
    if d > MS_ZU_LANG:
        return "zu_lang"

    # Fallback
    return "natuerlich"


def klassifiziere_alle_pausen(pausen: List[Pause]) -> None:
    """Wendet die Klassifikation auf alle Pausen an."""
    for p in pausen:
        p.typ = klassifiziere_pause(p)


# =============================================================================
# KERNBOTSCHAFT-CHECK (Abschnitt 6.5)
# =============================================================================

def pruefe_kernbotschaften(
    saetze: List[Satz],
    pausen: List[Pause]
) -> List[KernbotschaftCheck]:
    """
    Prüft für jede Kernbotschaft: existiert eine rhetorische Pause (≥ 800 ms)
    davor oder danach?
    """
    ergebnisse = []

    for s in saetze:
        if not s.ist_kernbotschaft:
            continue

        # Suche Pause direkt davor (endet bei Satz-Start)
        pause_davor = None
        pause_danach = None

        # Finde Pause, die direkt vor diesem Satz endet
        for p in pausen:
            if abs(p.end_ms - s.start_ms) < 50:  # 50 ms Toleranz
                pause_davor = p
                break

        # Finde Pause, die direkt nach diesem Satz beginnt
        for p in pausen:
            if abs(p.start_ms - s.end_ms) < 50:
                pause_danach = p
                break

        # Alternative: Suche in einem kleinen Fenster
        if pause_davor is None:
            kandidaten = [p for p in pausen if p.end_ms <= s.start_ms and s.start_ms - p.end_ms < 2000]
            if kandidaten:
                pause_davor = max(kandidaten, key=lambda p: p.dauer_ms)

        if pause_danach is None:
            kandidaten = [p for p in pausen if p.start_ms >= s.end_ms and p.start_ms - s.end_ms < 2000]
            if kandidaten:
                pause_danach = min(kandidaten, key=lambda p: p.start_ms - s.end_ms)

        hat_rhetorisch = False
        if pause_davor and pause_davor.dauer_ms >= MS_RHETORISCH:
            hat_rhetorisch = True
        if pause_danach and pause_danach.dauer_ms >= MS_RHETORISCH:
            hat_rhetorisch = True

        ergebnisse.append(KernbotschaftCheck(
            kernbotschaft_text=s.text[:80] + "..." if len(s.text) > 80 else s.text,
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            pause_davor=pause_davor,
            pause_danach=pause_danach,
            hat_rhetorische_pause=hat_rhetorisch
        ))

    return ergebnisse


# =============================================================================
# SCORING (Abschnitt 6.4)
# =============================================================================

def berechne_d1_rhetorische_qualitaet(pausen: List[Pause]) -> Tuple[int, float, str]:
    """
    D1: Rhetorische Qualität (40%)
    Ratio = Rhetorische Pausen / (Rhetorische Pausen + Stocker)
    """
    rhetorische = [p for p in pausen if p.ist_rhetorisch]
    stocker = [p for p in pausen if p.ist_stocker]

    anzahl_rhetorisch = len(rhetorische)
    anzahl_stocker = len(stocker)

    gesamt = anzahl_rhetorisch + anzahl_stocker

    if gesamt == 0:
        ratio = 0.0
    else:
        ratio = anzahl_rhetorisch / gesamt

    if ratio >= 0.60:
        punkte = 100
        bewertung = "Exzellent"
    elif ratio >= 0.40:
        punkte = 85
        bewertung = "Gut"
    elif ratio >= 0.20:
        punkte = 65
        bewertung = "Ausbaufähig"
    elif ratio >= 0.05:
        punkte = 40
        bewertung = "Wenig"
    else:
        punkte = 20
        bewertung = "Kaum rhetorisch"

    return punkte, ratio, bewertung


def berechne_d2_stocker_rate(pausen: List[Pause], dauer_min: float) -> Tuple[int, float, str]:
    """
    D2: Stocker-Rate (30%)
    Stocker pro Minute.
    """
    anzahl_stocker = len([p for p in pausen if p.ist_stocker])

    if dauer_min > 0:
        rate = anzahl_stocker / dauer_min
    else:
        rate = 0.0

    if rate < 1.0:
        punkte = 100
        bewertung = "Sehr flüssig"
    elif rate < 3.0:
        punkte = 85
        bewertung = "Flüssig"
    elif rate < 5.0:
        punkte = 65
        bewertung = "Akzeptabel"
    elif rate < 8.0:
        punkte = 40
        bewertung = "Hörbar"
    else:
        punkte = 20
        bewertung = "Störend"

    return punkte, rate, bewertung


def berechne_d3_pausen_haushalt(pausen: List[Pause], dauer_min: float) -> Tuple[int, float, str]:
    """
    D3: Gesamtpausen-Haushalt (30%)
    Pausen/Min (alle zählbaren, also ohne ignorieren und segment_grenze).
    Referenz: 8--15 Pausen/Min für gute Sprecher (O'Connell & Kowal + TED).
    """
    anzahl_zaehlbar = len([p for p in pausen if p.ist_zaehlbar])

    if dauer_min > 0:
        rate = anzahl_zaehlbar / dauer_min
    else:
        rate = 0.0

    if 8.0 <= rate <= 15.0:
        punkte = 100
        bewertung = "Optimal"
    elif 5.0 <= rate < 8.0 or 15.0 < rate <= 20.0:
        punkte = 75
        bewertung = "Etwas ungewöhnlich"
    elif rate < 5.0:
        punkte = 40
        bewertung = "Kaum Pausen (gehetzt)"
    else:  # > 20
        punkte = 40
        bewertung = "Zu viele (stockend)"

    return punkte, rate, bewertung


def berechne_gesamtscore(d1: int, d2: int, d3: int) -> int:
    """Gewichteter Gesamtscore 40/30/30, gerundet."""
    score = d1 * GEWICHT_D1 + d2 * GEWICHT_D2 + d3 * GEWICHT_D3
    return int(round(score))


# =============================================================================
# REPORT-GENERIERUNG
# =============================================================================

def generiere_txt_report(
    pausen: List[Pause],
    kb_checks: List[KernbotschaftCheck],
    d1_score: int,
    d1_ratio: float,
    d1_text: str,
    d2_score: int,
    d2_rate: float,
    d2_text: str,
    d3_score: int,
    d3_rate: float,
    d3_text: str,
    gesamt_score: int,
    dauer_min: float,
    transkript_name: str
) -> str:
    """Generiert den menschenlesbaren TXT-Report."""

    lines = []
    lines.append("=" * 70)
    lines.append("PAUSEN-ANALYSE REPORT")
    lines.append("=" * 70)
    lines.append(f"Quelle: {transkript_name}")
    lines.append(f"Präsentationsdauer: {dauer_min:.2f} Minuten")
    lines.append("")

    # Zusammenfassung
    lines.append("-" * 70)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("-" * 70)
    lines.append(f"Gesamt-Score: {gesamt_score}/100")
    lines.append("")
    lines.append(f"  D1 Rhetorische Qualität (40%):  {d1_score}/100 — {d1_text}")
    lines.append(f"      (Ratio rhetorisch / (rhetorisch + Stocker) = {d1_ratio:.2%})")
    lines.append(f"  D2 Stocker-Rate       (30%):  {d2_score}/100 — {d2_text}")
    lines.append(f"      ({d2_rate:.2f} Stocker/Min)")
    lines.append(f"  D3 Pausen-Haushalt    (30%):  {d3_score}/100 — {d3_text}")
    lines.append(f"      ({d3_rate:.2f} Pausen/Min)")
    lines.append("")

    # Detail-Aufschlüsselung
    lines.append("-" * 70)
    lines.append("PAUSEN-ÜBERSICHT NACH TYP")
    lines.append("-" * 70)

    typen = {}
    for p in pausen:
        typen[p.typ] = typen.get(p.typ, 0) + 1

    for typ in sorted(typen.keys(), key=lambda t: -typen[t]):
        anzahl = typen[typ]
        prozent = anzahl / len(pausen) * 100 if pausen else 0
        lines.append(f"  {typ:25s}: {anzahl:4d} ({prozent:5.1f}%)")

    lines.append("")

    # Stocker-Detail (nur hier gescored!)
    stocker_pausen = [p for p in pausen if p.ist_stocker]
    if stocker_pausen:
        lines.append("-" * 70)
        lines.append("STOCKER-DETAIL (einzeln aufgelistet)")
        lines.append("-" * 70)
        lines.append("Hinweis: Stocker werden ausschließlich in diesem Modul bewertet.")
        lines.append("         (Fix v2: Keine Doppelzählung mit sprechfluss_analyse.py)")
        lines.append("")
        for p in stocker_pausen[:20]:  # Max 20 anzeigen
            lines.append(
                f"  [{ms_to_zeitstr(p.start_ms)}] {p.dauer_ms:6.0f} ms — "
                f"{p.typ:18s} | '{p.vorheriges_wort}' -> '{p.naechstes_wort}'"
            )
        if len(stocker_pausen) > 20:
            lines.append(f"  ... und {len(stocker_pausen) - 20} weitere.")
        lines.append("")

    # Rhetorische Pausen
    rhet_pausen = [p for p in pausen if p.ist_rhetorisch]
    if rhet_pausen:
        lines.append("-" * 70)
        lines.append("RHETORISCHE PAUSEN (Sinnpause / Wirkungspause / Strukturpause)")
        lines.append("-" * 70)
        for p in rhet_pausen:
            kontext = []
            if p.vor_kernbotschaft:
                kontext.append("vor KB")
            if p.nach_kernbotschaft:
                kontext.append("nach KB")
            if p.an_struktur_uebergang:
                kontext.append("Struktur-Übergang")

            kontext_str = f" ({', '.join(kontext)})" if kontext else ""
            lines.append(
                f"  [{ms_to_zeitstr(p.start_ms)}] {p.dauer_ms:6.0f} ms — "
                f"{p.typ:18s}{kontext_str}"
            )
        lines.append("")

    # Kernbotschaft-Check
    lines.append("-" * 70)
    lines.append("KERNBOTSCHAFT-CHECK (Abschnitt 6.5)")
    lines.append("-" * 70)
    lines.append("Kernbotschaften sollten von rhetorischen Pausen (>= 800 ms) umrahmt sein.")
    lines.append("")

    if kb_checks:
        fehlende = [k for k in kb_checks if not k.hat_rhetorische_pause]
        lines.append(f"Geprüfte Kernbotschaften: {len(kb_checks)}")
        lines.append(f"Mit rhetorischer Pause:   {len(kb_checks) - len(fehlende)}")
        lines.append(f"OHNE rhetorische Pause:   {len(fehlende)}")
        lines.append("")

        if fehlende:
            lines.append("Fehlende rhetorische Pausen bei:")
            for k in fehlende:
                lines.append(f"  • [{ms_to_zeitstr(k.start_ms)}] {k.text}")
            lines.append("")
    else:
        lines.append("Keine Kernbotschaften in der Inhaltsanalyse gefunden.")
        lines.append("")

    # Studien-Referenzen
    lines.append("-" * 70)
    lines.append("STUDIEN-REFERENZEN")
    lines.append("-" * 70)
    lines.append("• Campione & Véronis (2002): Pausenverteilung, Cluster bei 150/500/1500 ms")
    lines.append("• Heldner & Edlund (2010): 180 ms als kleinste sinnvolle Pausengrenze")
    lines.append("• O'Connell & Kowal (1983): Mean-Pausendauer 940 ms bei Storytelling")
    lines.append("• Karrierebibel / rhetorik-online.de: Sinnpause 1--2 Sek, Wirkungspause 2--4 Sek")
    lines.append("• folienwerke.ch: Publikum nimmt Pausen ~5x kürzer wahr als der Sprecher")
    lines.append("• Hieke (1983): Psychologisch funktionale Pausen ab 130 ms")
    lines.append("")
    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def analyse_pausen(
    transkript_pfad: Path,
    inhalt_pfad: Optional[Path] = None,
    output_json_pfad: Optional[Path] = None,
    output_txt_pfad: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Haupt-Einstiegspunkt für die Pausen-Analyse.

    Args:
        transkript_pfad: Pfad zum Transkript (Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm)
        inhalt_pfad: Pfad zu inhalt_analyse_output.json (optional, aber empfohlen)
        output_json_pfad: Zielpfad für JSON-Output
        output_txt_pfad: Zielpfad für TXT-Report

    Returns:
        Dictionary mit allen Ergebnissen (für gesamtscore.py)
    """

    print(f"[pausen_analyse] Starte Analyse: {transkript_pfad.name}")

    # 1. Transkript parsen
    woerter = parse_transkript(transkript_pfad)
    if not woerter:
        raise ValueError("Keine Wörter im Transkript gefunden.")

    print(f"[pausen_analyse] {len(woerter)} Wörter geladen.")

    # 2. Inhaltsanalyse laden
    inhalt_data = None
    if inhalt_pfad and inhalt_pfad.exists():
        inhalt_data = lade_inhalt_analyse(inhalt_pfad)
        if inhalt_data:
            print("[pausen_analyse] Inhaltsanalyse geladen.")

    # 3. Sätze extrahieren und anreichern
    saetze = extrahiere_saetze(woerter, inhalt_data)
    print(f"[pausen_analyse] {len(saetze)} Sätze erkannt.")

    markiere_kernbotschaften(saetze, inhalt_data)
    markiere_struktur_uebergaenge(saetze, inhalt_data)

    kernbotschaft_count = sum(1 for s in saetze if s.ist_kernbotschaft)
    print(f"[pausen_analyse] {kernbotschaft_count} Kernbotschaften markiert.")

    # 4. Pausen finden und klassifizieren
    pausen = finde_pausen(woerter, saetze)
    klassifiziere_alle_pausen(pausen)

    zaehlbare = [p for p in pausen if p.ist_zaehlbar]
    stocker = [p for p in pausen if p.ist_stocker]
    rhetorische = [p for p in pausen if p.ist_rhetorisch]

    print(f"[pausen_analyse] {len(pausen)} Lücken gefunden.")
    print(f"[pausen_analyse]   -> {len(zaehlbare)} zählbare Pausen")
    print(f"[pausen_analyse]   -> {len(stocker)} Stocker")
    print(f"[pausen_analyse]   -> {len(rhetorische)} rhetorische Pausen")

    # 5. Präsentationsdauer
    gesamt_dauer_ms = woerter[-1].end_ms - woerter[0].start_ms
    dauer_min = gesamt_dauer_ms / 60000.0

    # 6. Kernbotschaft-Check
    kb_checks = pruefe_kernbotschaften(saetze, pausen)
    fehlende_kb = [k for k in kb_checks if not k.hat_rhetorische_pause]

    # 7. Scoring
    d1_score, d1_ratio, d1_text = berechne_d1_rhetorische_qualitaet(pausen)
    d2_score, d2_rate, d2_text = berechne_d2_stocker_rate(pausen, dauer_min)
    d3_score, d3_rate, d3_text = berechne_d3_pausen_haushalt(pausen, dauer_min)
    gesamt_score = berechne_gesamtscore(d1_score, d2_score, d3_score)

    print(f"[pausen_analyse] Scoring: D1={d1_score}, D2={d2_score}, D3={d3_score}, Gesamt={gesamt_score}")

    # 8. JSON-Output vorbereiten
    output_data = {
        "modul": "pausen_analyse",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input": str(transkript_pfad),
        "meta": {
            "woerter_gesamt": len(woerter),
            "saetze_gesamt": len(saetze),
            "praesentationsdauer_min": round(dauer_min, 3),
            "praesentationsdauer_ms": round(gesamt_dauer_ms, 3),
        },
        "pausen": [p.to_dict() for p in pausen],
        "statistiken": {
            "anzahl_pausen": len(pausen),
            "anzahl_zaehlbar": len(zaehlbare),
            "anzahl_stocker": len(stocker),
            "anzahl_rhetorisch": len(rhetorische),
            "stocker_rate_pro_min": round(d2_rate, 2),
            "pausen_rate_pro_min": round(d3_rate, 2),
            "rhetorisch_ratio": round(d1_ratio, 4),
        },
        "kernbotschaft_check": {
            "gesamt": len(kb_checks),
            "mit_rhetorischer_pause": len(kb_checks) - len(fehlende_kb),
            "ohne_rhetorische_pause": len(fehlende_kb),
            "details": [k.to_dict() for k in kb_checks]
        },
        "scoring": {
            "d1_rhetorische_qualitaet": {
                "gewichtung": GEWICHT_D1,
                "punkte": d1_score,
                "bewertung": d1_text,
                "ratio": round(d1_ratio, 4)
            },
            "d2_stocker_rate": {
                "gewichtung": GEWICHT_D2,
                "punkte": d2_score,
                "bewertung": d2_text,
                "rate_pro_min": round(d2_rate, 2)
            },
            "d3_pausen_haushalt": {
                "gewichtung": GEWICHT_D3,
                "punkte": d3_score,
                "bewertung": d3_text,
                "rate_pro_min": round(d3_rate, 2)
            },
            "gesamtscore": gesamt_score
        }
    }

    # 9. Speichern
    if output_json_pfad:
        output_json_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_pfad, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[pausen_analyse] JSON gespeichert: {output_json_pfad}")

    if output_txt_pfad:
        output_txt_pfad.parent.mkdir(parents=True, exist_ok=True)
        report = generiere_txt_report(
            pausen, kb_checks,
            d1_score, d1_ratio, d1_text,
            d2_score, d2_rate, d2_text,
            d3_score, d3_rate, d3_text,
            gesamt_score, dauer_min,
            transkript_pfad.name
        )
        with open(output_txt_pfad, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[pausen_analyse] Report gespeichert: {output_txt_pfad}")

    print(f"[pausen_analyse] Fertig. Gesamt-Score: {gesamt_score}/100")

    return output_data


# =============================================================================
# CLI / MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pausen-Analyse für Präsentationsbewertungs-AI"
    )
    parser.add_argument(
        "transkript",
        type=str,
        help="Pfad zum Transkript (Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm)"
    )
    parser.add_argument(
        "--inhalt",
        type=str,
        default=None,
        help="Pfad zu inhalt_analyse_output.json (optional, empfohlen)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="zwischen_output/pausen_analyse_output.json",
        help="Zielpfad für JSON-Output"
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Zielpfad für TXT-Report (Default: reports/pausen/pausen_report_TIMESTAMP.txt)"
    )

    args = parser.parse_args()

    transkript_pfad = Path(args.transkript)
    inhalt_pfad = Path(args.inhalt) if args.inhalt else DEFAULT_INPUT_DIR / "inhalt_analyse_output.json"

    output_json = Path(args.output_json)

    if args.output_txt:
        output_txt = Path(args.output_txt)
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_txt = DEFAULT_REPORT_DIR / f"pausen_report_{ts}.txt"

    if not transkript_pfad.exists():
        print(f"[FEHLER] Transkript nicht gefunden: {transkript_pfad}")
        exit(1)

    try:
        analyse_pausen(transkript_pfad, inhalt_pfad, output_json, output_txt)
    except Exception as e:
        print(f"[FEHLER] {e}")
        raise
