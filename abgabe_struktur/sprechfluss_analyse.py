#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sprechfluss_analyse.py
======================
Erkennt Wiederholungen und Abbrüche im Redefluss.
Bewertet kontextuell: Disfluenzen bei Kernbotschaften wiegen schwerer.

Input:
  - Transkript: "Wort HH:MM:SS.mmm HH:MM:SS.mmm"
  - inhalt_analyse_output.json (optional, für Satzgrenzen + Kernbotschaften)
  - pausen_analyse_output.json (optional, für Stocker-Info, NICHT für Scoring)

Output:
  - zwischen_output/sprechfluss_analyse_output.json
  - reports/sprechfluss/sprechfluss_report_[TIMESTAMP].txt"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# KONSTANTEN
# =============================================================================

GEWICHT_D1 = 0.40
GEWICHT_D2 = 0.30
GEWICHT_D3 = 0.30

# D1: Wiederholungs- + Abbruch-Rate pro Minute
# Studien: Bosker et al. 2013, TED-Niveau
D1_PERFECT = 0.0
D1_SEHR_FLUESSIG = 0.5
D1_FLUESSIG = 1.0
D1_AUFFAELLIG = 2.0

# D2: Kernbotschafts-Fluenz (Anteil sauber)
D2_PERFECT = 1.0
D2_SEHR_GUT = 0.80
D2_AKZEPTABEL = 0.60
D2_AUFFAELLIG = 0.40

# D3: Cluster-Anteil (Sätze mit >1 Disfluenz / Sätze mit ≥1 Disfluenz)
D3_ISOLIERT = 0.10
D3_GELEGENTLICH = 0.25
D3_HAEUFIG = 0.50


# =============================================================================
# DATENKLASSEN
# =============================================================================

@dataclass
class Wort:
    text: str
    start_ms: float
    end_ms: float
    index: int = 0


@dataclass
class Satz:
    index: int
    text: str
    start_ms: float
    end_ms: float
    woerter: List[Wort] = field(default_factory=list)
    wortanzahl: int = 0
    ist_kernbotschaft: bool = False


@dataclass
class Disfluenz:
    """Ein Disfluenz-Ereignis: Wiederholung oder Abbruch."""
    typ: str                    # "wiederholung" oder "abbruch"
    position: int               # Wort-Index
    satz_index: int
    wort: str
    wort_gereinigt: str
    start_ms: float
    end_ms: float
    kontext: str = ""           # z.B. "in Kernbotschaft"

    def to_dict(self) -> dict:
        return {
            "typ": self.typ,
            "position": self.position,
            "satz_index": self.satz_index,
            "wort": self.wort,
            "wort_gereinigt": self.wort_gereinigt,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "kontext": self.kontext,
        }


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def zeitstr_to_ms(zeit_str: str) -> float:
    """Parst Zeitstempel zu ms."""
    zeit_str = zeit_str.strip()
    if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", zeit_str):
        h, m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    if re.match(r"^\d{2}:\d{2}\.\d{3}$", zeit_str):
        m, s_ms = zeit_str.split(":")
        s, ms = s_ms.split(".")
        return int(m) * 60000 + int(s) * 1000 + int(ms)
    for fmt in ("%H:%M:%S.%f", "%M:%S.%f", "%H:%M:%S", "%M:%S"):
        try:
            dt = datetime.strptime(zeit_str, fmt)
            return (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
        except ValueError:
            continue
    raise ValueError(f"Unbekanntes Zeitformat: {zeit_str}")


def ms_to_zeitstr(ms: float) -> str:
    """ms zu lesbarem Zeitstempel."""
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
    """Parst Transkript im Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm"""
    woerter = []
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)$")

    with open(transkript_pfad, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                wort_text, start_str, end_str = m.groups()
            else:
                teile = line.split()
                if len(teile) >= 3:
                    wort_text, start_str, end_str = teile[0], teile[-2], teile[-1]
                else:
                    continue
            try:
                start_ms = zeitstr_to_ms(start_str)
                end_ms = zeitstr_to_ms(end_str)
            except ValueError:
                continue
            woerter.append(Wort(text=wort_text, start_ms=start_ms, end_ms=end_ms, index=len(woerter)))

    woerter.sort(key=lambda w: w.start_ms)
    for i, w in enumerate(woerter):
        w.index = i
    return woerter


def lade_json(pfad: Path) -> Optional[Dict]:
    if not pfad.exists():
        return None
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Konnte {pfad} nicht laden: {e}")
        return None


def extrahiere_saetze(woerter: List[Wort], inhalt_data: Optional[Dict]) -> List[Satz]:
    """Extrahiert Sätze aus Inhaltsanalyse oder heuristisch."""
    saetze = []

    if inhalt_data and "satzgrenzen" in inhalt_data:
        raw_saetze = inhalt_data["satzgrenzen"]
        for i, raw in enumerate(raw_saetze):
            start_ms = raw.get("start_ms", raw.get("start"))
            end_ms = raw.get("end_ms", raw.get("end"))
            if isinstance(start_ms, str):
                start_ms = zeitstr_to_ms(start_ms)
            if isinstance(end_ms, str):
                end_ms = zeitstr_to_ms(end_ms)
            if start_ms is None or end_ms is None:
                continue
            satz_woerter = [w for w in woerter if start_ms <= w.start_ms < end_ms]
            saetze.append(Satz(
                index=i,
                text=raw.get("text", ""),
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                woerter=satz_woerter,
                wortanzahl=len(satz_woerter)
            ))

    if not saetze:
        # Heuristik: Satzende bei . ! ?
        aktuelle = []
        idx = 0
        for w in woerter:
            aktuelle.append(w)
            if w.text.rstrip().endswith((".", "!", "?")) and len(aktuelle) > 1:
                saetze.append(Satz(
                    index=idx,
                    text=" ".join(x.text for x in aktuelle),
                    start_ms=aktuelle[0].start_ms,
                    end_ms=aktuelle[-1].end_ms,
                    woerter=list(aktuelle),
                    wortanzahl=len(aktuelle)
                ))
                aktuelle = []
                idx += 1
        if aktuelle:
            saetze.append(Satz(
                index=idx,
                text=" ".join(x.text for x in aktuelle),
                start_ms=aktuelle[0].start_ms,
                end_ms=aktuelle[-1].end_ms,
                woerter=aktuelle,
                wortanzahl=len(aktuelle)
            ))

    return saetze


def markiere_kernbotschaften(saetze: List[Satz], inhalt_data: Optional[Dict]) -> None:
    if not inhalt_data or "kernbotschaften" not in inhalt_data:
        return
    for kb in inhalt_data["kernbotschaften"]:
        kb_start = kb.get("start_ms", kb.get("start"))
        kb_end = kb.get("end_ms", kb.get("end"))
        if isinstance(kb_start, str):
            kb_start = zeitstr_to_ms(kb_start)
        if isinstance(kb_end, str):
            kb_end = zeitstr_to_ms(kb_end)
        if kb_start is None or kb_end is None:
            continue
        for s in saetze:
            if s.start_ms <= float(kb_end) and s.end_ms >= float(kb_start):
                s.ist_kernbotschaft = True


# =============================================================================
# DISFLUENZ-ERKENNUNG
# =============================================================================

def bereinige_wort(wort: str) -> str:
    """Entfernt Interpunktion für den Vergleich.

    Fix: Die Vorgänger-Version verwendete r"[^\\w\\-äöüÄÖÜß]", wodurch die
    doppelten Backslashes und der '-' zwischen '\\' und 'ä' versehentlich eine
    Range von ord(92) bis ord(228) bildeten. Das behielt alle
    Kleinbuchstaben, strippte aber Grossbuchstaben — satzinitiale
    Wiederholungen wie "Also also" wurden nicht als Match erkannt.
    Jetzt: nur einfache Backslashes.
    """
    return re.sub(r"[^\w\-äöüÄÖÜß]", "", wort, flags=re.UNICODE).lower()


def ist_abbruch(wort: str) -> bool:
    """
    Prüft ob ein Wort ein Abbruch ist.

    Fix: Die Heuristik `"-" in wort and len(wort) < 6` markierte auch
    normale Wörter mit Bindestrich fälschlich als Abbruch (z.B. "e-Mail",
    "5-fach", "T-Shirt"). Whisper-Transkripte enthalten praktisch keine
    Fragment-Marker ausser "wort-" am Wortende. Deshalb nur noch: Wort
    endet auf "-" (ohne dass weiteres folgt).
    """
    stripped = wort.rstrip(".,;:!?\"'()[]")
    return stripped.endswith("-") and len(stripped) > 1


def erkenne_disfluenzen(woerter: List[Wort], saetze: List[Satz]) -> List[Disfluenz]:
    """
    Erkennt Wiederholungen und Abbrüche.

    Wiederholung: wort_N == wort_N+1 (case-insensitiv, ohne Interpunktion)
    Abbruch: Wort endet auf "-" oder Fragment-Heuristik
    """
    disfluenzen = []

    # Schneller Lookup: Wort-Index -> Satz
    wort_zu_satz = {}
    for s in saetze:
        for w in s.woerter:
            wort_zu_satz[w.index] = s

    # 1. Wiederholungen erkennen
    for i in range(len(woerter) - 1):
        w1 = woerter[i]
        w2 = woerter[i + 1]

        w1_clean = bereinige_wort(w1.text)
        w2_clean = bereinige_wort(w2.text)

        if w1_clean and w1_clean == w2_clean and len(w1_clean) > 1:
            s = wort_zu_satz.get(w1.index)
            kontext = "in Kernbotschaft" if s and s.ist_kernbotschaft else ""
            disfluenzen.append(Disfluenz(
                typ="wiederholung",
                position=i,
                satz_index=s.index if s else -1,
                wort=w1.text,
                wort_gereinigt=w1_clean,
                start_ms=w1.start_ms,
                end_ms=w2.end_ms,
                kontext=kontext
            ))

    # 2. Abbrüche erkennen
    for w in woerter:
        if ist_abbruch(w.text):
            s = wort_zu_satz.get(w.index)
            kontext = "in Kernbotschaft" if s and s.ist_kernbotschaft else ""
            disfluenzen.append(Disfluenz(
                typ="abbruch",
                position=w.index,
                satz_index=s.index if s else -1,
                wort=w.text,
                wort_gereinigt=bereinige_wort(w.text),
                start_ms=w.start_ms,
                end_ms=w.end_ms,
                kontext=kontext
            ))

    # Sortieren nach Position
    disfluenzen.sort(key=lambda d: d.position)
    return disfluenzen


# =============================================================================
# SCORING
# =============================================================================

def berechne_d1(disfluenzen: List[Disfluenz], dauer_min: float) -> Tuple[int, float, str]:
    """
    D1: Wiederholungs- + Abbruch-Rate (40%)
    Ereignisse pro Minute.
    """
    anzahl = len(disfluenzen)
    if dauer_min > 0:
        rate = anzahl / dauer_min
    else:
        rate = 0.0

    if rate == 0:
        punkte = 100
        bewertung = "Perfekt"
    elif rate < D1_SEHR_FLUESSIG:
        punkte = 90
        bewertung = "Sehr flüssig"
    elif rate < D1_FLUESSIG:
        punkte = 70
        bewertung = "Flüssig"
    elif rate < D1_AUFFAELLIG:
        punkte = 40
        bewertung = "Auffällig"
    else:
        punkte = 20
        bewertung = "Störend"

    return punkte, rate, bewertung


def berechne_d2(saetze: List[Satz], disfluenzen: List[Disfluenz]) -> Tuple[int, float, str]:
    """
    D2: Kernbotschafts-Fluenz (30%)
    Prozent der Kernbotschaften ohne Wiederholung/Abbruch.
    """
    kernbotschaften = [s for s in saetze if s.ist_kernbotschaft]
    if not kernbotschaften:
        return 100, 1.0, "Keine Kernbotschaften gefunden"

    # Disfluenzen nach Satz-Index gruppieren
    disfluenz_saetze = set(d.satz_index for d in disfluenzen)

    sauber = sum(1 for s in kernbotschaften if s.index not in disfluenz_saetze)
    anteil = sauber / len(kernbotschaften)

    if anteil >= D2_PERFECT:
        punkte = 100
        bewertung = "Perfekt"
    elif anteil >= D2_SEHR_GUT:
        punkte = 85
        bewertung = "Sehr gut"
    elif anteil >= D2_AKZEPTABEL:
        punkte = 65
        bewertung = "Akzeptabel"
    elif anteil >= D2_AUFFAELLIG:
        punkte = 40
        bewertung = "Auffällig"
    else:
        punkte = 20
        bewertung = "Kritisch"

    return punkte, anteil, bewertung


def berechne_d3(saetze: List[Satz], disfluenzen: List[Disfluenz]) -> Tuple[int, float, str]:
    """
    D3: Länge der Disfluenz-Cluster (30%)
    Anteil der Sätze mit >1 Disfluenz-Ereignis an allen Sätzen mit ≥1 Ereignis.
    """
    if not disfluenzen:
        return 100, 0.0, "Keine Disfluenzen"

    # Gruppiere Disfluenzen nach Satz
    satz_disfluenzen: Dict[int, int] = {}
    for d in disfluenzen:
        satz_disfluenzen[d.satz_index] = satz_disfluenzen.get(d.satz_index, 0) + 1

    saetze_mit_disfluenz = sum(1 for count in satz_disfluenzen.values() if count >= 1)
    saetze_mit_cluster = sum(1 for count in satz_disfluenzen.values() if count > 1)

    if saetze_mit_disfluenz == 0:
        return 100, 0.0, "Keine betroffenen Sätze"

    cluster_anteil = saetze_mit_cluster / saetze_mit_disfluenz

    if cluster_anteil < D3_ISOLIERT:
        punkte = 100
        bewertung = "Isolierte Aussetzer"
    elif cluster_anteil < D3_GELEGENTLICH:
        punkte = 75
        bewertung = "Gelegentliche Cluster"
    elif cluster_anteil < D3_HAEUFIG:
        punkte = 45
        bewertung = "Häufige Cluster"
    else:
        punkte = 20
        bewertung = "Sprachplanungs-Problem"

    return punkte, cluster_anteil, bewertung


def berechne_gesamtscore(d1: int, d2: int, d3: int) -> int:
    score = d1 * GEWICHT_D1 + d2 * GEWICHT_D2 + d3 * GEWICHT_D3
    return int(round(score))


# =============================================================================
# STOCKER-INFO (nur informativ, Fix v2)
# =============================================================================

def lade_stocker_info(pausen_data: Optional[Dict]) -> Dict[str, Any]:
    """Liest Stocker-Statistiken aus pausen_analyse_output.json — nur für den Report."""
    if not pausen_data:
        return {"verfuegbar": False, "stocker_anzahl": 0, "stocker_rate": 0.0}

    statistiken = pausen_data.get("statistiken", {})
    return {
        "verfuegbar": True,
        "stocker_anzahl": statistiken.get("anzahl_stocker", 0),
        "stocker_rate": statistiken.get("stocker_rate_pro_min", 0.0),
        "stocker_details": [
            p for p in pausen_data.get("pausen", [])
            if p.get("typ") in ("kleiner_stocker", "stocker", "stocker_lang", "zu_lang")
        ]
    }


# =============================================================================
# REPORT
# =============================================================================

def generiere_report(
    disfluenzen: List[Disfluenz],
    d1_score: int, d1_rate: float, d1_text: str,
    d2_score: int, d2_anteil: float, d2_text: str,
    d3_score: int, d3_anteil: float, d3_text: str,
    gesamt_score: int,
    dauer_min: float,
    transkript_name: str,
    stocker_info: Dict[str, Any]
) -> str:

    lines = []
    lines.append("=" * 70)
    lines.append("SPRECHFLUSS-ANALYSE REPORT")
    lines.append("=" * 70)
    lines.append(f"Quelle: {transkript_name}")
    lines.append(f"Präsentationsdauer: {dauer_min:.2f} Minuten")
    lines.append("")

    lines.append("-" * 70)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("-" * 70)
    lines.append(f"Gesamt-Score: {gesamt_score}/100")
    lines.append("")
    lines.append(f"  D1 Wiederholung/Abbruch-Rate (40%): {d1_score}/100 — {d1_text}")
    lines.append(f"      ({d1_rate:.2f} Ereignisse/Min)")
    lines.append(f"  D2 Kernbotschafts-Fluenz      (30%): {d2_score}/100 — {d2_text}")
    lines.append(f"      ({d2_anteil:.1%} der Kernbotschaften sauber)")
    lines.append(f"  D3 Disfluenz-Cluster         (30%): {d3_score}/100 — {d3_text}")
    lines.append(f"      ({d3_anteil:.1%} der betroffenen Sätze haben Cluster)")
    lines.append("")

    # Stocker-Info (informativ)
    lines.append("-" * 70)
    lines.append("STOCKER-INFO (aus pausen_analyse.py — nur informativ, Fix v2)")
    lines.append("-" * 70)
    if stocker_info["verfuegbar"]:
        lines.append(f"Stocker-Anzahl:   {stocker_info['stocker_anzahl']}")
        lines.append(f"Stocker-Rate:     {stocker_info['stocker_rate']:.2f}/Min")
        lines.append("Hinweis: Stocker werden ausschließlich in pausen_analyse.py gescored.")
    else:
        lines.append("Keine pausen_analyse_output.json gefunden.")
    lines.append("")

    # Disfluenzen-Detail
    if disfluenzen:
        lines.append("-" * 70)
        lines.append("DISFLUENZEN (Wiederholungen + Abbrüche)")
        lines.append("-" * 70)
        wiederholungen = [d for d in disfluenzen if d.typ == "wiederholung"]
        abbrueche = [d for d in disfluenzen if d.typ == "abbruch"]

        lines.append(f"Wiederholungen: {len(wiederholungen)}")
        lines.append(f"Abbrüche:       {len(abbrueche)}")
        lines.append("")

        for d in disfluenzen:
            kb_mark = " [KB]" if d.kontext else ""
            lines.append(
                f"  [{ms_to_zeitstr(d.start_ms)}] {d.typ:15s} — "
                f"'{d.wort}'{kb_mark}"
            )
        lines.append("")
    else:
        lines.append("-" * 70)
        lines.append("✅ Keine Disfluenzen erkannt.")
        lines.append("-" * 70)
        lines.append("")

    lines.append("-" * 70)
    lines.append("STUDIEN-REFERENZEN")
    lines.append("-" * 70)
    lines.append("• Bosker, Pinget, Quené, Sanders, De Jong (2013): What makes speech sound fluent?")
    lines.append("• Clark & Fox Tree (2002): Filler-as-word Hypothese")
    lines.append("• Laserna (2014): 3--4× mehr Diskursmarker bei jungen Erwachsenen")
    lines.append("")
    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def analyse_sprechfluss(
    transkript_pfad: Path,
    inhalt_pfad: Optional[Path] = None,
    pausen_pfad: Optional[Path] = None,
    output_json_pfad: Optional[Path] = None,
    output_txt_pfad: Optional[Path] = None
) -> Dict[str, Any]:

    print(f"[sprechfluss] Starte Analyse: {transkript_pfad.name}")

    # 1. Daten laden
    woerter = parse_transkript(transkript_pfad)
    if not woerter:
        raise ValueError("Keine Wörter im Transkript.")
    print(f"[sprechfluss] {len(woerter)} Wörter geladen.")

    inhalt_data = lade_json(inhalt_pfad) if inhalt_pfad else None
    pausen_data = lade_json(pausen_pfad) if pausen_pfad else None

    # 2. Sätze extrahieren
    saetze = extrahiere_saetze(woerter, inhalt_data)
    markiere_kernbotschaften(saetze, inhalt_data)
    print(f"[sprechfluss] {len(saetze)} Sätze, {sum(1 for s in saetze if s.ist_kernbotschaft)} Kernbotschaften.")

    # 3. Disfluenzen erkennen
    disfluenzen = erkenne_disfluenzen(woerter, saetze)
    wiederholungen = [d for d in disfluenzen if d.typ == "wiederholung"]
    abbrueche = [d for d in disfluenzen if d.typ == "abbruch"]
    print(f"[sprechfluss] {len(disfluenzen)} Disfluenzen: {len(wiederholungen)} Wiederholungen, {len(abbrueche)} Abbrüche.")

    # 4. Dauer & Scoring
    gesamt_dauer_ms = woerter[-1].end_ms - woerter[0].start_ms
    dauer_min = gesamt_dauer_ms / 60000.0
    if dauer_min <= 0:
        dauer_min = 1.0

    d1_score, d1_rate, d1_text = berechne_d1(disfluenzen, dauer_min)
    d2_score, d2_anteil, d2_text = berechne_d2(saetze, disfluenzen)
    d3_score, d3_anteil, d3_text = berechne_d3(saetze, disfluenzen)
    gesamt_score = berechne_gesamtscore(d1_score, d2_score, d3_score)

    print(f"[sprechfluss] Scoring: D1={d1_score}, D2={d2_score}, D3={d3_score}, Gesamt={gesamt_score}")

    # 5. Stocker-Info (nur informativ)
    stocker_info = lade_stocker_info(pausen_data)

    # 6. Output
    output_data = {
        "modul": "sprechfluss_analyse",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input": str(transkript_pfad),
        "meta": {
            "woerter_gesamt": len(woerter),
            "saetze_gesamt": len(saetze),
            "praesentationsdauer_min": round(dauer_min, 3),
        },
        "disfluenzen": [d.to_dict() for d in disfluenzen],
        "statistiken": {
            "anzahl_disfluenzen": len(disfluenzen),
            "anzahl_wiederholungen": len(wiederholungen),
            "anzahl_abbrueche": len(abbrueche),
            "disfluenz_rate_pro_min": round(d1_rate, 2),
        },
        "stocker_info": {
            "aus_pausen_analyse": stocker_info["verfuegbar"],
            "stocker_anzahl": stocker_info["stocker_anzahl"],
            "stocker_rate": stocker_info["stocker_rate"],
            "hinweis": "Stocker werden ausschließlich in pausen_analyse.py gescored (Fix v2)."
        },
        "scoring": {
            "d1_wiederholung_abbruch_rate": {
                "gewichtung": GEWICHT_D1,
                "punkte": d1_score,
                "bewertung": d1_text,
                "rate_pro_min": round(d1_rate, 2)
            },
            "d2_kernbotschafts_fluenz": {
                "gewichtung": GEWICHT_D2,
                "punkte": d2_score,
                "bewertung": d2_text,
                "anteil_sauber": round(d2_anteil, 4)
            },
            "d3_disfluenz_cluster": {
                "gewichtung": GEWICHT_D3,
                "punkte": d3_score,
                "bewertung": d3_text,
                "cluster_anteil": round(d3_anteil, 4)
            },
            "gesamtscore": gesamt_score
        }
    }

    if output_json_pfad:
        output_json_pfad.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_pfad, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"[sprechfluss] JSON gespeichert: {output_json_pfad}")

    if output_txt_pfad:
        output_txt_pfad.parent.mkdir(parents=True, exist_ok=True)
        report = generiere_report(
            disfluenzen,
            d1_score, d1_rate, d1_text,
            d2_score, d2_anteil, d2_text,
            d3_score, d3_anteil, d3_text,
            gesamt_score, dauer_min,
            transkript_pfad.name,
            stocker_info
        )
        with open(output_txt_pfad, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[sprechfluss] Report gespeichert: {output_txt_pfad}")

    print(f"[sprechfluss] Fertig. Gesamt-Score: {gesamt_score}/100")
    return output_data


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sprechfluss-Analyse für Präsentationsbewertungs-AI")
    parser.add_argument("transkript", type=str, help="Pfad zum Transkript")
    parser.add_argument("--inhalt", type=str, default=None, help="Pfad zu inhalt_analyse_output.json")
    parser.add_argument("--pausen", type=str, default=None, help="Pfad zu pausen_analyse_output.json (informativ)")
    parser.add_argument("--output-json", type=str, default="zwischen_output/sprechfluss_analyse_output.json")
    parser.add_argument("--output-txt", type=str, default=None)

    args = parser.parse_args()

    transkript = Path(args.transkript)
    inhalt = Path(args.inhalt) if args.inhalt else None
    pausen = Path(args.pausen) if args.pausen else None
    out_json = Path(args.output_json)

    if args.output_txt:
        out_txt = Path(args.output_txt)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt = Path("reports/sprechfluss") / f"sprechfluss_report_{ts}.txt"

    if not transkript.exists():
        print(f"[FEHLER] Transkript nicht gefunden: {transkript}")
        exit(1)

    try:
        analyse_sprechfluss(transkript, inhalt, pausen, out_json, out_txt)
    except Exception as e:
        print(f"[FEHLER] {e}")
        raise
