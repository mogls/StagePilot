#!/usr/bin/env python3
"""füllwörter_analyse_v2.py
==============================================================================

4 Kategorien mit sprachwissenschaftlichen Bezeichnungen:
  1. Verzögerungslaute (Hesitation Sounds)     — Belz 2021, Uni Trier 2023
  2. Heckenausdrücke (Hedges)                  — Wellner 2023, Prince 1982
  3. Modalpartikeln + Diskursmarker             — Weinrich, Thurmair, Meibauer
  4. Intensivierer / Gradpartikeln              — Duden Grammatik

Bewertung: 2 Metriken parallel
  A. Rate pro Minute                            — Quantified Communications
  B. Dichte (Prozent aller Wörter)             — NCT05444114 (N=182)

Score: Malus-System (Basis 100, Abzüge je Kategorie)"""

import re
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime
from collections import Counter

import tkinter as tk
from tkinter import filedialog


# ============================================================================
# PFADE
# ============================================================================

PROJEKT_ROOT = Path(__file__).resolve().parent.parent
JSON_INHALT_PFAD = PROJEKT_ROOT / "zwischen_output" / "inhalt_analyse_output.json"
JSON_SPRECHTEMPO_PFAD = PROJEKT_ROOT / "zwischen_output" / "sprechtempo_analyse_output.json"
JSON_OUTPUT_PFAD = PROJEKT_ROOT / "zwischen_output" / "fuellwoerter_analyse_output.json"
REPORT_ORDNER = PROJEKT_ROOT / "reports" / "fuellwoerter"


# ============================================================================
# WORTLISTEN — Kategorie 1: Verzögerungslaute
# Quelle: Belz (2021), Uni Trier 2023 (Braun et al.), Wikipedia-Linguistik
# ============================================================================

VERZOEGERUNGSLAUTE = {
    # v2-Fix: 'eh' und 'ehm' ergaenzt — Whisper transkribiert deutsche
    # Verzögerungen häufig ohne Umlaut. Alle Varianten müssen abgedeckt sein.
    'äh', 'ähm', 'öh', 'öhm', 'hm', 'hmm', 'mh', 'mhm', 'mmm',
    'ehh', 'ähh', 'eh', 'ehm',
}


# ============================================================================
# WORTLISTEN — Kategorie 2: Heckenausdrücke (Hedges)
# Quellen: Wellner (2023) "Towards a taxonomy of hedging devices",
#          Uni Leipzig Lexikologie, CAU Kiel Semantik, Prince et al. (1982)
# ============================================================================

HECKENAUSDRUECKE_1 = {
    # Modaladverbien (Wellner-Kategorie)
    'eigentlich', 'sozusagen', 'quasi', 'gewissermaßen', 'einigermaßen',
    'eher', 'strenggenommen', 'vermutlich', 'wahrscheinlich',
    # Unbestimmte Ausdrücke
    'irgendwie', 'irgendwas', 'irgendwo', 'irgendwann', 'irgendein',
    'etwas', 'ungefähr',
    # Modale Absicherung
    'vielleicht', 'allenfalls',
}

HECKENAUSDRUECKE_2 = {
    'ein bisschen', 'ein wenig',
    'an sich', 'im prinzip', 'in etwa',
    'meines erachtens',
}

HECKENAUSDRUECKE_3 = {
    'mehr oder weniger', 'in gewisser weise',
    'im weitesten sinne', 'man könnte sagen',
}


# ============================================================================
# WORTLISTEN — Kategorie 3: Modalpartikeln + Diskursmarker
# Quellen: Weinrich "Textgrammatik", Thurmair (1989), Meibauer (1994),
#          Helbig (1990), Duden Grammatik
# ============================================================================

MODALPARTIKELN_1 = {
    # Klassische Modalpartikeln (Weinrich/Thurmair/Duden)
    'halt', 'eben', 'doch', 'mal', 'ja', 'denn', 'schon', 'wohl', 'auch', 'bloß',
    # Diskursmarker / Gliederungssignale
    'also', 'so', 'nun', 'genau', 'tja', 'na',
    # Bejahungspartikeln (Filterung über Bejahungs-Kontext)
    'klar', 'richtig', 'stimmt', 'selbstverständlich', 'natürlich',
    # Konnektor-Füllwörter
    'sprich', 'nämlich', 'beziehungsweise',
    # Abschluss-Floskeln (1-Wort)
    'letztlich', 'sowieso', 'ohnedies',
}

MODALPARTIKELN_2 = {
    'und zwar', 'das heißt',
    'wie gesagt', 'kurz gesagt',
    'im grunde', 'letzten endes', 'im endeffekt',
    'wie auch immer',
}

MODALPARTIKELN_3 = set()  # aktuell keine 3-Wort-Modalpartikeln

MODALPARTIKELN_4 = {
    'im großen und ganzen',
    'am ende des tages',
}


# ============================================================================
# WORTLISTEN — Kategorie 4: Intensivierer / Gradpartikeln
# Quellen: Duden Grammatik, Wellner (Gradpartikeln als eigene Klasse)
# ============================================================================

INTENSIVIERER_1 = {
    'absolut', 'total', 'definitiv', 'eindeutig', 'komplett',
    'sehr', 'wirklich', 'echt', 'durchaus', 'unbedingt',
}

INTENSIVIERER_3 = {
    'auf jeden fall', 'ganz und gar',
}


# ============================================================================
# BEJAHUNGS-KONTEXT
# 'ja'/'doch' als alleinstehende Antwort werden NICHT als Füllwort gezählt
# ============================================================================

BEJAHUNGS_KANDIDATEN = {'ja', 'doch'}


# ============================================================================
# BEWERTUNGS-SKALEN
# Format: (max_pro_min, level_key, icon, label, malus)
# Der Malus geht in den Score (100 - sum_of_malus).
# Grenze: pro_min <= max_pro_min → dieser Eintrag greift.
# ============================================================================

# Kategorie 1: Verzögerungslaute
# Quelle: Quantified Communications (Ideal 1/min, Ø 5/min, kritisch >7/min)
# + Christenfeld (UCSD): negative Publikums-Wahrnehmung
VERZOEGERUNG_SKALA = [
    (0.001, 'optimal',    '✅', 'Optimal',      0),
    (1.0,   'sehr_gut',   '✅', 'Sehr gut',     3),
    (3.0,   'akzeptabel', '🟡', 'Akzeptabel',   10),
    (5.0,   'auffaellig', '🟠', 'Auffällig',    25),
    (999,   'stoerend',   '🔴', 'Störend',      40),
]

# Kategorie 2: Heckenausdrücke
# Quelle: Rhetorik-Praxis, Wellner-Taxonomie (keine harte Studien-Schwelle,
# aber weitgehender Konsens: sparsam einsetzen)
HECKEN_SKALA = [
    (0.001, 'zu_steif',   '⚠️', 'Zu steif',           3),
    (2.0,   'ideal',      '✅', 'Ideal',               0),
    (4.0,   'akzeptabel', '🟡', 'Akzeptabel',          5),
    (6.0,   'zu_viele',   '🟠', 'Zu viele',           12),
    (999,   'kritisch',   '🔴', 'Kompetenz-Verlust',  20),
]

# Kategorie 3: Modalpartikeln
# Quelle: Weinrich/Thurmair — Modalpartikeln sind natürliches Element
# der gesprochenen Sprache, aber Übermass wirkt unstrukturiert.
# TED-Talk-Kalibrierung: gute Sprecher 3-7/min
MODALPARTIKEL_SKALA = [
    (2.0,   'zu_steif',    '⚠️', 'Zu steif',    3),
    (7.0,   'ideal',       '✅', 'Ideal',        0),
    (12.0,  'erhoeht',     '🟡', 'Erhöht',       3),
    (18.0,  'auffaellig',  '🟠', 'Auffällig',    8),
    (999,   'dominierend', '🔴', 'Dominierend', 15),
]

# Kategorie 4: Intensivierer
# Quelle: Toastmasters "Verbal Intensifiers" — sparsam wirkungsvoller
INTENSIVIERER_SKALA = [
    (1.0,   'zurueckhaltend', '✅', 'Zurückhaltend', 0),
    (3.0,   'gut',            '✅', 'Gut',           0),
    (5.0,   'viele',          '🟡', 'Viele',         3),
    (8.0,   'zu_viele',       '🟠', 'Zu viele',      6),
    (999,   'inflationaer',   '🔴', 'Inflationär',  10),
]


# ============================================================================
# METRIK B: DICHTE (Prozent aller Wörter)
# Quelle: NCT05444114 (N=182): Mittelphrase 1.5% + andere 4.4% = ~5.9% Ø
# 1 SD Dispersion: bis ~9%. Über 10% ist deutlich auffällig.
# ============================================================================

def dichte_malus(prozent):
    """Malus basierend auf Prozent aller Wörter, die Füllwörter sind."""
    if prozent < 3.0:
        return 0  # deutlich unter Durchschnitt
    if prozent < 6.0:
        return 0  # im NCT-Durchschnitt
    if prozent < 10.0:
        return 3  # 1 SD über Mittelwert
    return 10     # 2+ SD über Mittelwert


def dichte_label(prozent):
    if prozent < 3.0:  return ('✅', 'Ideal (unter NCT-Ø)')
    if prozent < 6.0:  return ('✅', 'Im NCT-Durchschnitt')
    if prozent < 10.0: return ('🟡', 'Leicht erhöht (1 SD)')
    return ('🔴', 'Deutlich auffällig (>2 SD)')


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def zeit_zu_sekunden(zeitstring):
    """HH:MM:SS.mmm -> Sekunden."""
    teile = zeitstring.strip().split(':')
    return int(teile[0]) * 3600 + int(teile[1]) * 60 + float(teile[2])


def hole_ende(eintrag):
    """Feldname 'end' bevorzugt, 'ende' als Fallback."""
    if 'end' in eintrag:
        return eintrag['end']
    if 'ende' in eintrag:
        return eintrag['ende']
    raise KeyError("Weder 'end' noch 'ende' im Eintrag")


def normalisiere_wort(wort):
    """Interpunktion entfernen, lowercase."""
    return re.sub(r"[.,!?;:\"'()\[\]{}„“”«»‚‘’…–—]", '', wort.lower()).strip()


# ============================================================================
# EINLESEN
# ============================================================================

def lade_transkript(pfad):
    """
    Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm
    Gibt Liste von Dicts: [{wort_raw, wort, start_s, ende_s}, ...]
    """
    muster = re.compile(
        r'(\S+)\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+(\d{2}:\d{2}:\d{2}\.\d+)'
    )

    def parse(text):
        eintraege = []
        for m in muster.finditer(text):
            wort_raw = m.group(1)
            wort_norm = normalisiere_wort(wort_raw)
            if not wort_norm:
                continue
            eintraege.append({
                'wort_raw': wort_raw,
                'wort': wort_norm,
                'start_s': zeit_zu_sekunden(m.group(2)),
                'ende_s': zeit_zu_sekunden(m.group(3)),
            })
        return eintraege

    try:
        with open(pfad, 'r', encoding='utf-8') as f:
            return parse(f.read())
    except UnicodeDecodeError:
        with open(pfad, 'r', encoding='latin-1') as f:
            return parse(f.read())


def lade_json(pfad):
    if not pfad.exists():
        return None
    try:
        with open(pfad, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================================
# N-GRAM MATCHING (längste Übereinstimmung gewinnt)
# ============================================================================

# Struktur: [(kategorie, n_gram_length, wortliste), ...]
def build_lookup():
    return [
        ('verzoegerung', 1, VERZOEGERUNGSLAUTE),
        ('hedge',        1, HECKENAUSDRUECKE_1),
        ('hedge',        2, HECKENAUSDRUECKE_2),
        ('hedge',        3, HECKENAUSDRUECKE_3),
        ('modal',        1, MODALPARTIKELN_1),
        ('modal',        2, MODALPARTIKELN_2),
        ('modal',        3, MODALPARTIKELN_3),
        ('modal',        4, MODALPARTIKELN_4),
        ('intensiv',     1, INTENSIVIERER_1),
        ('intensiv',     3, INTENSIVIERER_3),
    ]


KATEGORIE_LOOKUP = build_lookup()


def finde_matches(woerter):
    """
    Sliding-Window N-Gram-Matching. Prüft n=4,3,2,1 (längste zuerst).
    Wenn ein Match gefunden wird, überspringen wir alle beteiligten Wörter.
    Gibt Trefferliste zurück.
    """
    treffer = []
    i = 0
    n_woerter = len(woerter)

    while i < n_woerter:
        gefunden = False
        # Von 4-Gram (längster) zu 1-Gram (kürzester)
        for n in [4, 3, 2, 1]:
            if i + n > n_woerter:
                continue
            phrase = ' '.join(woerter[i + k]['wort'] for k in range(n))
            # Gegen alle Kategorien mit dieser Länge prüfen
            for kat, kat_n, liste in KATEGORIE_LOOKUP:
                if kat_n != n:
                    continue
                if phrase in liste:
                    treffer.append({
                        'kategorie': kat,
                        'phrase': phrase,
                        'start_s': woerter[i]['start_s'],
                        'ende_s': woerter[i + n - 1]['ende_s'],
                        'wortanzahl': n,
                        'wort_index': i,
                    })
                    i += n
                    gefunden = True
                    break
            if gefunden:
                break
        if not gefunden:
            i += 1

    return treffer


# ============================================================================
# BEJAHUNGS-KONTEXT-FILTER
# 'ja'/'doch' am Satzanfang eines sehr kurzen Satzes = Antwort, kein Füllwort
# ============================================================================

def erstelle_satz_lookup(inhalt_json):
    """Zeitraum-Liste der Sätze aus inhalt_analyse_output.json."""
    if not inhalt_json:
        return []
    lookup = []
    for s in inhalt_json.get('satzgrenzen', []):
        try:
            start = zeit_zu_sekunden(s['start'])
            ende = zeit_zu_sekunden(hole_ende(s))
        except (KeyError, ValueError):
            continue
        text = s.get('text', '')
        lookup.append({
            'satz_id': s.get('satz_id'),
            'start_s': start,
            'ende_s': ende,
            'text': text,
            'wortanzahl': len(text.split()),
        })
    return lookup


def finde_satz(zeit_s, satz_lookup):
    """Findet den Satz, in dem sich ein bestimmter Zeitpunkt befindet."""
    for satz in satz_lookup:
        if satz['start_s'] - 0.05 <= zeit_s <= satz['ende_s'] + 0.05:
            return satz
    return None


def filter_bejahungs_antworten(treffer, woerter, satz_lookup):
    """
    Entfernt 'ja'/'doch' Treffer, die als Antwort erkannt werden.
    Regel: erstes Wort im Satz UND Satz hat max. 3 Wörter → Antwort.
    """
    if not satz_lookup:
        return treffer, 0

    gefiltert = []
    ausgefiltert = 0
    for t in treffer:
        # Nur bei Bejahungs-Kandidaten prüfen
        if t['phrase'] not in BEJAHUNGS_KANDIDATEN:
            gefiltert.append(t)
            continue

        satz = finde_satz(t['start_s'], satz_lookup)
        if satz is None:
            gefiltert.append(t)
            continue

        # Ist es das erste Wort im Satz?
        idx = t['wort_index']
        if idx < 0 or idx >= len(woerter):
            gefiltert.append(t)
            continue
        wort_start = woerter[idx]['start_s']
        ist_erstes = abs(wort_start - satz['start_s']) < 0.2

        # Regel: erstes Wort + kurzer Satz → Antwort
        if ist_erstes and satz['wortanzahl'] <= 3:
            ausgefiltert += 1
            continue

        gefiltert.append(t)

    return gefiltert, ausgefiltert


# ============================================================================
# STATISTIK
# ============================================================================

def berechne_statistik(woerter, treffer):
    """Aggregiert die Trefferliste zu Kategorie-Statistiken."""
    if not woerter:
        return None
    dauer_sek = woerter[-1]['ende_s'] - woerter[0]['start_s']
    dauer_min = dauer_sek / 60 if dauer_sek > 0 else 0
    if dauer_min <= 0:
        return None

    def cat_stats(kategorie):
        cat_treffer = [t for t in treffer if t['kategorie'] == kategorie]
        # Wortanzahl = Summe der beteiligten Wörter (für Dichte)
        wort_summe = sum(t['wortanzahl'] for t in cat_treffer)
        return {
            'treffer': cat_treffer,
            'anzahl': len(cat_treffer),
            'wort_summe': wort_summe,
            'pro_min': len(cat_treffer) / dauer_min,
            'counter': Counter(t['phrase'] for t in cat_treffer),
        }

    stat = {
        'dauer_sek': dauer_sek,
        'dauer_min': dauer_min,
        'wort_gesamt': len(woerter),
        'verzoegerung': cat_stats('verzoegerung'),
        'hedge':        cat_stats('hedge'),
        'modal':        cat_stats('modal'),
        'intensiv':     cat_stats('intensiv'),
    }

    fuellsumme = sum(
        stat[k]['wort_summe'] for k in ('verzoegerung', 'hedge', 'modal', 'intensiv')
    )
    stat['fuell_woerter_gesamt'] = fuellsumme
    stat['dichte_prozent'] = (fuellsumme / len(woerter)) * 100 if woerter else 0.0

    return stat


# ============================================================================
# BEWERTUNG
# ============================================================================

def bewerte(pro_min, skala):
    """Sucht Skala-Eintrag: erster Eintrag mit pro_min <= max_pro_min."""
    for max_wert, level, icon, label, malus in skala:
        if pro_min <= max_wert:
            return {'level': level, 'icon': icon, 'label': label, 'malus': malus}
    return {'level': 'unbekannt', 'icon': '?', 'label': '?', 'malus': 0}


def bewerte_alle(stat):
    return {
        'verzoegerung': bewerte(stat['verzoegerung']['pro_min'], VERZOEGERUNG_SKALA),
        'hedge':        bewerte(stat['hedge']['pro_min'],        HECKEN_SKALA),
        'modal':        bewerte(stat['modal']['pro_min'],        MODALPARTIKEL_SKALA),
        'intensiv':     bewerte(stat['intensiv']['pro_min'],     INTENSIVIERER_SKALA),
    }


# ============================================================================
# SCORE (Malus-System)
# ============================================================================

def berechne_score(bewertungen, dichte_prozent):
    malus_gesamt = (
        bewertungen['verzoegerung']['malus']
        + bewertungen['hedge']['malus']
        + bewertungen['modal']['malus']
        + bewertungen['intensiv']['malus']
        + dichte_malus(dichte_prozent)
    )
    return max(0, 100 - malus_gesamt)


# ============================================================================
# FEEDBACK-TEXTE (5 Levels pro Kategorie, Studien-basiert)
# ============================================================================

FEEDBACK_VERZOEGERUNG = {
    'optimal': (
        "Keine hörbaren Verzögerungslaute. Das entspricht dem Ideal-Bereich "
        "nach Quantified Communications (Behavioral-Science-Studie: Ideal ≤1/min). "
        "Souveraener Auftritt."
    ),
    'sehr_gut': (
        "Nur vereinzelte Verzoegerungslaute (unter 1/min). Das ist der "
        "Quantified-Communications-Ideal-Bereich für professionelle Redner "
        "und deutlich unter dem Durchschnittssprecher (5/min)."
    ),
    'akzeptabel': (
        "Die Rate liegt im normalen Bereich. Der Durchschnittssprecher "
        "verwendet laut Studien 5 Verzögerungslaute pro Minute — Sie liegen "
        "darunter. Verbesserung möglich durch bewusste Pausen statt Fuelllauten."
    ),
    'auffaellig': (
        "Verzögerungslaute sind deutlich hörbar. Christenfeld (UCSD) zeigt: "
        "Publikum bewertet Sprecher in diesem Bereich als 'unprepared, nervous, "
        "lacking confidence'. Ziel: unter 3/min."
    ),
    'stoerend': (
        "Kritisch. Clark & Fox Tree (2002, Journal of Cognition) und Christenfeld "
        "zeigen: bei dieser Frequenz leidet die wahrgenommene Kompetenz "
        "erheblich. Sofortmassnahmen: mehr Vorbereitung, bewusste Atem-Pausen "
        "statt Fuelllauten."
    ),
}

FEEDBACK_HEDGE = {
    'zu_steif': (
        "Keine Heckenausdrücke. Die Sprache wirkt sehr direkt, evtl. "
        "abgelesen. Etwas Modalisierung ('sozusagen', 'im Prinzip') macht "
        "nahbarer und natürlicher (Wellner 2023)."
    ),
    'ideal': (
        "Ausgewogenes Verhältnis. Modalisierung ist vorhanden, ohne dass "
        "Aussagen verwaessert werden (Wellner 2023: Taxonomy of hedging devices)."
    ),
    'akzeptabel': (
        "Erhoeht, aber tolerabel. Prüfen Sie die häufigsten Ausdrücke — "
        "meist reicht es, ein einzelnes Gewohnheitswort zu streichen."
    ),
    'zu_viele': (
        "Zu viele Abschwaecher — Aussagen wirken unsicher. Konkret: statt "
        "'Das ist eigentlich wichtig' → 'Das ist wichtig'. Streichen Sie "
        "'irgendwie' komplett — es hat fast nie eine echte Funktion."
    ),
    'kritisch': (
        "Fast jede Aussage wird abgemildert. Christenfeld: Publikum nimmt "
        "starken Kompetenz-Verlust wahr. Formulieren Sie Kernaussagen bewusst "
        "hart, streichen Sie systematisch alle Hedges aus dem Skript."
    ),
}

FEEDBACK_MODAL = {
    'zu_steif': (
        "Kaum Modalpartikeln — die Rede wirkt geschrieben statt gesprochen. "
        "Etwas 'halt', 'ja', 'schon' macht die Sprache lebendiger "
        "(Weinrich-Grammatik, Thurmair 1989)."
    ),
    'ideal': (
        "Natürlicher, lebendiger Sprachfluss. Modalpartikeln sind ein "
        "anerkanntes Stilmittel gesprochener Sprache und tragen zur "
        "Authentizitaet bei."
    ),
    'erhoeht': (
        "Häufigkeit am oberen Ende des akzeptablen Bereichs. Achten Sie "
        "auf Wiederholungen — meist dominiert ein Gewohnheitswort."
    ),
    'auffaellig': (
        "Modalpartikeln dominieren die Rede — der rote Faden geht verloren. "
        "Streichen Sie in Kernaussagen alle 'halt', 'eben', 'auch'."
    ),
    'dominierend': (
        "Kritisch: Große Teile der Redezeit bestehen aus bedeutungsleeren "
        "Partikeln. Sprechen Sie in klaren Hauptsaetzen ohne Modalisierung."
    ),
}

FEEDBACK_INTENSIV = {
    'zurueckhaltend': (
        "Intensivierer werden zurückhaltend eingesetzt. Das verstaerkt ihre "
        "Wirkung, wenn sie kommen."
    ),
    'gut': (
        "Gute Balance zwischen Betonung und Zurückhaltung."
    ),
    'viele': (
        "Viele Verstaerker ('absolut', 'total'). Sie verlieren an Wirkung "
        "durch Inflation. Setzen Sie sie gezielter ein."
    ),
    'zu_viele': (
        "Zu viele Intensivierer. Wenn alles 'absolut wichtig' ist, ist nichts "
        "mehr wichtig."
    ),
    'inflationaer': (
        "Verstaerker werden inflationaer eingesetzt. Das signalisiert "
        "Übertreibung, nicht Emphase."
    ),
}

FEEDBACK_MAP = {
    'verzoegerung': FEEDBACK_VERZOEGERUNG,
    'hedge':        FEEDBACK_HEDGE,
    'modal':        FEEDBACK_MODAL,
    'intensiv':     FEEDBACK_INTENSIV,
}


# ============================================================================
# VERTEILUNGS- & WIEDERHOLUNGS-ANALYSE
# ============================================================================

def verteilungs_analyse(treffer_liste, dauer_min):
    if len(treffer_liste) < 4:
        return None
    dauer_sek = dauer_min * 60
    drittel = dauer_sek / 3
    anfang = sum(1 for t in treffer_liste if t['start_s'] < drittel)
    mitte = sum(1 for t in treffer_liste if drittel <= t['start_s'] < 2 * drittel)
    ende = sum(1 for t in treffer_liste if t['start_s'] >= 2 * drittel)
    total = len(treffer_liste)

    if anfang / total > 0.5:
        return ("Über die Haelfte tritt im ersten Drittel auf — deutet auf "
                "Anfangs-Nervositaet hin. Ueben Sie besonders den Einstieg "
                "(die ersten zwei Sätze auswendig).")
    if ende / total > 0.5:
        return ("Über die Haelfte tritt im letzten Drittel auf — "
                "Konzentrationsabfall gegen Ende. Planen Sie den Schluss "
                "bewusster und ueben Sie das Ende separat.")
    if mitte / total > 0.6:
        return ("Die meisten treten im Mittelteil auf — der ist evtl. "
                "weniger gut vorbereitet als Anfang und Schluss.")
    return None


def wiederholungs_analyse(counter, total):
    if total < 5 or not counter:
        return None
    wort, anzahl = counter.most_common(1)[0]
    anteil = anzahl / total
    if anteil > 0.5:
        return (f"'{wort}' macht {anteil*100:.0f} % aller Ausdrücke dieser "
                f"Kategorie aus ({anzahl}x) — typisches Tick-Muster. Diese "
                f"eine Angewohnheit gezielt reduzieren wirkt am schnellsten.")
    return None


# ============================================================================
# KONSISTENZ-CHECK (mit Sprechtempo)
# Referenz: Clark & Fox Tree (2002) — langsame Sprecher nutzen mehr Füllwörter
#
# v2-Fix: Cross-Module-Checks laufen jetzt zentral in gesamtscore.py
# (Abschnitt 12.5 der Planung). Wenn die Umgebungsvariable
# PAI_SKIP_LOCAL_CONSISTENCY = "1" gesetzt ist (main.py setzt das),
# wird dieser lokale Check übersprungen und die Rohdaten fliessen nur
# über JSON zum Aggregator. Standalone-Ausführung ohne main.py lässt
# den Check aus Kompatibilitaetsgruenden weiter laufen.
# ============================================================================

def konsistenz_check(stat, sprechtempo_json):
    import os
    if os.environ.get("PAI_SKIP_LOCAL_CONSISTENCY") == "1":
        return None
    if sprechtempo_json is None:
        return None
    tempo = sprechtempo_json.get('gesamttempo_silben_sek')
    if tempo is None:
        return None

    hinweise = []
    verzoeg_pm = stat['verzoegerung']['pro_min']
    hedge_pm = stat['hedge']['pro_min']
    modal_pm = stat['modal']['pro_min']

    # Muster 1: langsam + viele Verzögerungslaute (klassisches Clark-Muster)
    if tempo < 3.5 and verzoeg_pm > 3:
        hinweise.append(
            "Kombination aus langsamem Tempo und häufigen Verzögerungs-"
            "lauten entspricht dem klassischen Muster nach Clark & Fox Tree "
            "(2002): Verzögerungslaute signalisieren Sprachplanungs-"
            "Verzögerungen. Empfehlung: Vortrag mehrfach ueben, damit "
            "weniger 'gesucht' werden muss."
        )

    # Muster 2: langsam + viele Hedges (Nervositaet)
    if tempo < 3.5 and hedge_pm > 3:
        hinweise.append(
            "Langsames Tempo mit vielen Heckenausdrücken deutet auf "
            "Unsicherheit bei Kernaussagen hin (Nervositaets-Muster)."
        )

    # Muster 3: schnell + viele Modalpartikeln (gehetzt)
    if tempo > 5.5 and modal_pm > 12:
        hinweise.append(
            "Hohes Tempo mit vielen Modalpartikeln — die Rede wirkt "
            "möglicherweise gehetzt und unfokussiert. Etwas Verlangsamung "
            "würde die Wirkung stärken."
        )

    return hinweise if hinweise else None


# ============================================================================
# ZEITSTRAHL (ASCII)
# ============================================================================

def zeitstrahl(treffer_liste, dauer_min, breite=60):
    if not treffer_liste:
        return "  [keine gefunden]"
    zeile = ['-'] * breite
    dauer_sek = dauer_min * 60
    for t in treffer_liste:
        pos = int((t['start_s'] / dauer_sek) * (breite - 1))
        pos = max(0, min(breite - 1, pos))
        zeile[pos] = '|'
    return f"  0min {''.join(zeile)} {int(dauer_min)}min"


# ============================================================================
# TXT-REPORT
# ============================================================================

KATEGORIE_TITEL = {
    'verzoegerung': "1. VERZÖGERUNGSLAUTE (Hesitation Sounds)",
    'hedge':        "2. HECKENAUSDRÜCKE (Hedges)",
    'modal':        "3. MODALPARTIKELN + DISKURSMARKER",
    'intensiv':     "4. INTENSIVIERER (Gradpartikeln)",
}

KATEGORIE_QUELLE = {
    'verzoegerung': "Quellen: Belz (2021), Uni Trier 2023, Wikipedia-Linguistik",
    'hedge':        "Quellen: Wellner (2023), Uni Leipzig Lexikologie, Prince et al. (1982)",
    'modal':        "Quellen: Weinrich, Thurmair (1989), Meibauer (1994), Duden",
    'intensiv':     "Quellen: Duden Grammatik, Wellner (Gradpartikeln)",
}


def erstelle_txt_report(transkript_pfad, stat, bewertungen, score,
                        konsistenz, bejahung_ausgefiltert, pfad):
    z = []
    sep_1 = "=" * 72
    sep_2 = "-" * 72

    # Kopf
    z += [
        sep_1,
        "  FÜLLWÖRTER-ANALYSE (studien-basiert, v2)",
        sep_1,
        f"  Datei:      {transkript_pfad.name}",
        f"  Datum:      {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"  Dauer:      {int(stat['dauer_min'])}:{int(stat['dauer_sek'] % 60):02d} min",
        f"  Wörter:    {stat['wort_gesamt']}",
        sep_1,
        "",
        f"  GESAMTSCORE: {score}/100",
        "",
    ]

    # Score-Balken
    n_voll = score // 5
    balken = "█" * n_voll + "░" * (20 - n_voll)
    z += [f"  [{balken}]", ""]

    # Metrik B: Dichte
    d_icon, d_label = dichte_label(stat['dichte_prozent'])
    z += [
        sep_1,
        "  METRIK B: FÜLLWORT-DICHTE (Anteil aller Wörter)",
        sep_2,
        f"  Anteil:     {stat['dichte_prozent']:.1f} % aller Wörter",
        f"  Bewertung:  {d_icon} {d_label}",
        f"  Referenz:   NCT05444114 (N=182): Durchschnitt ~5.9 %, 1 SD ~9 %",
        "",
    ]

    # Kategorien
    for kat_key in ('verzoegerung', 'hedge', 'modal', 'intensiv'):
        kat_stat = stat[kat_key]
        b = bewertungen[kat_key]
        z += [
            sep_1,
            f"  {KATEGORIE_TITEL[kat_key]}",
            sep_2,
            f"  Bewertung:   {b['icon']} {b['label']}",
            f"  Total:       {kat_stat['anzahl']}x",
            f"  Pro Minute:  {kat_stat['pro_min']:.1f}/min",
            f"  Malus:       -{b['malus']}",
        ]
        if kat_stat['counter']:
            top = kat_stat['counter'].most_common(6)
            top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
            z.append(f"  Häufigste:  {top_str}")
        else:
            z.append("  Häufigste:  keine gefunden")

        z.append(f"  Zeitstrahl:")
        z.append(zeitstrahl(kat_stat['treffer'], stat['dauer_min']))
        z.append("")
        z.append("  Bewertung im Detail:")
        for zeile in FEEDBACK_MAP[kat_key][b['level']].split('\n'):
            z.append(f"    {zeile}")

        # Verteilungs- und Wiederholungs-Hinweise
        vh = verteilungs_analyse(kat_stat['treffer'], stat['dauer_min'])
        if vh:
            z.append(f"    → {vh}")
        wh = wiederholungs_analyse(kat_stat['counter'], kat_stat['anzahl'])
        if wh:
            z.append(f"    → {wh}")

        z.append(f"  {KATEGORIE_QUELLE[kat_key]}")
        z.append("")

    # Konsistenz-Check
    if konsistenz:
        z += [
            sep_1,
            "  KONSISTENZ-CHECK (mit Sprechtempo-Analyse)",
            sep_2,
        ]
        for h in konsistenz:
            for zeile in h.split('\n'):
                z.append(f"    {zeile}")
            z.append("")

    # Kontext-Filter-Info
    if bejahung_ausgefiltert > 0:
        z += [
            sep_1,
            "  KONTEXT-FILTER",
            sep_2,
            f"  {bejahung_ausgefiltert}x 'ja'/'doch' wurden als Bejahungs-Antworten "
            f"erkannt und NICHT als Füllwort gezählt.",
            "",
        ]

    # Referenzen
    z += [
        sep_1,
        "  STUDIEN-REFERENZEN",
        sep_1,
        "  Bewertungsskalen basieren auf:",
        "  - Quantified Communications: Ideal ≤1 Fuellwort/min, Ø 5/min",
        "  - Clark & Fox Tree (2002), Journal of Cognition:",
        "    'Using uh and um in spontaneous speaking'",
        "  - Christenfeld (UCSD): Publikums-Wahrnehmung bei Füllwörtern",
        "  - NCT05444114 (klinische Studie, N=182): Füllwort-Dichte in % aller Wörter",
        "  - Belz (2021): Die Phonetik von aeh und aehm (Springer Open Access)",
        "  - Uni Trier / Braun et al. (2023): Haesitationsverhalten",
        "  - Wellner (2023): Towards a taxonomy of hedging devices in Standard German",
        "  - Weinrich: Textgrammatik der deutschen Sprache",
        "  - Thurmair (1989): Modalpartikeln und ihre Kombinationen",
        sep_1,
    ]

    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, 'w', encoding='utf-8') as f:
        f.write('\n'.join(z))


# ============================================================================
# JSON-OUTPUT (Pipeline-kompatibel)
# ============================================================================

def erstelle_json_output(stat, bewertungen, score, konsistenz,
                         bejahung_ausgefiltert, treffer, pfad):
    def kat_json(key):
        kat_stat = stat[key]
        b = bewertungen[key]
        return {
            'anzahl': kat_stat['anzahl'],
            'pro_min': round(kat_stat['pro_min'], 2),
            'wort_summe': kat_stat['wort_summe'],
            'level': b['level'],
            'label': b['label'],
            'malus': b['malus'],
            'haeufigste': [
                {'phrase': w, 'anzahl': c}
                for w, c in kat_stat['counter'].most_common(10)
            ],
        }

    output = {
        'gesamtscore': score,
        'dauer_min': round(stat['dauer_min'], 2),
        'wort_gesamt': stat['wort_gesamt'],
        'fuell_woerter_gesamt': stat['fuell_woerter_gesamt'],
        'dichte_prozent': round(stat['dichte_prozent'], 2),
        'dichte_malus': dichte_malus(stat['dichte_prozent']),
        'kategorien': {
            'verzoegerung': kat_json('verzoegerung'),
            'hedge':        kat_json('hedge'),
            'modal':        kat_json('modal'),
            'intensiv':     kat_json('intensiv'),
        },
        'bejahungs_antworten_ausgefiltert': bejahung_ausgefiltert,
        'konsistenz_hinweise': konsistenz or [],
        'treffer': [
            {
                'phrase': t['phrase'],
                'kategorie': t['kategorie'],
                'start_s': round(t['start_s'], 3),
                'ende_s': round(t['ende_s'], 3),
                'wortanzahl': t['wortanzahl'],
            }
            for t in treffer
        ],
    }
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================================
# FILE-PICKER
# ============================================================================

def waehle_datei(titel, dateitypen, start_dir=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    kwargs = {'title': titel, 'filetypes': dateitypen}
    if start_dir:
        kwargs['initialdir'] = str(start_dir)
    pfad = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return pfad


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 72)
    print("  FÜLLWÖRTER-ANALYSE (v2, studien-basiert)")
    print("=" * 72)
    print()

    # ---- Schritt 1: Transkript wählen ----
    print("Schritt 1/8: Transkript waehlen...")
    if len(sys.argv) > 1:
        transkript_pfad = Path(sys.argv[1])
    else:
        start = PROJEKT_ROOT / "daten" / "transkripte"
        if not start.exists():
            start = PROJEKT_ROOT
        pfad_str = waehle_datei(
            "Transkript waehlen (.txt)",
            [("Textdateien", "*.txt"), ("Alle Dateien", "*.*")],
            start_dir=start,
        )
        if not pfad_str:
            print("  [x] Kein Transkript gewaehlt. Abbruch.")
            return
        transkript_pfad = Path(pfad_str)
    if not transkript_pfad.exists():
        print(f"  [x] Datei nicht gefunden: {transkript_pfad}")
        return
    print(f"  [OK] {transkript_pfad.name}")

    # ---- Schritt 2: Transkript einlesen ----
    print("Schritt 2/8: Transkript einlesen...")
    try:
        woerter = lade_transkript(transkript_pfad)
    except Exception as e:
        print(f"  [x] Fehler beim Einlesen: {e}")
        return
    if not woerter:
        print("  [x] Keine Wörter im Transkript gefunden.")
        return
    print(f"  [OK] {len(woerter)} Wörter eingelesen")

    # ---- Schritt 3: Inhalt-JSON laden (optional aber empfohlen) ----
    print("Schritt 3/8: Inhalt-Analyse JSON laden (fuer Kontext-Filter)...")
    inhalt_json = lade_json(JSON_INHALT_PFAD)
    if inhalt_json:
        satz_lookup = erstelle_satz_lookup(inhalt_json)
        print(f"  [OK] {len(satz_lookup)} Satzgrenzen geladen")
    else:
        satz_lookup = []
        print(f"  [!] Nicht gefunden ({JSON_INHALT_PFAD}). "
              f"Bejahungs-Kontext-Filter wird übersprungen.")

    # ---- Schritt 4: Sprechtempo-JSON laden (optional) ----
    print("Schritt 4/8: Sprechtempo JSON laden (fuer Konsistenz-Check)...")
    sprechtempo_json = lade_json(JSON_SPRECHTEMPO_PFAD)
    if sprechtempo_json:
        print(f"  [OK] Sprechtempo geladen "
              f"({sprechtempo_json.get('gesamttempo_silben_sek', '?')} Silben/Sek)")
    else:
        print(f"  [!] Nicht gefunden — Konsistenz-Check wird übersprungen.")

    # ---- Schritt 5: N-Gram-Matching ----
    print("Schritt 5/8: Fuellwoerter finden (N-Gram-Matching)...")
    treffer_roh = finde_matches(woerter)
    print(f"  [OK] {len(treffer_roh)} Treffer gefunden")

    # ---- Schritt 6: Bejahungs-Kontext-Filter ----
    print("Schritt 6/8: Kontext-Filter anwenden (ja/doch als Antwort)...")
    treffer, bejahung_ausgefiltert = filter_bejahungs_antworten(
        treffer_roh, woerter, satz_lookup
    )
    if bejahung_ausgefiltert > 0:
        print(f"  [OK] {bejahung_ausgefiltert} Bejahungs-Antworten ausgefiltert "
              f"→ {len(treffer)} echte Füllwörter")
    else:
        print(f"  [OK] Keine Bejahungs-Antworten erkannt")

    # ---- Schritt 7: Bewertung, Statistik, Score ----
    print("Schritt 7/8: Bewerten und Score berechnen...")
    stat = berechne_statistik(woerter, treffer)
    if stat is None:
        print("  [x] Statistik konnte nicht berechnet werden.")
        return
    bewertungen = bewerte_alle(stat)
    score = berechne_score(bewertungen, stat['dichte_prozent'])
    konsistenz = konsistenz_check(stat, sprechtempo_json)

    print(f"  [OK] Verzögerung:   {stat['verzoegerung']['anzahl']:>4}x "
          f"({stat['verzoegerung']['pro_min']:.1f}/min) → "
          f"{bewertungen['verzoegerung']['label']}")
    print(f"  [OK] Hedges:         {stat['hedge']['anzahl']:>4}x "
          f"({stat['hedge']['pro_min']:.1f}/min) → "
          f"{bewertungen['hedge']['label']}")
    print(f"  [OK] Modalpartikeln: {stat['modal']['anzahl']:>4}x "
          f"({stat['modal']['pro_min']:.1f}/min) → "
          f"{bewertungen['modal']['label']}")
    print(f"  [OK] Intensivierer:  {stat['intensiv']['anzahl']:>4}x "
          f"({stat['intensiv']['pro_min']:.1f}/min) → "
          f"{bewertungen['intensiv']['label']}")
    print(f"  [OK] Dichte: {stat['dichte_prozent']:.1f} % aller Wörter")
    print(f"  [OK] Gesamtscore: {score}/100")

    # ---- Schritt 8: Output ----
    print("Schritt 8/8: Report und JSON schreiben...")
    zeitstempel = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_pfad = REPORT_ORDNER / f"füllwörter_{transkript_pfad.stem}_{zeitstempel}.txt"
    erstelle_txt_report(
        transkript_pfad, stat, bewertungen, score,
        konsistenz, bejahung_ausgefiltert, report_pfad
    )
    print(f"  [OK] TXT-Report:       {report_pfad}")
    erstelle_json_output(
        stat, bewertungen, score, konsistenz,
        bejahung_ausgefiltert, treffer, JSON_OUTPUT_PFAD
    )
    print(f"  [OK] JSON-Intermediate: {JSON_OUTPUT_PFAD}")

    print()
    print("=" * 72)
    print(f"  FERTIG — Gesamtscore: {score}/100")
    print("=" * 72)


if __name__ == "__main__":
    main()
