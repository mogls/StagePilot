#!/usr/bin/env python3
"""
Füllwörter-Analyse für Präsentationstranskripte
Format: Wort    00:00:12.300  00:00:12.780
"""

import sys
import os
import re
from datetime import datetime
from collections import Counter

# ─── FÜLLWÖRTER LISTEN ───────────────────────────────────────────────────────

HESITATION = {
    'ähm', 'äh', 'hm', 'mh', 'öh', 'mmm', 'ehh', 'ähh'
}

WEICHMACHER = {
    'eigentlich', 'irgendwie', 'quasi', 'sozusagen', 'gewissermaßen',
    'ein bisschen', 'ein wenig', 'etwas', 'mehr oder weniger',
    'in gewisser weise', 'an sich', 'im prinzip', 'vielleicht',
    'allenfalls', 'ziemlich', 'ungefähr', 'in etwa', 'einigermaßen',
    'irgendwas', 'irgendwo', 'irgendwann', 'irgendein',
    'meines erachtens', 'man könnte sagen'
}

LEICHT = {
    'also', 'so', 'genau', 'ja', 'natürlich', 'wirklich', 'halt',
    'mal', 'einfach', 'ganz', 'nun', 'eben', 'doch', 'stimmt',
    'richtig', 'klar', 'selbstverständlich', 'exakt', 'tatsächlich',
    'in der tat', 'und zwar', 'das heißt', 'sprich', 'nämlich',
    'beziehungsweise', 'im endeffekt', 'am ende des tages',
    'letzten endes', 'letztlich', 'wie gesagt', 'kurz gesagt',
    'im grunde', 'im großen und ganzen', 'wie auch immer',
    'nichtsdestotrotz', 'sowieso', 'ohnedies', 'absolut', 'total',
    'definitiv', 'eindeutig', 'auf jeden fall', 'komplett',
    'ganz und gar', 'sehr', 'durchaus', 'unbedingt'
}

# ─── BEWERTUNGSSKALEN ────────────────────────────────────────────────────────

def bewerte_hesitation(pro_min):
    if pro_min == 0:      return ("✅", "Optimal")
    elif pro_min <= 1:    return ("✅", "Noch gut")
    elif pro_min <= 3:    return ("🟡", "Akzeptabel")
    elif pro_min <= 6:    return ("🟠", "Störend")
    else:                 return ("🔴", "Sehr störend")

def bewerte_weichmacher(pro_min):
    if pro_min == 0:      return ("⚠️",  "Wirkt sehr steif")
    elif pro_min <= 1:    return ("✅", "Optimal")
    elif pro_min <= 3:    return ("🟡", "Akzeptabel")
    elif pro_min <= 5:    return ("🟠", "Zu viele")
    else:                 return ("🔴", "Störend")

def bewerte_leicht(pro_min):
    if pro_min <= 2:      return ("⚠️",  "Wirkt sehr steif")
    elif pro_min <= 7:    return ("✅", "Optimal")
    elif pro_min <= 10:   return ("🟡", "Akzeptabel")
    elif pro_min <= 14:   return ("🟠", "Zu viele")
    else:                 return ("🔴", "Störend")

def berechne_score(h_pm, w_pm, l_pm):
    """Gesamtscore 0-100, Hesitation gewichtet stärker"""
    # Hesitation: 40% des Scores
    if h_pm == 0:       h_score = 100
    elif h_pm <= 1:     h_score = 90
    elif h_pm <= 3:     h_score = 70
    elif h_pm <= 6:     h_score = 40
    else:               h_score = 10

    # Weichmacher: 30% des Scores
    if w_pm == 0:       w_score = 70   # steif
    elif w_pm <= 1:     w_score = 100
    elif w_pm <= 3:     w_score = 80
    elif w_pm <= 5:     w_score = 50
    else:               w_score = 20

    # Leichte Füllwörter: 30% des Scores
    if l_pm <= 2:       l_score = 70   # steif
    elif l_pm <= 7:     l_score = 100
    elif l_pm <= 10:    l_score = 80
    elif l_pm <= 14:    l_score = 50
    else:               l_score = 20

    return round(h_score * 0.4 + w_score * 0.3 + l_score * 0.3)

# ─── TRANSKRIPT EINLESEN ─────────────────────────────────────────────────────

def parse_transkript(filepath):
    """Liest TXT Transkript und gibt Liste von (wort, start_sek, ende_sek) zurück"""
    entries = []

    def to_sec(ts):
        parts = ts.split(':')
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: Wort   00:00:12.300  00:00:12.780
    # Kopfzeile und Trennlinie werden übersprungen (kein gültiges Zeitstempel-Format)
    pattern = r'(\S+)\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+(\d{2}:\d{2}:\d{2}\.\d+)'

    for match in re.finditer(pattern, content):
        wort_raw = match.group(1)
        start_str = match.group(2)
        ende_str = match.group(3)

        # Wort bereinigen (Satzzeichen entfernen)
        wort = wort_raw.strip('.,!?;:').lower()

        entries.append((wort, to_sec(start_str), to_sec(ende_str)))

    return entries

# ─── ANALYSE ─────────────────────────────────────────────────────────────────

def analysiere(entries):
    """Hauptanalyse: findet Füllwörter und berechnet Statistiken"""
    if not entries:
        return None

    dauer_sek = entries[-1][2] - entries[0][1]
    dauer_min = dauer_sek / 60

    h_liste = []  # (wort, zeitpunkt)
    w_liste = []
    l_liste = []

    for wort, start, ende in entries:
        if wort in HESITATION:
            h_liste.append((wort, start))
        elif wort in WEICHMACHER:
            w_liste.append((wort, start))
        elif wort in LEICHT:
            l_liste.append((wort, start))

    # Pro Minute
    h_pm = len(h_liste) / dauer_min
    w_pm = len(w_liste) / dauer_min
    l_pm = len(l_liste) / dauer_min

    # Häufigste Wörter
    h_counter = Counter(w for w, _ in h_liste)
    w_counter = Counter(w for w, _ in w_liste)
    l_counter = Counter(w for w, _ in l_liste)

    return {
        'dauer_min': dauer_min,
        'dauer_sek': dauer_sek,
        'h_liste': h_liste,
        'w_liste': w_liste,
        'l_liste': l_liste,
        'h_pm': h_pm,
        'w_pm': w_pm,
        'l_pm': l_pm,
        'h_counter': h_counter,
        'w_counter': w_counter,
        'l_counter': l_counter,
        'score': berechne_score(h_pm, w_pm, l_pm),
    }

# ─── ZEITSTRAHL ──────────────────────────────────────────────────────────────

def zeitstrahl(liste, dauer_min, breite=60):
    """Erstellt ASCII Zeitstrahl der Füllwörter"""
    if not liste:
        return "  [keine gefunden]"

    zeile = ['-'] * breite
    dauer_sek = dauer_min * 60

    for _, zeitpunkt in liste:
        pos = int((zeitpunkt / dauer_sek) * (breite - 1))
        pos = max(0, min(breite - 1, pos))
        zeile[pos] = '|'

    # Zeitmarken
    marks = ""
    for i in range(0, int(dauer_min) + 1, 2):
        pos = int((i / dauer_min) * (breite - 1))
        marks += f"{i}min".ljust(max(1, int(2 / dauer_min * breite)))

    return f"  0min {''.join(zeile)} {int(dauer_min)}min"

# ─── FEEDBACK GENERIEREN ─────────────────────────────────────────────────────

def _verteilungs_analyse(liste, dauer_min):
    """Analysiert ob Füllwörter gehäuft am Anfang/Ende/Mitte vorkommen."""
    if len(liste) < 4:
        return None

    dauer_sek = dauer_min * 60
    drittel = dauer_sek / 3

    anfang = sum(1 for _, t in liste if t < drittel)
    mitte  = sum(1 for _, t in liste if drittel <= t < 2 * drittel)
    ende   = sum(1 for _, t in liste if t >= 2 * drittel)
    total  = len(liste)

    if anfang / total > 0.5:
        return ("Auffällig: Über die Hälfte kommt im ersten Drittel vor — "
                "das deutet auf Nervosität am Anfang hin. Übe besonders deinen Einstieg, "
                "z.B. die ersten zwei Sätze auswendig lernen.")
    if ende / total > 0.5:
        return ("Auffällig: Über die Hälfte kommt im letzten Drittel vor — "
                "möglicherweise lässt die Konzentration nach. Plane den Schluss "
                "bewusster und übe das Ende separat.")
    if mitte / total > 0.6:
        return ("Auffällig: Die meisten kommen im Mittelteil vor — "
                "eventuell ist dieser Teil weniger gut vorbereitet als Anfang und Schluss.")
    return None


def _wiederholungs_analyse(counter, total):
    """Prüft ob ein einzelnes Wort dominiert."""
    if total < 5 or not counter:
        return None

    wort, anzahl = counter.most_common(1)[0]
    anteil = anzahl / total

    if anteil > 0.5:
        return (f"Auffällig: '{wort}' macht {anteil*100:.0f}% aller Füllwörter dieser "
                f"Kategorie aus ({anzahl}x) — das fällt dem Publikum als Tick auf. "
                f"Konzentriere dich gezielt darauf, genau dieses Wort zu reduzieren.")
    return None


def generiere_feedback(daten):
    """Regelbasiertes Feedback — deckt ALLE Bewertungsstufen aller Kategorien ab."""
    feedback = []

    h_icon, h_text = bewerte_hesitation(daten['h_pm'])
    w_icon, w_text = bewerte_weichmacher(daten['w_pm'])
    l_icon, l_text = bewerte_leicht(daten['l_pm'])

    # ══════════════════ KATEGORIE 1: HESITATIONSLAUTE ══════════════════

    h_block = [f"HESITATIONSLAUTE — {h_icon} {h_text} ({daten['h_pm']:.1f}/min)"]

    if h_icon == "✅" and daten['h_pm'] == 0:
        h_block.append(
            "  Sehr gut: Keine hörbaren 'ähm' oder 'äh' — das wirkt souverän "
            "und professionell. Weiter so!"
        )
    elif h_icon == "✅":
        top = daten['h_counter'].most_common(2)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        h_block.append(
            f"  Gut: Nur vereinzelte Laute ({top_str}) — das ist völlig normal "
            f"und stört nicht. Kein Handlungsbedarf."
        )
    elif h_icon == "🟡":
        top = daten['h_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        h_block += [
            f"  Noch akzeptabel, aber verbesserbar. Häufigste: {top_str}",
            f"  Ziel: Unter 2 pro Minute kommen (aktuell {daten['h_pm']:.1f}).",
            f"  Tipp: Nimm dich beim Üben auf und höre die Aufnahme an — "
            f"Bewusstsein ist der erste Schritt. Ersetze den Laut durch eine "
            f"kurze stille Pause.",
        ]
    elif h_icon == "🟠":
        top = daten['h_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        h_block += [
            f"  Problem: {daten['h_pm']:.1f} Hesitationslaute pro Minute sind deutlich "
            f"hörbar und lenken das Publikum ab. Häufigste: {top_str}",
            f"  Ziel: Unter 2 pro Minute (aktuell mehr als das Doppelte).",
            f"  Tipps:",
            f"  1. Sprich langsamer — schnelles Sprechen erzeugt mehr Fülllaute.",
            f"  2. Mach bewusste Pausen an Satzenden statt sie zu füllen.",
            f"  3. Übe den Vortrag laut mindestens 3x — Unsicherheit im Stoff "
            f"ist die Hauptursache.",
        ]
    else:  # 🔴
        top = daten['h_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        h_block += [
            f"  Kritisch: {daten['h_pm']:.1f} Laute pro Minute — das Publikum nimmt "
            f"fast nur noch die 'ähm's wahr statt den Inhalt. Häufigste: {top_str}",
            f"  Die Glaubwürdigkeit leidet stark darunter.",
            f"  Tipps:",
            f"  1. Trainiere mit einem Partner der bei jedem Laut klatscht — "
            f"das schafft schnelles Bewusstsein.",
            f"  2. Bereite den Vortrag deutlich besser vor — Fülllaute entstehen "
            f"vor allem beim Suchen nach Worten.",
            f"  3. Atme bewusst: Vor jedem neuen Gedanken kurz einatmen statt 'ähm'.",
            f"  4. Reduziere schrittweise: erst auf 5/min, dann 3/min, dann 2/min.",
        ]

    hint = _verteilungs_analyse(daten['h_liste'], daten['dauer_min'])
    if hint:
        h_block.append(f"  {hint}")
    hint = _wiederholungs_analyse(daten['h_counter'], len(daten['h_liste']))
    if hint:
        h_block.append(f"  {hint}")

    feedback.append('\n'.join(h_block))

    # ══════════════════ KATEGORIE 2: WEICHMACHER ══════════════════

    w_block = [f"WEICHMACHER — {w_icon} {w_text} ({daten['w_pm']:.1f}/min)"]

    if w_icon == "⚠️":
        genutzt = {w for w, _ in daten['w_liste']}
        vorschlaege = sorted(WEICHMACHER - genutzt)[:8]
        w_block += [
            f"  Hinweis: Du verwendest praktisch keine Weichmacher — deine Sprache "
            f"wirkt dadurch sehr direkt, eventuell steif oder auswendig gelernt.",
            f"  Das ist kein grosses Problem, aber etwas Lockerheit macht dich nahbarer.",
            f"  Vorschlag: Baue gelegentlich (ca. 1x pro Minute) eines dieser Wörter ein:",
            f"  → {', '.join(vorschlaege)}",
            f"  Beispiel: statt 'Das ist falsch' → 'Das ist eigentlich nicht ganz richtig'",
        ]
    elif w_icon == "✅":
        top = daten['w_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top) if top else "—"
        w_block.append(
            f"  Sehr gut: Die Balance stimmt — deine Sprache ist bestimmt aber "
            f"nicht steif. Verwendet: {top_str}. Kein Handlungsbedarf."
        )
    elif w_icon == "🟡":
        top = daten['w_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        w_block += [
            f"  Noch akzeptabel, aber an der Grenze. Häufigste: {top_str}",
            f"  Ziel: Maximal 1 pro Minute (aktuell {daten['w_pm']:.1f}).",
            f"  Tipp: Achte besonders auf die häufigsten Wörter — meist reicht es, "
            f"ein einziges Gewohnheitswort zu streichen.",
        ]
    elif w_icon == "🟠":
        top = daten['w_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        w_block += [
            f"  Problem: {daten['w_pm']:.1f} Weichmacher pro Minute schwächen deine "
            f"Aussagen spürbar ab — du wirkst unsicher, auch wenn du es nicht bist.",
            f"  Häufigste: {top_str}",
            f"  Tipps:",
            f"  1. Formuliere Aussagen direkt: statt 'Das ist eigentlich wichtig' "
            f"→ 'Das ist wichtig'.",
            f"  2. Streiche 'irgendwie' komplett — es hat fast nie eine Funktion.",
            f"  3. Wenn du unsicher bist, sag es explizit ('Ich schätze...') statt "
            f"mit Weichmachern zu verschleiern.",
        ]
    else:  # 🔴
        top = daten['w_counter'].most_common(5)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        w_block += [
            f"  Kritisch: {daten['w_pm']:.1f} Weichmacher pro Minute — fast jeder "
            f"Satz wird relativiert. Das Publikum zweifelt an deiner Kompetenz, "
            f"selbst wenn der Inhalt stimmt.",
            f"  Häufigste: {top_str}",
            f"  Tipps:",
            f"  1. Schreibe deine Kernaussagen auf und streiche JEDEN Weichmacher "
            f"aus dem Skript.",
            f"  2. Übe die wichtigsten Sätze bewusst hart formuliert: 'Das ist so.' "
            f"statt 'Das ist eigentlich sozusagen irgendwie so.'",
            f"  3. Bitte eine Testperson, bei jedem Weichmacher die Hand zu heben.",
        ]

    hint = _verteilungs_analyse(daten['w_liste'], daten['dauer_min'])
    if hint:
        w_block.append(f"  {hint}")
    hint = _wiederholungs_analyse(daten['w_counter'], len(daten['w_liste']))
    if hint:
        w_block.append(f"  {hint}")

    feedback.append('\n'.join(w_block))

    # ══════════════════ KATEGORIE 3: LEICHTE FÜLLWÖRTER ══════════════════

    l_block = [f"LEICHTE FÜLLWÖRTER — {l_icon} {l_text} ({daten['l_pm']:.1f}/min)"]

    if l_icon == "⚠️":
        genutzt = {w for w, _ in daten['l_liste']}
        vorschlaege = sorted(LEICHT - genutzt)[:10]
        l_block += [
            f"  Hinweis: Nur {daten['l_pm']:.1f} Übergangswörter pro Minute — deine "
            f"Sprache wirkt dadurch abgehackt und roboterhaft, wie abgelesen.",
            f"  Übergangswörter helfen dem Publikum, deinem Gedankengang zu folgen.",
            f"  Vorschlag: Baue diese Wörter natürlich ein (3-7 pro Minute ist ideal):",
            f"  → {', '.join(vorschlaege)}",
            f"  Beispiel: 'Also, schauen wir uns das genauer an' oder "
            f"'Das heißt, wir müssen umdenken'",
        ]
    elif l_icon == "✅":
        top = daten['l_counter'].most_common(3)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top) if top else "—"
        l_block.append(
            f"  Sehr gut: Natürlicher, flüssiger Sprachfluss. "
            f"Häufigste: {top_str}. Kein Handlungsbedarf."
        )
    elif l_icon == "🟡":
        top = daten['l_counter'].most_common(5)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        l_block += [
            f"  Noch akzeptabel, aber es werden viele. Häufigste: {top_str}",
            f"  Ziel: 3-7 pro Minute (aktuell {daten['l_pm']:.1f}).",
            f"  Tipp: Meist ist es EIN Gewohnheitswort das zu oft kommt — "
            f"identifiziere deins und reduziere gezielt.",
        ]
    elif l_icon == "🟠":
        top = daten['l_counter'].most_common(5)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        l_block += [
            f"  Problem: {daten['l_pm']:.1f} Füllwörter pro Minute — der rote Faden "
            f"geht verloren, die Rede wirkt unstrukturiert.",
            f"  Häufigste: {top_str}",
            f"  Tipps:",
            f"  1. Ersetze Übergangs-Füllwörter durch echte Übergänge: statt "
            f"'Also, ja, genau...' → 'Kommen wir zum nächsten Punkt.'",
            f"  2. Kürze deine Sätze — lange verschachtelte Sätze erzeugen "
            f"mehr Füllwörter.",
            f"  3. Plane Übergänge zwischen Themen im Voraus.",
        ]
    else:  # 🔴
        top = daten['l_counter'].most_common(5)
        top_str = ", ".join(f"'{w}' ({c}x)" for w, c in top)
        l_block += [
            f"  Kritisch: {daten['l_pm']:.1f} Füllwörter pro Minute — ein grosser Teil "
            f"deiner Redezeit besteht aus Wörtern ohne Inhalt. Das Publikum "
            f"verliert die Aufmerksamkeit.",
            f"  Häufigste: {top_str}",
            f"  Tipps:",
            f"  1. Skripte deinen Vortrag und markiere alle Füllwörter — "
            f"dann übe die Sätze ohne sie.",
            f"  2. Sprich in kurzen, klaren Hauptsätzen.",
            f"  3. Nutze bewusste Pausen zwischen Gedanken statt Übergangswörtern.",
            f"  4. Reduziere schrittweise: Nimm dir pro Übungsdurchlauf EIN Wort vor.",
        ]

    hint = _verteilungs_analyse(daten['l_liste'], daten['dauer_min'])
    if hint:
        l_block.append(f"  {hint}")
    hint = _wiederholungs_analyse(daten['l_counter'], len(daten['l_liste']))
    if hint:
        l_block.append(f"  {hint}")

    feedback.append('\n'.join(l_block))

    # ══════════════════ GESAMTFAZIT ══════════════════

    icons = [h_icon, w_icon, l_icon]
    optimal_count = icons.count("✅")

    if optimal_count == 3:
        fazit = ("GESAMTFAZIT\n  Alle drei Kategorien im optimalen Bereich — "
                 "hervorragende Präsentation in Bezug auf Füllwörter! "
                 "Behalte dieses Niveau bei.")
    elif "🔴" in icons:
        schlecht = []
        if h_icon == "🔴": schlecht.append("Hesitationslaute")
        if w_icon == "🔴": schlecht.append("Weichmacher")
        if l_icon == "🔴": schlecht.append("leichte Füllwörter")
        fazit = (f"GESAMTFAZIT\n  Dringender Handlungsbedarf bei: {', '.join(schlecht)}. "
                 f"Konzentriere dich zuerst auf diese Kategorie(n) — "
                 f"dort ist der grösste Hebel für Verbesserung.")
    elif "🟠" in icons:
        maessig = []
        if h_icon == "🟠": maessig.append("Hesitationslaute")
        if w_icon == "🟠": maessig.append("Weichmacher")
        if l_icon == "🟠": maessig.append("leichte Füllwörter")
        fazit = (f"GESAMTFAZIT\n  Verbesserungspotenzial bei: {', '.join(maessig)}. "
                 f"Mit gezieltem Üben (Aufnahmen anhören, bewusste Pausen) "
                 f"ist der optimale Bereich gut erreichbar.")
    elif "⚠️" in icons:
        fazit = ("GESAMTFAZIT\n  Keine störenden Füllwörter, aber die Sprache "
                 "wirkt teils zu steif. Etwas mehr Natürlichkeit macht den "
                 "Vortrag lebendiger und nahbarer.")
    else:
        fazit = ("GESAMTFAZIT\n  Solide Leistung mit kleinen Optimierungsmöglichkeiten. "
                 "Die Details stehen oben bei den einzelnen Kategorien.")

    feedback.append(fazit)

    return feedback

# ─── REPORT ERSTELLEN ────────────────────────────────────────────────────────

def erstelle_report(filepath, daten, output_dir):
    """Erstellt TXT Report"""
    os.makedirs(output_dir, exist_ok=True)

    dateiname = os.path.splitext(os.path.basename(filepath))[0]
    report_path = os.path.join(output_dir, f"{dateiname}_report.txt")

    h_icon, h_text = bewerte_hesitation(daten['h_pm'])
    w_icon, w_text = bewerte_weichmacher(daten['w_pm'])
    l_icon, l_text = bewerte_leicht(daten['l_pm'])
    feedback = generiere_feedback(daten)

    dauer_str = f"{int(daten['dauer_min'])}:{int(daten['dauer_sek'] % 60):02d} min"

    lines = []
    sep = "=" * 65
    sep2 = "-" * 65

    lines += [
        sep,
        "  FÜLLWÖRTER-ANALYSE REPORT",
        sep,
        f"  Datei     : {os.path.basename(filepath)}",
        f"  Datum     : {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"  Dauer     : {dauer_str}",
        sep,
        "",
        "  GESAMTSCORE",
        sep2,
        f"  {daten['score']}/100",
        "",
    ]

    # Score Balken
    balken = "█" * (daten['score'] // 5) + "░" * (20 - daten['score'] // 5)
    lines.append(f"  [{balken}]")
    lines.append("")

    # Kategorie Übersichten
    lines += [
        sep,
        "  KATEGORIE 1: HESITATIONSLAUTE",
        sep2,
        f"  Bewertung  : {h_icon} {h_text}",
        f"  Total      : {len(daten['h_liste'])}x",
        f"  Pro Minute : {daten['h_pm']:.1f}/min",
    ]

    if daten['h_counter']:
        top = daten['h_counter'].most_common(5)
        lines.append(f"  Häufigste  : {', '.join(f'{w} ({c}x)' for w,c in top)}")
    else:
        lines.append("  Häufigste  : keine gefunden")

    lines.append(f"  Zeitstrahl :")
    lines.append(zeitstrahl(daten['h_liste'], daten['dauer_min']))
    lines.append("")

    lines += [
        sep,
        "  KATEGORIE 2: WEICHMACHER",
        sep2,
        f"  Bewertung  : {w_icon} {w_text}",
        f"  Total      : {len(daten['w_liste'])}x",
        f"  Pro Minute : {daten['w_pm']:.1f}/min",
    ]

    if daten['w_counter']:
        top = daten['w_counter'].most_common(5)
        lines.append(f"  Häufigste  : {', '.join(f'{w} ({c}x)' for w,c in top)}")
    else:
        lines.append("  Häufigste  : keine gefunden")

    lines.append(f"  Zeitstrahl :")
    lines.append(zeitstrahl(daten['w_liste'], daten['dauer_min']))
    lines.append("")

    lines += [
        sep,
        "  KATEGORIE 3: LEICHTE FÜLLWÖRTER",
        sep2,
        f"  Bewertung  : {l_icon} {l_text}",
        f"  Total      : {len(daten['l_liste'])}x",
        f"  Pro Minute : {daten['l_pm']:.1f}/min",
    ]

    if daten['l_counter']:
        top = daten['l_counter'].most_common(8)
        lines.append(f"  Häufigste  : {', '.join(f'{w} ({c}x)' for w,c in top)}")
    else:
        lines.append("  Häufigste  : keine gefunden")

    lines.append(f"  Zeitstrahl :")
    lines.append(zeitstrahl(daten['l_liste'], daten['dauer_min']))
    lines.append("")

    # Feedback
    lines += [
        sep,
        "  FEEDBACK & VERBESSERUNGSVORSCHLÄGE",
        sep,
    ]

    for fb in feedback:
        lines.append("")
        for zeile in fb.split('\n'):
            lines.append(f"  {zeile}")

    lines += [
        "",
        sep,
        "  Ende des Reports",
        sep,
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return report_path

# ─── MAIN ────────────────────────────────────────────────────────────────────

def waehle_datei():
    """Öffnet einen Datei-Auswahl-Dialog, startet direkt im Transkripte-Ordner"""
    import tkinter as tk
    from tkinter import filedialog

    # Transkripte-Ordner liegt eine Ebene über dem Skript-Ordner (Füllwörter/)
    skript_ordner = os.path.dirname(os.path.abspath(__file__))
    projekt_ordner = os.path.dirname(skript_ordner)
    transkripte_ordner = os.path.join(projekt_ordner, "Transkripte")

    # Fallback: falls Ordner nicht existiert, normales Startverzeichnis
    start_dir = transkripte_ordner if os.path.isdir(transkripte_ordner) else projekt_ordner

    root = tk.Tk()
    root.withdraw()  # Hauptfenster verstecken
    root.attributes('-topmost', True)  # Dialog im Vordergrund

    filepath = filedialog.askopenfilename(
        title="Transkript auswählen",
        initialdir=start_dir,
        filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
    )

    root.destroy()
    return filepath


def main():
    print("=" * 50)
    print("  FÜLLWÖRTER-ANALYSE")
    print("=" * 50)

    # ── Schritt 1: Datei wählen ──
    print("\n[Schritt 1/6] Datei auswählen...")

    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
        print(f"  → Datei per Argument: {filepath}")
    else:
        print("  → Auswahl-Fenster öffnet sich...")
        filepath = waehle_datei()

        if not filepath:
            print("  ✗ Keine Datei ausgewählt. Abbruch.")
            input("\nDrücke Enter zum Beenden...")
            sys.exit(0)
        print(f"  → Gewählt: {filepath}")

    if not os.path.exists(filepath):
        print(f"  ✗ Fehler: Datei '{filepath}' nicht gefunden.")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    print("  ✓ Datei gefunden")

    # ── Schritt 2: Transkript einlesen ──
    print("\n[Schritt 2/6] Transkript einlesen...")

    try:
        entries = parse_transkript(filepath)
    except Exception as e:
        print(f"  ✗ Fehler beim Einlesen: {e}")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    if not entries:
        print("  ✗ Fehler: Keine Wörter im Transkript gefunden.")
        print("    Erwartetes Format: Wort    00:00:12.300 00:00:12.780")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    print(f"  ✓ {len(entries)} Wörter eingelesen")

    # ── Schritt 3: Analysieren ──
    print("\n[Schritt 3/6] Füllwörter analysieren...")

    daten = analysiere(entries)

    print(f"  ✓ Dauer: {daten['dauer_min']:.1f} Minuten")
    print(f"  ✓ Hesitationslaute:   {len(daten['h_liste'])}x ({daten['h_pm']:.1f}/min)")
    print(f"  ✓ Weichmacher:        {len(daten['w_liste'])}x ({daten['w_pm']:.1f}/min)")
    print(f"  ✓ Leichte Füllwörter: {len(daten['l_liste'])}x ({daten['l_pm']:.1f}/min)")

    # ── Schritt 4: Score berechnen ──
    print("\n[Schritt 4/6] Score berechnen...")
    print(f"  ✓ Gesamtscore: {daten['score']}/100")

    # ── Schritt 5: Feedback generieren ──
    print("\n[Schritt 5/6] Feedback generieren...")
    print("  ✓ Feedback erstellt")

    # ── Schritt 6: Report speichern ──
    print("\n[Schritt 6/6] Report speichern...")

    # Output Ordner: gleicher Ordner wie das Skript (Füllwörter/) + Unterordner
    skript_ordner = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(skript_ordner, "Füllwörter_Report")

    print(f"  → Zielordner: {output_dir}")

    try:
        report_path = erstelle_report(filepath, daten, output_dir)
        print(f"  ✓ Report gespeichert: {report_path}")
    except Exception as e:
        print(f"  ✗ Fehler beim Speichern: {e}")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  FERTIG!")
    print("=" * 50)

    input("\nDrücke Enter zum Beenden...")

if __name__ == "__main__":
    main()
