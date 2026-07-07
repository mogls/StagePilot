# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════
 inhalt_analyse.py — Transkript-Inhaltsanalyse
 Projekt: praesentation_ai
════════════════════════════════════════════════════════════════════

Analysiert ein Transkript (Standardformat: Wort  00:00:12.300  00:00:12.780)
und erzeugt alle Inhalts-Informationen, die die nachgelagerten
Audio-Analyse-Scripts benötigen.

INPUT:
    Transkript-Datei (.txt) — Auswahl über tkinter-Dateidialog

OUTPUT:
    1. zwischen_output/inhalt_analyse_output.json   (für Audio-Scripts)
    2. reports/inhalt_analyse_bericht_<name>.txt    (lesbarer Bericht)

MODELLE (werden einmalig geladen):
    - spaCy de_core_news_sm                  → Satzgrenzen, Satzstruktur,
                                               rhetorische Momente
    - Sahajtomar/German_Zeroshot             → Kernbotschaften, Struktur
    - oliverguhr/german-sentiment-bert       → Emotionaler Ton

Beim allerersten Start werden die zwei Hugging-Face-Modelle automatisch
heruntergeladen (~2 GB, Internet nötig). Danach laufen sie lokal.
"""

import json
import os
import re
import sys
from datetime import datetime

# ════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ════════════════════════════════════════════════════════════════════

# Modell-Namen
SPACY_MODELL = "de_core_news_sm"
ZEROSHOT_MODELL = "Sahajtomar/German_Zeroshot"
SENTIMENT_MODELL = "oliverguhr/german-sentiment-bert"

# Kernbotschaften: Zero-Shot-Labels + Schwellenwert
KERNBOTSCHAFT_LABELS = [
    "Kernbotschaft", "wichtige Aussage", "Nebeninformation",
    "Übergang", "Beispiel",
]
KERNBOTSCHAFT_SCHWELLE = 0.6

# Struktur: Zero-Shot-Labels + Zeitgrenzen
STRUKTUR_LABELS = [
    "Einleitung", "Hauptteil", "Übergang", "Schluss", "Zusammenfassung",
]
EINLEITUNG_ANTEIL = 0.15   # Einleitung nur in ersten 15% erlaubt
SCHLUSS_ANTEIL = 0.15      # Schluss nur in letzten 15% erlaubt

# Hypothesen-Vorlage für das deutsche Zero-Shot-Modell
HYPOTHESE_VORLAGE = "Dieser Satz ist {}."

# Satzstruktur: Längen-Grenzen (Wortanzahl)
KURZ_MAX = 8       # <= 8 Wörter  → kurz
LANG_MAX = 20      # 9–20 Wörter  → lang, > 20 → komplex

# Rhetorische Momente: Schlüsselwörter für Höhepunkte
HOEHEPUNKT_WOERTER = {"wichtig", "entscheidend", "niemals", "immer", "jeder"}

# Publikumsbezug: Ansprache-Wörter
# Kleingeschriebene Formen — immer Ansprache (case-insensitive geprüft)
ANSPRACHE_IMMER = {"wir", "uns", "euch", "du", "dir", "dich"}
# Grossgeschriebene Formen — nur Ansprache wenn wirklich grossgeschrieben
# UND nicht am Satzanfang (sonst nicht unterscheidbar von "sie" = dritte Person)
ANSPRACHE_FORMELL = {"Sie", "Ihnen", "Ihr"}

# Emotionaler Ton: Sentiment-Label → Präsentations-Ton
TON_MAPPING = {
    "positive": "inspirierend",
    "neutral": "sachlich",
    "negative": "ernst",
}

# Sentiment-Modell: max. Wörter pro Abschnitt (Sicherheitsabstand zu 512 Tokens)
SENTIMENT_MAX_WOERTER = 250


# ════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN — Zeitstempel
# ════════════════════════════════════════════════════════════════════

def zeit_zu_sekunden(zeitstempel):
    """Wandelt 'HH:MM:SS.mmm' in Sekunden (float) um.

    Beispiel: '00:01:12.300' → 72.3
    """
    match = re.match(r"^(\d+):(\d+):(\d+)\.(\d+)$", zeitstempel.strip())
    if not match:
        raise ValueError(f"Ungültiger Zeitstempel: '{zeitstempel}'")
    stunden, minuten, sekunden, millis = match.groups()
    # Millisekunden auf 3 Stellen normalisieren ('3' → '300')
    millis = millis.ljust(3, "0")[:3]
    return (
        int(stunden) * 3600
        + int(minuten) * 60
        + int(sekunden)
        + int(millis) / 1000.0
    )


def sekunden_zu_zeit(sekunden):
    """Wandelt Sekunden (float) in 'HH:MM:SS.mmm' um.

    Beispiel: 72.3 → '00:01:12.300'
    """
    millis = int(round(sekunden * 1000))
    stunden, rest = divmod(millis, 3600 * 1000)
    minuten, rest = divmod(rest, 60 * 1000)
    sek, ms = divmod(rest, 1000)
    return f"{stunden:02d}:{minuten:02d}:{sek:02d}.{ms:03d}"


# ════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN — Pfade
# ════════════════════════════════════════════════════════════════════

def projekt_root():
    """Ermittelt den Projekt-Root (Ordner über analyse/).

    Liegt das Script in praesentation_ai/analyse/, ist der Root
    praesentation_ai/. Sonst wird der Ordner des Scripts verwendet.
    """
    script_ordner = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_ordner).lower() == "analyse":
        return os.path.dirname(script_ordner)
    return script_ordner


# ════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Transkript einlesen
# ════════════════════════════════════════════════════════════════════

def lade_transkript(pfad=None):
    """Liest die Transkript-Datei ein.

    Öffnet einen tkinter-Dateidialog (falls kein Pfad übergeben wurde)
    und parst jede Zeile im Standardformat:

        Wort  00:00:12.300  00:00:12.780

    Rückgabe: (woerter, pfad)
        woerter = Liste von Dicts:
        {
          "wort":    "Bildung",
          "start":   "00:00:12.300",   (Original-String)
          "end":     "00:00:12.780",
          "start_s": 12.3,             (Sekunden als float)
          "end_s":   12.78
        }
    """
    if pfad is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        pfad = filedialog.askopenfilename(
            title="Transkript-Datei auswählen",
            filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")],
        )
        root.destroy()
        if not pfad:
            print("  ✗ Keine Datei ausgewählt — Abbruch.")
            sys.exit(1)

    if not os.path.isfile(pfad):
        print(f"  ✗ Datei nicht gefunden: {pfad}")
        sys.exit(1)

    woerter = []
    fehlerhafte_zeilen = 0

    with open(pfad, "r", encoding="utf-8") as f:
        for zeilen_nr, zeile in enumerate(f, start=1):
            zeile = zeile.strip()
            if not zeile:
                continue  # Leere Zeilen überspringen

            teile = zeile.split()
            # Erwartet: mindestens Wort + 2 Zeitstempel.
            # Falls das "Wort" aus mehreren Teilen besteht (z.B. "das heisst"),
            # sind die letzten 2 Teile die Zeitstempel.
            if len(teile) < 3:
                fehlerhafte_zeilen += 1
                if fehlerhafte_zeilen <= 5:
                    print(f"  ⚠ Zeile {zeilen_nr} übersprungen "
                          f"(zu wenige Spalten): '{zeile}'")
                continue

            wort = " ".join(teile[:-2])
            start_str, end_str = teile[-2], teile[-1]

            try:
                start_s = zeit_zu_sekunden(start_str)
                end_s = zeit_zu_sekunden(end_str)
            except ValueError:
                fehlerhafte_zeilen += 1
                if fehlerhafte_zeilen <= 5:
                    print(f"  ⚠ Zeile {zeilen_nr} übersprungen "
                          f"(ungültiger Zeitstempel): '{zeile}'")
                continue

            woerter.append({
                "wort": wort,
                "start": start_str,
                "end": end_str,
                "start_s": start_s,
                "end_s": end_s,
            })

    if fehlerhafte_zeilen > 5:
        print(f"  ⚠ Insgesamt {fehlerhafte_zeilen} fehlerhafte Zeilen "
              f"übersprungen.")

    if not woerter:
        print("  ✗ Transkript enthält keine gültigen Wörter — Abbruch.")
        sys.exit(1)

    return woerter, pfad


# ════════════════════════════════════════════════════════════════════
# SCHRITT 2 — Modelle laden
# ════════════════════════════════════════════════════════════════════

def lade_modelle():
    """Lädt alle drei Modelle einmalig.

    Rückgabe: (nlp, zero_shot, sentiment)
    """
    import spacy
    from transformers import pipeline

    print("  → spaCy laden...")
    try:
        nlp = spacy.load(SPACY_MODELL)
    except OSError:
        print(f"  ✗ spaCy-Modell '{SPACY_MODELL}' nicht gefunden.")
        print(f"    Installation: python -m spacy download {SPACY_MODELL}")
        sys.exit(1)
    print("    ✓ spaCy bereit")

    print("  → Zero-Shot-Modell laden (beim ersten Mal: Download ~1.4 GB)...")
    zero_shot = pipeline(
        "zero-shot-classification",
        model=ZEROSHOT_MODELL,
        device=-1,  # CPU
    )
    print("    ✓ Zero-Shot bereit")

    print("  → Sentiment-Modell laden (beim ersten Mal: Download ~0.4 GB)...")
    sentiment = pipeline(
        "text-classification",
        model=SENTIMENT_MODELL,
        device=-1,  # CPU
        top_k=None,  # Alle Label-Scores zurückgeben (nicht nur Top-1)
    )
    print("    ✓ Sentiment bereit")

    return nlp, zero_shot, sentiment


# ════════════════════════════════════════════════════════════════════
# SCHRITT 3 — Satzgrenzen (Basis für alles andere)
# ════════════════════════════════════════════════════════════════════

def berechne_satzgrenzen(woerter, nlp):
    """Erkennt Satzgrenzen mit spaCy und ordnet Zeitstempel zu.

    Ablauf:
      1. Alle Wörter zu einem Volltext zusammensetzen,
         dabei die Zeichen-Position jedes Worts merken.
      2. spaCy erkennt die Satzgrenzen im Volltext.
      3. Pro Satz: erstes/letztes Wort → Start-/End-Zeitstempel.

    Rückgabe: Liste von Dicts:
        { "satz_id": 1, "text": "...", "start": "...", "end": "..." }
    """
    # 1. Volltext bauen + Zeichenposition pro Wort merken
    volltext_teile = []
    wort_positionen = []  # (zeichen_start, zeichen_end, wort_index)
    position = 0

    for i, w in enumerate(woerter):
        wort = w["wort"]
        volltext_teile.append(wort)
        wort_positionen.append((position, position + len(wort), i))
        position += len(wort) + 1  # +1 für das Leerzeichen

    volltext = " ".join(volltext_teile)

    # 2. spaCy Satzgrenzen erkennen
    doc = nlp(volltext)

    # 3. Pro Satz: welche Wörter gehören dazu?
    satzgrenzen = []
    satz_id = 0

    for sent in doc.sents:
        # Alle Wort-Indizes deren Startposition im Satz-Bereich liegt
        indizes = [
            wi for (zs, ze, wi) in wort_positionen
            if sent.start_char <= zs < sent.end_char
        ]
        if not indizes:
            continue

        satz_id += 1
        erstes = woerter[indizes[0]]
        letztes = woerter[indizes[-1]]

        satzgrenzen.append({
            "satz_id": satz_id,
            "text": sent.text.strip(),
            "start": erstes["start"],
            "end": letztes["end"],
        })

    return satzgrenzen


# ════════════════════════════════════════════════════════════════════
# SCHRITT 4 — Satzstruktur
# ════════════════════════════════════════════════════════════════════

def berechne_satzstruktur(satzgrenzen, nlp):
    """Bestimmt pro Satz: Typ (Frage/Aussage), Länge, Wortanzahl.

    Längen-Klassifikation:
        <= 8 Wörter          → kurz
        9–20 Wörter          → lang
        > 20 Wörter          → komplex
        Nebensatz vorhanden  → komplex (via spaCy Dependency Parsing)

    Rückgabe: Liste von Dicts:
        { "satz_id": 4, "text": "...", "typ": "Frage",
          "laenge": "kurz", "wortanzahl": 6, "start": "...", "end": "..." }
    """
    satzstruktur = []

    for satz in satzgrenzen:
        text = satz["text"]
        doc = nlp(text)

        # Wortanzahl (nur echte Wörter, keine Satzzeichen)
        wortanzahl = sum(1 for token in doc if not token.is_punct)

        # Typ: Frage oder Aussage
        typ = "Frage" if text.rstrip().endswith("?") else "Aussage"

        # Länge nach Wortanzahl
        if wortanzahl <= KURZ_MAX:
            laenge = "kurz"
        elif wortanzahl <= LANG_MAX:
            laenge = "lang"
        else:
            laenge = "komplex"

        # Nebensatz-Erkennung: unterordnende Konjunktion (weil, dass, ...)
        # oder Relativpronomen → Satz ist komplex
        if laenge != "komplex":
            hat_nebensatz = any(
                token.pos_ == "SCONJ" or token.tag_ in ("PRELS", "PRELAT")
                for token in doc
            )
            if hat_nebensatz:
                laenge = "komplex"

        satzstruktur.append({
            "satz_id": satz["satz_id"],
            "text": text,
            "typ": typ,
            "laenge": laenge,
            "wortanzahl": wortanzahl,
            "start": satz["start"],
            "end": satz["end"],
        })

    return satzstruktur


# ════════════════════════════════════════════════════════════════════
# SCHRITT 5 — Rhetorische Momente
# ════════════════════════════════════════════════════════════════════

def berechne_rhetorische_momente(satzgrenzen, nlp):
    """Erkennt Fragen und Höhepunkte.

    Frage:      Satz endet auf '?'
    Höhepunkt:  Satz endet auf '!'
                ODER enthält Superlativ (spaCy Morphologie)
                ODER enthält Schlüsselwort (wichtig, entscheidend, ...)

    Ein Satz kann nur Frage ODER Höhepunkt sein — Frage hat Priorität.

    Rückgabe: Liste von Dicts:
        { "typ": "Frage", "satz_id": 7, "text": "...",
          "start": "...", "end": "..." }
    """
    momente = []

    for satz in satzgrenzen:
        text = satz["text"].rstrip()
        typ = None

        # 1. Frage hat Priorität
        if text.endswith("?"):
            typ = "Frage"
        else:
            # 2. Höhepunkt-Signale prüfen
            ist_hoehepunkt = False

            if text.endswith("!"):
                ist_hoehepunkt = True

            if not ist_hoehepunkt:
                doc = nlp(text)
                for token in doc:
                    # Superlativ (z.B. "am wichtigsten", "grösste")
                    if "Sup" in token.morph.get("Degree"):
                        ist_hoehepunkt = True
                        break
                    # Schlüsselwörter
                    if token.text.lower() in HOEHEPUNKT_WOERTER:
                        ist_hoehepunkt = True
                        break

            if ist_hoehepunkt:
                typ = "Höhepunkt"

        if typ:
            momente.append({
                "typ": typ,
                "satz_id": satz["satz_id"],
                "text": satz["text"],
                "start": satz["start"],
                "end": satz["end"],
            })

    return momente


# ════════════════════════════════════════════════════════════════════
# SCHRITT 6 — Kernbotschaften (Zero-Shot AI)
# ════════════════════════════════════════════════════════════════════

def berechne_kernbotschaften(satzgrenzen, zero_shot):
    """Erkennt Kernbotschaften mit dem Zero-Shot-Modell.

    Pro Satz:
      - Zero-Shot-Klassifikation mit 5 Labels
      - Gesamt-Score = Durchschnitt aus Score('Kernbotschaft')
                       und Score('wichtige Aussage')
      - Gesamt-Score >= 0.6 → Kernbotschaft

    Rückgabe: Liste von Dicts (absteigend nach Score sortiert):
        { "satz_id": 3, "text": "...", "start": "...", "end": "...",
          "score": 0.82 }
    """
    kernbotschaften = []
    gesamt = len(satzgrenzen)

    for i, satz in enumerate(satzgrenzen, start=1):
        # Fortschrittsanzeige (gleiche Zeile überschreiben)
        print(f"\r    Analysiere Satz {i}/{gesamt}...", end="", flush=True)

        resultat = zero_shot(
            satz["text"],
            candidate_labels=KERNBOTSCHAFT_LABELS,
            multi_label=False,
            hypothesis_template=HYPOTHESE_VORLAGE,
        )

        # Scores den Labels zuordnen
        scores = dict(zip(resultat["labels"], resultat["scores"]))
        score_kern = scores.get("Kernbotschaft", 0.0)
        score_wichtig = scores.get("wichtige Aussage", 0.0)
        gesamt_score = (score_kern + score_wichtig) / 2.0

        if gesamt_score >= KERNBOTSCHAFT_SCHWELLE:
            kernbotschaften.append({
                "satz_id": satz["satz_id"],
                "text": satz["text"],
                "start": satz["start"],
                "end": satz["end"],
                "score": round(gesamt_score, 4),
            })

    print()  # Zeilenumbruch nach Fortschrittsanzeige

    # Absteigend nach Score sortieren
    kernbotschaften.sort(key=lambda k: k["score"], reverse=True)
    return kernbotschaften


# ════════════════════════════════════════════════════════════════════
# SCHRITT 7 — Nebensätze (abgeleitet)
# ════════════════════════════════════════════════════════════════════

def berechne_nebensaetze(satzgrenzen, kernbotschaften):
    """Alle Sätze die KEINE Kernbotschaft sind → Nebensatz.

    Rückgabe: Liste von Dicts:
        { "satz_id": 5, "text": "...", "start": "...", "end": "..." }
    """
    kern_ids = {k["satz_id"] for k in kernbotschaften}

    return [
        {
            "satz_id": s["satz_id"],
            "text": s["text"],
            "start": s["start"],
            "end": s["end"],
        }
        for s in satzgrenzen
        if s["satz_id"] not in kern_ids
    ]


# ════════════════════════════════════════════════════════════════════
# SCHRITT 8 — Struktur (Zero-Shot AI + Zeit-Plausibilität)
# ════════════════════════════════════════════════════════════════════

def berechne_struktur(satzgrenzen, zero_shot):
    """Erkennt Struktur-Segmente: Einleitung / Hauptteil / Übergang / Schluss.

    Ablauf:
      1. Gesamtlänge = letzter End-Zeitstempel
      2. Pro Satz: Zero-Shot-Klassifikation → Label mit höchstem Score
      3. Zeit-Plausibilität:
           'Einleitung' nur in ersten 15% erlaubt
           'Schluss'/'Zusammenfassung' nur in letzten 15% erlaubt
           sonst → 'Hauptteil'
         ('Zusammenfassung' wird zu 'Schluss' zusammengefasst)
      4. Aufeinanderfolgende Sätze mit gleichem Typ → ein Segment

    Rückgabe: Liste von Dicts:
        { "typ": "Einleitung", "start": "...", "end": "..." }
    """
    if not satzgrenzen:
        return []

    # 1. Gesamtlänge & Zeitgrenzen
    gesamt_ende = zeit_zu_sekunden(satzgrenzen[-1]["end"])
    gesamt_start = zeit_zu_sekunden(satzgrenzen[0]["start"])
    dauer = max(gesamt_ende - gesamt_start, 0.001)

    einleitung_grenze = gesamt_start + dauer * EINLEITUNG_ANTEIL
    schluss_grenze = gesamt_ende - dauer * SCHLUSS_ANTEIL

    # 2. + 3. Pro Satz klassifizieren + plausibilisieren
    satz_typen = []
    gesamt = len(satzgrenzen)

    for i, satz in enumerate(satzgrenzen, start=1):
        print(f"\r    Analysiere Satz {i}/{gesamt}...", end="", flush=True)

        resultat = zero_shot(
            satz["text"],
            candidate_labels=STRUKTUR_LABELS,
            multi_label=False,
            hypothesis_template=HYPOTHESE_VORLAGE,
        )
        typ = resultat["labels"][0]  # Label mit höchstem Score

        # Zeitliche Mitte des Satzes
        mitte = (zeit_zu_sekunden(satz["start"])
                 + zeit_zu_sekunden(satz["end"])) / 2.0

        # Plausibilitätsprüfung
        if typ == "Einleitung" and mitte > einleitung_grenze:
            typ = "Hauptteil"
        if typ in ("Schluss", "Zusammenfassung"):
            if mitte < schluss_grenze:
                typ = "Hauptteil"
            else:
                typ = "Schluss"  # Zusammenfassung → Schluss

        satz_typen.append((satz, typ))

    print()

    # 4. Aufeinanderfolgende gleiche Typen zu Segmenten zusammenfassen
    segmente = []
    for satz, typ in satz_typen:
        if segmente and segmente[-1]["typ"] == typ:
            segmente[-1]["end"] = satz["end"]
        else:
            segmente.append({
                "typ": typ,
                "start": satz["start"],
                "end": satz["end"],
            })

    return segmente


# ════════════════════════════════════════════════════════════════════
# SCHRITT 9 — Emotionaler Ton (Sentiment AI)
# ════════════════════════════════════════════════════════════════════

def berechne_emotionaler_ton(satzgrenzen, sentiment):
    """Bestimmt den emotionalen Gesamtton der Präsentation.

    Ablauf:
      1. Alle Sätze zu Abschnitten von max. ~250 Wörtern gruppieren
         (Sicherheitsabstand zum 512-Token-Limit des Modells)
      2. Pro Abschnitt: Sentiment-Scores (positiv/neutral/negativ)
      3. Durchschnitt über alle Abschnitte
      4. Label mappen: positiv → inspirierend, neutral → sachlich,
         negativ → ernst

    Rückgabe: { "label": "inspirierend", "score": 0.87 }
    """
    # 1. Abschnitte bilden
    abschnitte = []
    aktueller = []
    wortzahl = 0

    for satz in satzgrenzen:
        n = len(satz["text"].split())
        if wortzahl + n > SENTIMENT_MAX_WOERTER and aktueller:
            abschnitte.append(" ".join(aktueller))
            aktueller = []
            wortzahl = 0
        aktueller.append(satz["text"])
        wortzahl += n

    if aktueller:
        abschnitte.append(" ".join(aktueller))

    if not abschnitte:
        return {"label": "sachlich", "score": 0.0}

    # 2. + 3. Pro Abschnitt klassifizieren, Scores mitteln
    summen = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    for abschnitt in abschnitte:
        resultat = sentiment(abschnitt, truncation=True, max_length=512)

        # Pipeline mit top_k=None gibt Liste aller Labels zurück.
        # Format kann [[{...}]] oder [{...}] sein — beide abfangen.
        if resultat and isinstance(resultat[0], list):
            eintraege = resultat[0]
        else:
            eintraege = resultat

        for eintrag in eintraege:
            label = eintrag["label"].lower()
            if label in summen:
                summen[label] += eintrag["score"]

    mittel = {label: wert / len(abschnitte) for label, wert in summen.items()}

    # 4. Bestes Label wählen + mappen
    bestes_label = max(mittel, key=mittel.get)

    return {
        "label": TON_MAPPING[bestes_label],
        "score": round(mittel[bestes_label], 4),
    }


# ════════════════════════════════════════════════════════════════════
# SCHRITT 10 — Publikumsbezug (algorithmisch)
# ════════════════════════════════════════════════════════════════════

def berechne_publikumsbezug(satzgrenzen):
    """Erkennt Sätze mit direkter Publikumsansprache.

    Regeln:
      - 'wir', 'uns', 'euch', 'du', 'dir', 'dich'
        → immer Ansprache (case-insensitive)
      - 'Sie', 'Ihnen', 'Ihr' (grossgeschrieben)
        → nur Ansprache wenn NICHT am Satzanfang
          (am Satzanfang nicht unterscheidbar von 'sie' = dritte Person)

    Rückgabe: Liste von Dicts:
        { "satz_id": 9, "text": "...", "start": "...", "end": "..." }
    """
    publikumsbezug = []

    for satz in satzgrenzen:
        # Wörter ohne Satzzeichen extrahieren
        tokens = re.findall(r"[\wäöüÄÖÜß]+", satz["text"])
        gefunden = False

        for pos, token in enumerate(tokens):
            # Immer-Ansprache (case-insensitive)
            if token.lower() in ANSPRACHE_IMMER:
                gefunden = True
                break
            # Formelle Ansprache: exakt grossgeschrieben + nicht Satzanfang
            if token in ANSPRACHE_FORMELL and pos > 0:
                gefunden = True
                break

        if gefunden:
            publikumsbezug.append({
                "satz_id": satz["satz_id"],
                "text": satz["text"],
                "start": satz["start"],
                "end": satz["end"],
            })

    return publikumsbezug


# ════════════════════════════════════════════════════════════════════
# OUTPUT — JSON + TXT-Bericht
# ════════════════════════════════════════════════════════════════════

def schreibe_json(output, root):
    """Speichert den JSON-Output nach zwischen_output/."""
    ordner = os.path.join(root, "zwischen_output")
    os.makedirs(ordner, exist_ok=True)
    pfad = os.path.join(ordner, "inhalt_analyse_output.json")

    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return pfad


def schreibe_bericht(output, root, transkript_name):
    """Schreibt den lesbaren TXT-Bericht nach reports/."""
    ordner = os.path.join(root, "reports")
    os.makedirs(ordner, exist_ok=True)
    basis = os.path.splitext(os.path.basename(transkript_name))[0]
    pfad = os.path.join(ordner, f"inhalt_analyse_bericht_{basis}.txt")

    sg = output["satzgrenzen"]
    kb = output["kernbotschaften"]
    ns = output["nebensaetze"]
    st = output["struktur"]
    rm = output["rhetorische_momente"]
    ss = output["satzstruktur"]
    et = output["emotionaler_ton"]
    pb = output["publikumsbezug"]

    z = []  # Zeilen des Berichts
    breite = 70

    def titel(text):
        z.append("")
        z.append("═" * breite)
        z.append(f"  {text}")
        z.append("═" * breite)

    # ── Kopf ──
    z.append("═" * breite)
    z.append("  INHALTSANALYSE — BERICHT")
    z.append("  praesentation_ai")
    z.append("═" * breite)
    z.append(f"  Transkript:  {os.path.basename(transkript_name)}")
    z.append(f"  Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    if sg:
        dauer_s = zeit_zu_sekunden(sg[-1]["end"])
        minuten = int(dauer_s // 60)
        sekunden = int(dauer_s % 60)
        z.append(f"  Dauer:       {minuten} Min {sekunden} Sek")
    z.append(f"  Sätze:       {len(sg)}")

    # ── 1. Emotionaler Ton ──
    titel("1. EMOTIONALER GESAMTTON")
    z.append(f"  Ton:   {et['label']}")
    z.append(f"  Score: {et['score']:.2f}")

    # ── 2. Struktur ──
    titel("2. STRUKTUR DER PRÄSENTATION")
    if st:
        for seg in st:
            z.append(f"  {seg['start']} – {seg['end']}   {seg['typ']}")
    else:
        z.append("  Keine Struktur-Segmente erkannt.")

    # ── 3. Kernbotschaften ──
    titel(f"3. KERNBOTSCHAFTEN ({len(kb)})")
    if kb:
        for k in kb:
            z.append(f"  [{k['start']}]  (Score {k['score']:.2f})")
            z.append(f"    «{k['text']}»")
    else:
        z.append("  Keine Kernbotschaften erkannt.")
    z.append("")
    z.append(f"  Nebensätze: {len(ns)} "
             f"(alle übrigen Sätze, Details im JSON)")

    # ── 4. Rhetorische Momente ──
    fragen = [m for m in rm if m["typ"] == "Frage"]
    hoehepunkte = [m for m in rm if m["typ"] == "Höhepunkt"]
    titel(f"4. RHETORISCHE MOMENTE ({len(rm)})")
    z.append(f"  Fragen:     {len(fragen)}")
    z.append(f"  Höhepunkte: {len(hoehepunkte)}")
    for m in rm:
        z.append(f"  [{m['start']}]  {m['typ']}")
        z.append(f"    «{m['text']}»")

    # ── 5. Satzstruktur ──
    titel("5. SATZSTRUKTUR")
    anzahl_fragen = sum(1 for s in ss if s["typ"] == "Frage")
    anzahl_aussagen = len(ss) - anzahl_fragen
    anzahl_kurz = sum(1 for s in ss if s["laenge"] == "kurz")
    anzahl_lang = sum(1 for s in ss if s["laenge"] == "lang")
    anzahl_komplex = sum(1 for s in ss if s["laenge"] == "komplex")
    z.append(f"  Aussagen: {anzahl_aussagen}   Fragen: {anzahl_fragen}")
    z.append(f"  Kurz: {anzahl_kurz}   Lang: {anzahl_lang}   "
             f"Komplex: {anzahl_komplex}")
    if ss:
        durchschnitt = sum(s["wortanzahl"] for s in ss) / len(ss)
        z.append(f"  Durchschnittliche Satzlänge: {durchschnitt:.1f} Wörter")

    # ── 6. Publikumsbezug ──
    titel(f"6. PUBLIKUMSBEZUG ({len(pb)} Sätze mit direkter Ansprache)")
    if sg:
        anteil = len(pb) / len(sg) * 100
        z.append(f"  Anteil: {anteil:.0f}% aller Sätze")
    for p_satz in pb[:10]:
        z.append(f"  [{p_satz['start']}]  «{p_satz['text']}»")
    if len(pb) > 10:
        z.append(f"  ... und {len(pb) - 10} weitere (Details im JSON)")

    # ── Fuss ──
    titel("HINWEIS FÜR NACHGELAGERTE SCRIPTS")
    z.append("  Der vollständige maschinenlesbare Output liegt in:")
    z.append("  zwischen_output/inhalt_analyse_output.json")
    z.append("")
    z.append("  Enthaltene Felder: satzgrenzen, kernbotschaften,")
    z.append("  nebensaetze, struktur, rhetorische_momente,")
    z.append("  satzstruktur, emotionaler_ton, publikumsbezug")
    z.append("═" * breite)

    with open(pfad, "w", encoding="utf-8") as f:
        f.write("\n".join(z))

    return pfad


# ════════════════════════════════════════════════════════════════════
# MAIN — Orchestrierung
# ════════════════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 60)
    print("  praesentation_ai — INHALTSANALYSE")
    print("═" * 60)

    # ── Schritt 1: Transkript einlesen ──
    print("\n[1/9] Transkript einlesen...")
    # Optional: Pfad als Kommandozeilen-Argument (sonst Dateidialog)
    pfad_argument = sys.argv[1] if len(sys.argv) > 1 else None
    woerter, transkript_pfad = lade_transkript(pfad_argument)
    print(f"  ✓ {len(woerter)} Wörter geladen aus "
          f"{os.path.basename(transkript_pfad)}")

    # ── Schritt 2: Modelle laden ──
    print("\n[2/9] Modelle laden...")
    nlp, zero_shot, sentiment = lade_modelle()
    print("  ✓ Alle Modelle geladen")

    # ── Schritt 3: Satzgrenzen ──
    print("\n[3/9] Satzgrenzen erkennen (spaCy)...")
    satzgrenzen = berechne_satzgrenzen(woerter, nlp)
    print(f"  ✓ {len(satzgrenzen)} Sätze erkannt")

    # ── Schritt 4: Satzstruktur ──
    print("\n[4/9] Satzstruktur analysieren (spaCy)...")
    satzstruktur = berechne_satzstruktur(satzgrenzen, nlp)
    print(f"  ✓ {len(satzstruktur)} Sätze klassifiziert")

    # ── Schritt 5: Rhetorische Momente ──
    print("\n[5/9] Rhetorische Momente erkennen (spaCy)...")
    rhetorische_momente = berechne_rhetorische_momente(satzgrenzen, nlp)
    print(f"  ✓ {len(rhetorische_momente)} Momente erkannt")

    # ── Schritt 6: Kernbotschaften ──
    print("\n[6/9] Kernbotschaften erkennen (Zero-Shot AI)...")
    print("  Hinweis: Das dauert auf CPU einige Minuten.")
    kernbotschaften = berechne_kernbotschaften(satzgrenzen, zero_shot)
    print(f"  ✓ {len(kernbotschaften)} Kernbotschaften gefunden")

    # ── Schritt 7: Nebensätze ──
    print("\n[7/9] Nebensätze ableiten...")
    nebensaetze = berechne_nebensaetze(satzgrenzen, kernbotschaften)
    print(f"  ✓ {len(nebensaetze)} Nebensätze")

    # ── Schritt 8: Struktur + Ton + Publikumsbezug ──
    print("\n[8/9] Struktur erkennen (Zero-Shot AI)...")
    struktur = berechne_struktur(satzgrenzen, zero_shot)
    print(f"  ✓ {len(struktur)} Struktur-Segmente")

    print("      Emotionalen Ton bestimmen (Sentiment AI)...")
    emotionaler_ton = berechne_emotionaler_ton(satzgrenzen, sentiment)
    print(f"  ✓ Ton: {emotionaler_ton['label']} "
          f"(Score {emotionaler_ton['score']:.2f})")

    print("      Publikumsbezug erkennen...")
    publikumsbezug = berechne_publikumsbezug(satzgrenzen)
    print(f"  ✓ {len(publikumsbezug)} Sätze mit direkter Ansprache")

    # ── Schritt 9: Output ──
    print("\n[9/9] Output speichern...")
    output = {
        "meta": {
            "transkript": os.path.basename(transkript_pfad),
            "erstellt_am": datetime.now().isoformat(timespec="seconds"),
            "anzahl_woerter": len(woerter),
            "anzahl_saetze": len(satzgrenzen),
        },
        "satzgrenzen": satzgrenzen,
        "kernbotschaften": kernbotschaften,
        "nebensaetze": nebensaetze,
        "struktur": struktur,
        "rhetorische_momente": rhetorische_momente,
        "satzstruktur": satzstruktur,
        "emotionaler_ton": emotionaler_ton,
        "publikumsbezug": publikumsbezug,
    }

    root = projekt_root()
    json_pfad = schreibe_json(output, root)
    bericht_pfad = schreibe_bericht(output, root, transkript_pfad)

    print(f"  ✓ JSON:    {json_pfad}")
    print(f"  ✓ Bericht: {bericht_pfad}")

    print()
    print("═" * 60)
    print("  ✓ INHALTSANALYSE ABGESCHLOSSEN")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
