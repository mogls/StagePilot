# -*- coding: utf-8 -*-
"""════════════════════════════════════════════════════════════════════
 inhalt_analyse.py — Transkript-Inhaltsanalyse (v2 — gehärtet)
 Projekt: präsentation_ai
════════════════════════════════════════════════════════════════════

Analysiert ein Transkript (Standardformat: Wort  00:00:12.300  00:00:12.780)
und erzeugt alle Inhalts-Informationen, die die nachgelagerten
Audio-Analyse-Scripts benötigen.

INPUT:
    Transkript-Datei (.txt) — Auswahl über tkinter-Dateidialog
    oder Pfad als Kommandozeilen-Argument

OUTPUT:
    1. zwischen_output/inhalt_analyse_output.json   (für Audio-Scripts)
    2. reports/inhalt_analyse_bericht_<name>.txt    (lesbarer Bericht)

════════════════════════════════════════════════════════════════════

MODELLE (werden einmalig geladen):
    - spaCy de_core_news_sm                  → Satzgrenzen, Satzstruktur,
                                               rhetorische Momente
    - Sahajtomar/German_Zeroshot             → Kernbotschaften, Struktur
    - oliverguhr/german-sentiment-bert       → Emotionaler Ton"""

import json
import os
import re
import sys
from datetime import datetime

# ════════════════════════════════════════════════════════════════════
# GLOBALE WARNUNGEN-LISTE
# ════════════════════════════════════════════════════════════════════
# Nicht-fatale Anomalien werden hier gesammelt und landen im JSON-Output
# unter meta.warnungen. So kann der User später nachvollziehen was
# problematisch war ohne dass die Analyse abbricht.
WARNUNGEN = []


def warn(nachricht):
    """Fügt eine Warnung zur globalen Liste hinzu und gibt sie aus."""
    WARNUNGEN.append(nachricht)
    print(f"  ⚠ {nachricht}")


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

# Fallback: wenn 0 Kernbotschaften gefunden werden, nimm die
# obersten 10% der Sätze nach Score (mindestens 1, höchstens 5).
# So haben nachgelagerte Scripts immer mindestens einen Vergleichspunkt.
KERNBOTSCHAFT_FALLBACK_ANTEIL = 0.10
KERNBOTSCHAFT_FALLBACK_MIN = 1
KERNBOTSCHAFT_FALLBACK_MAX = 5

# Struktur: Zero-Shot-Labels + Zeitgrenzen
STRUKTUR_LABELS = [
    "Einleitung", "Hauptteil", "Übergang", "Schluss", "Zusammenfassung",
]
EINLEITUNG_ANTEIL = 0.15   # Einleitung nur in ersten 15% erlaubt
SCHLUSS_ANTEIL = 0.15      # Schluss nur in letzten 15% erlaubt

# Bei sehr kurzen Präsentationen (< 10 Sätze) ist die Prozent-Regel
# unzuverlässig. Dann werden feste Satz-Positionen verwendet:
STRUKTUR_KURZ_SCHWELLE = 10
STRUKTUR_KURZ_EINLEITUNG_SAETZE = 1   # erste N Sätze = Einleitung möglich
STRUKTUR_KURZ_SCHLUSS_SAETZE = 1      # letzte N Sätze = Schluss möglich

# Hypothesen-Vorlage für das deutsche Zero-Shot-Modell
HYPOTHESE_VORLAGE = "Dieser Satz ist {}."

# Zero-Shot: bei sehr langen Sätzen truncieren
ZEROSHOT_MAX_ZEICHEN = 500

# Satzstruktur: Längen-Grenzen (Wortanzahl)
KURZ_MAX = 8       # <= 8 Wörter  → kurz
LANG_MAX = 20      # 9–20 Wörter  → lang, > 20 → komplex

# Rhetorische Momente: Schlüsselwörter für Höhepunkte
# Wortstämme werden verwendet, damit auch Flexionen erkannt werden
# (z.B. "wichtig", "wichtige", "wichtigen" → Stamm "wichtig")
HOEHEPUNKT_STAEMME = {
    "wichtig",     # wichtig, wichtige, wichtigen, wichtigsten...
    "entscheid",   # entscheidend, entscheidende, entscheidet...
    "niemals",
    "immer",
    "jeder",       # jeder, jede, jedes, jeden
    "jede",
    "jedes",
    "jeden",
}

# Publikumsbezug: Ansprache-Wörter
# Kleingeschriebene Formen — immer Ansprache (case-insensitive geprüft)
ANSPRACHE_IMMER = {"wir", "uns", "euch", "du", "dir", "dich", "unser",
                   "unsere", "unseren", "unserem", "unseres"}
# Grossgeschriebene Formen — nur Ansprache wenn wirklich grossgeschrieben
# UND nicht am Satzanfang (sonst nicht unterscheidbar von "sie" = dritte Person)
ANSPRACHE_FORMELL = {"Sie", "Ihnen", "Ihr", "Ihre", "Ihren", "Ihrem", "Ihres"}

# "ihr" (klein) wird bewusst NICHT aufgenommen:
# - kann 2. Person Plural (Ansprache) sein ODER
# - Possessivpronomen 3. Person Sing. fem./Plur. ("ihr Auto")
# - Dativ 3. Person Sing. fem. ("ich gebe ihr das Buch")
# Da die Mehrdeutigkeit ohne Kontext-Analyse nicht auflösbar ist,
# würde das zu vielen false positives führen.

# Emotionaler Ton: Sentiment-Label → Präsentations-Ton
TON_MAPPING = {
    "positive": "inspirierend",
    "neutral": "sachlich",
    "negative": "ernst",
}

# Sentiment-Modell: max. Wörter pro Abschnitt (Sicherheitsabstand zu 512 Tokens)
SENTIMENT_MAX_WOERTER = 250

# Encoding-Reihenfolge für Fallback beim Transkript-Einlesen
ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


# ════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN — Zeitstempel
# ════════════════════════════════════════════════════════════════════

# Regex akzeptiert sowohl Punkt als auch Komma als Millisekunden-Trenner
# (verschiedene Whisper-Ausgaben verwenden unterschiedliche Konventionen)
ZEIT_REGEX = re.compile(r"^(\d+):(\d+):(\d+)[.,](\d+)$")


def zeit_zu_sekunden(zeitstempel):
    """Wandelt 'HH:MM:SS.mmm' oder 'HH:MM:SS,mmm' in Sekunden (float) um.

    Beispiel: '00:01:12.300' → 72.3
              '00:01:12,300' → 72.3
    """
    if zeitstempel is None:
        raise ValueError("Zeitstempel ist None")

    match = ZEIT_REGEX.match(str(zeitstempel).strip())
    if not match:
        raise ValueError(f"Ungültiger Zeitstempel: '{zeitstempel}'")

    stunden, minuten, sekunden, millis = match.groups()

    # Millisekunden auf 3 Stellen normalisieren
    # ('3' → '300', '3456' → '345', '30' → '300')
    millis = millis.ljust(3, "0")[:3]

    m_int = int(minuten)
    s_int = int(sekunden)

    # Sanity-Check: Minuten und Sekunden im gültigen Bereich
    if m_int >= 60 or s_int >= 60:
        raise ValueError(
            f"Ungültiger Zeitstempel (Minuten/Sekunden >= 60): '{zeitstempel}'"
        )

    return (
        int(stunden) * 3600
        + m_int * 60
        + s_int
        + int(millis) / 1000.0
    )


def sekunden_zu_zeit(sekunden):
    """Wandelt Sekunden (float) in 'HH:MM:SS.mmm' um.

    Beispiel: 72.3 → '00:01:12.300'
    """
    # Negative Werte auf 0 klemmen (kann durch Rundungsfehler passieren)
    if sekunden < 0:
        sekunden = 0.0

    millis = int(round(sekunden * 1000))
    stunden, rest = divmod(millis, 3600 * 1000)
    minuten, rest = divmod(rest, 60 * 1000)
    sek, ms = divmod(rest, 1000)
    return f"{stunden:02d}:{minuten:02d}:{sek:02d}.{ms:03d}"


# ════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN — Pfade & Dateinamen
# ════════════════════════════════════════════════════════════════════

def projekt_root():
    """Ermittelt den Projekt-Root (Ordner über analyse/).

    Liegt das Script in präsentation_ai/analyse/, ist der Root
    präsentation_ai/. Sonst wird der Ordner des Scripts verwendet.
    """
    script_ordner = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_ordner).lower() == "analyse":
        return os.path.dirname(script_ordner)
    return script_ordner


def sicherer_dateiname(name):
    """Ersetzt Sonderzeichen in Dateinamen durch Unterstriche.

    Verhindert Fehler bei ungewöhnlichen Transkript-Namen
    (z.B. Doppelpunkte, Slashes, Umlaute in Umgebungen ohne UTF-8).
    """
    # Nur alphanumerische Zeichen, Unterstriche, Bindestriche und
    # gängige deutsche Sonderzeichen behalten
    return re.sub(r"[^\w\-äöüÄÖÜß]", "_", name)


# ════════════════════════════════════════════════════════════════════
# SCHRITT 1 — Transkript einlesen (mit Encoding-Fallback)
# ════════════════════════════════════════════════════════════════════

def _lese_datei_mit_encoding_fallback(pfad):
    """Liest eine Textdatei mit automatischem Encoding-Fallback.

    Versucht in dieser Reihenfolge:
      1. UTF-8 (Standard)
      2. UTF-8-BOM (Windows Notepad Default)
      3. CP1252 (Windows Standard)
      4. Latin-1 (Fallback, kann nie fehlschlagen)

    Bei jedem Fallback wird eine Warnung ausgegeben.
    """
    letzter_fehler = None

    for encoding in ENCODING_FALLBACKS:
        try:
            with open(pfad, "r", encoding=encoding) as f:
                inhalt = f.read()
            if encoding != "utf-8":
                warn(f"Transkript nicht als UTF-8 lesbar. "
                     f"Verwendet: {encoding}. "
                     f"Empfehlung: Datei in UTF-8 speichern.")
            return inhalt
        except UnicodeDecodeError as e:
            letzter_fehler = e
            continue

    # Sollte nie hier ankommen, weil Latin-1 immer funktioniert
    raise letzter_fehler


def lade_transkript(pfad=None):
    """Liest die Transkript-Datei ein.

    Öffnet einen tkinter-Dateidialog (falls kein Pfad übergeben wurde)
    und parst jede Zeile im Standardformat:

        Wort  00:00:12.300  00:00:12.780

    Rückgabe: (wörter, pfad)
        wörter = Liste von Dicts:
        {
          "wort":    "Bildung",
          "start":   "00:00:12.300",
          "end":     "00:00:12.780",
          "start_s": 12.3,
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

    # Datei mit Encoding-Fallback lesen
    try:
        inhalt = _lese_datei_mit_encoding_fallback(pfad)
    except Exception as e:
        print(f"  ✗ Datei konnte nicht gelesen werden: {e}")
        sys.exit(1)

    if not inhalt.strip():
        print("  ✗ Transkript-Datei ist leer — Abbruch.")
        sys.exit(1)

    woerter = []
    fehlerhafte_zeilen = 0
    letzte_end_s = -1.0

    for zeilen_nr, zeile in enumerate(inhalt.splitlines(), start=1):
        zeile = zeile.strip()
        if not zeile:
            continue  # Leere Zeilen überspringen

        teile = zeile.split()
        # Erwartet: mindestens Wort + 2 Zeitstempel.
        # Falls das "Wort" aus mehreren Teilen besteht,
        # sind die letzten 2 Teile die Zeitstempel.
        if len(teile) < 3:
            fehlerhafte_zeilen += 1
            if fehlerhafte_zeilen <= 5:
                print(f"  ⚠ Zeile {zeilen_nr} übersprungen "
                      f"(zu wenige Spalten): '{zeile[:80]}'")
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
                      f"(ungültiger Zeitstempel): '{zeile[:80]}'")
            continue

        # Sanity-Check: Start muss vor End sein
        # Bei Gleichheit (z.B. sehr kurze Wörter) tolerieren wir das
        if start_s > end_s:
            warn(f"Zeile {zeilen_nr}: Start-Zeit ({start_str}) liegt nach "
                 f"End-Zeit ({end_str}). Korrigiere durch Vertauschen.")
            start_s, end_s = end_s, start_s
            start_str, end_str = end_str, start_str

        # Sanity-Check: monotone Reihenfolge
        # Kleine Rückschritte tolerieren wir (< 100ms, kann durch
        # Whisper-Genauigkeit passieren), größere loggen wir
        if start_s < letzte_end_s - 0.1:
            warn(f"Zeile {zeilen_nr}: Zeitstempel läuft rückwärts "
                 f"(vorher endete bei {sekunden_zu_zeit(letzte_end_s)}, "
                 f"jetzt startet bei {start_str}). Wort trotzdem behalten.")

        letzte_end_s = max(letzte_end_s, end_s)

        woerter.append({
            "wort": wort,
            "start": start_str,
            "end": end_str,
            "start_s": start_s,
            "end_s": end_s,
        })

    if fehlerhafte_zeilen > 5:
        warn(f"Insgesamt {fehlerhafte_zeilen} fehlerhafte Zeilen "
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
    except Exception as e:
        print(f"  ✗ spaCy konnte nicht geladen werden: {e}")
        sys.exit(1)
    print("    ✓ spaCy bereit")

    print("  → Zero-Shot-Modell laden (beim ersten Mal: Download ~1.4 GB)...")
    try:
        zero_shot = pipeline(
            "zero-shot-classification",
            model=ZEROSHOT_MODELL,
            device=-1,  # CPU
        )
    except Exception as e:
        print(f"  ✗ Zero-Shot-Modell konnte nicht geladen werden: {e}")
        print("    Prüfe Internetverbindung (beim ersten Start) oder Speicher.")
        sys.exit(1)
    print("    ✓ Zero-Shot bereit")

    print("  → Sentiment-Modell laden (beim ersten Mal: Download ~0.4 GB)...")
    try:
        sentiment = pipeline(
            "text-classification",
            model=SENTIMENT_MODELL,
            device=-1,  # CPU
            top_k=None,  # Alle Label-Scores zurückgeben
        )
    except Exception as e:
        print(f"  ✗ Sentiment-Modell konnte nicht geladen werden: {e}")
        sys.exit(1)
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

    Fallback: Wenn spaCy keine Satzgrenzen erkennt (z.B. weil
    das Transkript keine Interpunktion enthält), wird das gesamte
    Transkript als ein Satz behandelt.

    Rückgabe: Liste von Dicts:
        { "satz_id": 1, "text": "...", "start": "...", "end": "..." }
    """
    if not woerter:
        return []

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

        # Text bereinigen: Leere Sätze überspringen
        satz_text = sent.text.strip()
        if not satz_text:
            continue

        satz_id += 1
        erstes = woerter[indizes[0]]
        letztes = woerter[indizes[-1]]

        satzgrenzen.append({
            "satz_id": satz_id,
            "text": satz_text,
            "start": erstes["start"],
            "end": letztes["end"],
        })

    # Fallback: spaCy hat gar keine Sätze erkannt
    # (kann bei Transkripten ohne Interpunktion passieren)
    if not satzgrenzen:
        warn("spaCy erkennt keine Satzgrenzen. Behandle Transkript als "
             "einen einzigen Satz. Empfehlung: Transkript mit Interpunktion "
             "versehen.")
        satzgrenzen.append({
            "satz_id": 1,
            "text": volltext.strip(),
            "start": woerter[0]["start"],
            "end": woerter[-1]["end"],
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

    Rückgabe: Liste von Dicts.
    """
    satzstruktur = []

    for satz in satzgrenzen:
        text = satz["text"]

        # Empty-Guard: Leere Texte überspringen (sollte nach
        # satzgrenzen-Bereinigung nicht mehr auftreten, aber sicher ist sicher)
        if not text or not text.strip():
            continue

        try:
            doc = nlp(text)
        except Exception as e:
            warn(f"spaCy-Analyse fehlgeschlagen für Satz {satz['satz_id']}: {e}")
            # Fallback-Klassifikation ohne spaCy
            satzstruktur.append({
                "satz_id": satz["satz_id"],
                "text": text,
                "typ": "Aussage",
                "laenge": "lang",
                "wortanzahl": len(text.split()),
                "start": satz["start"],
                "end": satz["end"],
            })
            continue

        # Wortanzahl (nur echte Wörter, keine Satzzeichen)
        wortanzahl = sum(1 for token in doc if not token.is_punct)

        # Bei 0 Wörtern: Mindestwert 1 (verhindert Division-durch-0 später)
        if wortanzahl == 0:
            wortanzahl = 1

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

def _ist_hoehepunkt_wort(token_text):
    """Prüft ob ein Wort ein Höhepunkt-Signal enthält.

    Wortstamm-Matching statt exaktem Match, damit auch Flexionen
    erkannt werden: 'wichtig' matcht 'wichtige', 'wichtigen',
    'wichtigsten' etc.
    """
    wort_klein = token_text.lower()
    return any(wort_klein.startswith(stamm) for stamm in HOEHEPUNKT_STAEMME)


def berechne_rhetorische_momente(satzgrenzen, nlp):
    """Erkennt Fragen und Höhepunkte.

    Frage:      Satz endet auf '?'
    Höhepunkt:  Satz endet auf '!'
                ODER enthält Superlativ (spaCy Morphologie)
                ODER enthält Wortstamm-Match aus HÖHEPUNKT_STAEMME

    Ein Satz kann nur Frage ODER Höhepunkt sein — Frage hat Priorität.
    """
    momente = []

    for satz in satzgrenzen:
        text = satz["text"].rstrip()
        if not text:
            continue

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
                try:
                    doc = nlp(text)
                    for token in doc:
                        # Superlativ (z.B. "am wichtigsten", "grösste")
                        # Robuste Prüfung: get() kann leere Liste zurückgeben
                        degree = token.morph.get("Degree")
                        if degree and "Sup" in degree:
                            ist_hoehepunkt = True
                            break
                        # Schlüsselwort-Wortstamm-Match
                        if _ist_hoehepunkt_wort(token.text):
                            ist_hoehepunkt = True
                            break
                except Exception:
                    # Bei spaCy-Fehler: nur einfaches Match ohne Morphologie
                    for wort in text.split():
                        if _ist_hoehepunkt_wort(wort):
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

def _truncate_text(text, max_zeichen):
    """Kürzt einen Text auf max. Zeichenanzahl, ohne Wörter zu zerreißen."""
    if len(text) <= max_zeichen:
        return text
    # Am letzten Leerzeichen vor max_zeichen abschneiden
    gekuerzt = text[:max_zeichen]
    letztes_leerzeichen = gekuerzt.rfind(" ")
    if letztes_leerzeichen > max_zeichen * 0.8:
        return gekuerzt[:letztes_leerzeichen]
    return gekuerzt


def berechne_kernbotschaften(satzgrenzen, zero_shot):
    """Erkennt Kernbotschaften mit dem Zero-Shot-Modell.

    Pro Satz:
      - Zero-Shot-Klassifikation mit 5 Labels
      - Gesamt-Score = Durchschnitt aus Score('Kernbotschaft')
                       und Score('wichtige Aussage')
      - Gesamt-Score >= 0.6 → Kernbotschaft

    Fallback: Wenn 0 Sätze die Schwelle erreichen, werden die Top-N
    Sätze nach Score als Kernbotschaften markiert (verhindert dass
    nachgelagerte Scripts keine Vergleichspunkte haben).

    Rückgabe: Liste von Dicts (absteigend nach Score sortiert).
    """
    if not satzgrenzen:
        return []

    # Erst alle Sätze klassifizieren und den Score speichern,
    # dann Schwellenwert anwenden (nötig für Fallback)
    alle_scores = []
    gesamt = len(satzgrenzen)

    for i, satz in enumerate(satzgrenzen, start=1):
        # Fortschrittsanzeige (gleiche Zeile überschreiben)
        print(f"\r    Analysiere Satz {i}/{gesamt}...", end="", flush=True)

        text = satz["text"].strip()

        # Empty-Guard: Modell nicht mit Leertext füttern
        if not text:
            alle_scores.append((satz, 0.0))
            continue

        # Truncation bei sehr langen Sätzen
        text_fuer_modell = _truncate_text(text, ZEROSHOT_MAX_ZEICHEN)

        try:
            resultat = zero_shot(
                text_fuer_modell,
                candidate_labels=KERNBOTSCHAFT_LABELS,
                multi_label=False,
                hypothesis_template=HYPOTHESE_VORLAGE,
            )
        except Exception as e:
            warn(f"Zero-Shot fehlgeschlagen für Satz {satz['satz_id']}: "
                 f"{str(e)[:80]}")
            alle_scores.append((satz, 0.0))
            continue

        # Scores den Labels zuordnen
        scores = dict(zip(resultat["labels"], resultat["scores"]))
        score_kern = scores.get("Kernbotschaft", 0.0)
        score_wichtig = scores.get("wichtige Aussage", 0.0)
        gesamt_score = (score_kern + score_wichtig) / 2.0

        alle_scores.append((satz, gesamt_score))

    print()  # Zeilenumbruch nach Fortschrittsanzeige

    # Primäre Auswahl: alle Sätze über Schwelle
    kernbotschaften = [
        {
            "satz_id": satz["satz_id"],
            "text": satz["text"],
            "start": satz["start"],
            "end": satz["end"],
            "score": round(score, 4),
        }
        for satz, score in alle_scores
        if score >= KERNBOTSCHAFT_SCHWELLE
    ]

    # Fallback: wenn 0 Sätze die Schwelle erreichen
    if not kernbotschaften:
        # Top-N nach Score bestimmen
        anzahl_fallback = max(
            KERNBOTSCHAFT_FALLBACK_MIN,
            min(
                KERNBOTSCHAFT_FALLBACK_MAX,
                int(len(alle_scores) * KERNBOTSCHAFT_FALLBACK_ANTEIL),
            )
        )
        anzahl_fallback = min(anzahl_fallback, len(alle_scores))

        # Nach Score sortieren und Top-N nehmen
        sortiert = sorted(alle_scores, key=lambda x: x[1], reverse=True)
        top_n = sortiert[:anzahl_fallback]

        warn(f"Keine Sätze über Schwelle {KERNBOTSCHAFT_SCHWELLE}. "
             f"Fallback: Top-{anzahl_fallback} nach Score als "
             f"Kernbotschaften markiert.")

        kernbotschaften = [
            {
                "satz_id": satz["satz_id"],
                "text": satz["text"],
                "start": satz["start"],
                "end": satz["end"],
                "score": round(score, 4),
            }
            for satz, score in top_n
        ]

    # Absteigend nach Score sortieren
    kernbotschaften.sort(key=lambda k: k["score"], reverse=True)
    return kernbotschaften


# ════════════════════════════════════════════════════════════════════
# SCHRITT 7 — Nebensätze (abgeleitet)
# ════════════════════════════════════════════════════════════════════

def berechne_nebensaetze(satzgrenzen, kernbotschaften):
    """Alle Sätze die KEINE Kernbotschaft sind → Nebensatz."""
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

def _erlaubte_typen_kurze_praesentation(satz_idx, gesamt_saetze):
    """Bestimmt erlaubte Struktur-Typen für sehr kurze Präsentationen.

    Bei < 10 Sätzen ist die 15%-Zeitregel unzuverlässig. Stattdessen:
      - Erste STRUKTUR_KURZ_EINLEITUNG_SAETZE Sätze: Einleitung erlaubt
      - Letzte STRUKTUR_KURZ_SCHLUSS_SAETZE Sätze: Schluss erlaubt
      - Dazwischen: nur Hauptteil/Übergang
    """
    ist_einleitung = satz_idx < STRUKTUR_KURZ_EINLEITUNG_SAETZE
    ist_schluss = satz_idx >= gesamt_saetze - STRUKTUR_KURZ_SCHLUSS_SAETZE
    return ist_einleitung, ist_schluss


def berechne_struktur(satzgrenzen, zero_shot):
    """Erkennt Struktur-Segmente: Einleitung / Hauptteil / Übergang / Schluss.

    Bei >= 10 Sätzen: 15%/85%-Zeitregel für Plausibilitätsprüfung.
    Bei < 10 Sätzen: satz-positions-basierte Regel (robuster für
    kurze Präsentationen).

    Rückgabe: Liste von Struktur-Segmenten.
    """
    if not satzgrenzen:
        return []

    n = len(satzgrenzen)
    ist_kurz = n < STRUKTUR_KURZ_SCHWELLE

    # 1. Gesamtlänge & Zeitgrenzen (nur bei langen Präsentationen relevant)
    if not ist_kurz:
        try:
            gesamt_ende = zeit_zu_sekunden(satzgrenzen[-1]["end"])
            gesamt_start = zeit_zu_sekunden(satzgrenzen[0]["start"])
            dauer = max(gesamt_ende - gesamt_start, 0.001)
            einleitung_grenze = gesamt_start + dauer * EINLEITUNG_ANTEIL
            schluss_grenze = gesamt_ende - dauer * SCHLUSS_ANTEIL
        except ValueError as e:
            warn(f"Zeitgrenzen konnten nicht berechnet werden: {e}. "
                 f"Verwende Satz-Position-Regel.")
            ist_kurz = True

    if ist_kurz and n < STRUKTUR_KURZ_SCHWELLE:
        warn(f"Sehr kurze Präsentation ({n} Sätze). Verwende satz-basierte "
             f"Struktur-Regel statt Zeit-Prozente.")

    # 2. + 3. Pro Satz klassifizieren + plausibilisieren
    satz_typen = []
    gesamt = len(satzgrenzen)

    for i, satz in enumerate(satzgrenzen, start=1):
        print(f"\r    Analysiere Satz {i}/{gesamt}...", end="", flush=True)

        text = satz["text"].strip()

        # Empty-Guard
        if not text:
            satz_typen.append((satz, "Hauptteil"))
            continue

        # Truncation bei sehr langen Sätzen
        text_fuer_modell = _truncate_text(text, ZEROSHOT_MAX_ZEICHEN)

        try:
            resultat = zero_shot(
                text_fuer_modell,
                candidate_labels=STRUKTUR_LABELS,
                multi_label=False,
                hypothesis_template=HYPOTHESE_VORLAGE,
            )
            typ = resultat["labels"][0]  # Label mit höchstem Score
        except Exception as e:
            warn(f"Zero-Shot fehlgeschlagen für Struktur, Satz "
                 f"{satz['satz_id']}: {str(e)[:80]}")
            typ = "Hauptteil"

        # Plausibilitätsprüfung
        if ist_kurz:
            # Satz-Positions-basiert
            ist_einl_erlaubt, ist_schl_erlaubt =\
                _erlaubte_typen_kurze_praesentation(i - 1, n)

            if typ == "Einleitung" and not ist_einl_erlaubt:
                typ = "Hauptteil"
            if typ in ("Schluss", "Zusammenfassung"):
                if not ist_schl_erlaubt:
                    typ = "Hauptteil"
                else:
                    typ = "Schluss"
        else:
            # Zeit-Prozent-basiert (Original-Regel)
            try:
                mitte = (zeit_zu_sekunden(satz["start"])
                         + zeit_zu_sekunden(satz["end"])) / 2.0

                if typ == "Einleitung" and mitte > einleitung_grenze:
                    typ = "Hauptteil"
                if typ in ("Schluss", "Zusammenfassung"):
                    if mitte < schluss_grenze:
                        typ = "Hauptteil"
                    else:
                        typ = "Schluss"
            except ValueError:
                # Bei kaputten Zeitstempeln: Typ ohne Plausi durchlassen
                if typ == "Zusammenfassung":
                    typ = "Schluss"

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
      2. Pro Abschnitt: Sentiment-Scores (positiv/neutral/negativ)
      3. Durchschnitt über alle Abschnitte
      4. Label mappen: positiv → inspirierend, neutral → sachlich,
         negativ → ernst

    Rückgabe: { "label": "inspirierend", "score": 0.87 }
    """
    if not satzgrenzen:
        return {"label": "sachlich", "score": 0.0}

    # 1. Abschnitte bilden
    abschnitte = []
    aktueller = []
    wortzahl = 0

    for satz in satzgrenzen:
        text = satz["text"].strip()
        if not text:
            continue

        n = len(text.split())

        # Einzelner Satz zu groß: eigener Abschnitt
        if n > SENTIMENT_MAX_WOERTER:
            if aktueller:
                abschnitte.append(" ".join(aktueller))
                aktueller = []
                wortzahl = 0
            # Diesen langen Satz für sich alleine (wird vom Modell truncated)
            abschnitte.append(text)
            continue

        if wortzahl + n > SENTIMENT_MAX_WOERTER and aktueller:
            abschnitte.append(" ".join(aktueller))
            aktueller = []
            wortzahl = 0

        aktueller.append(text)
        wortzahl += n

    if aktueller:
        abschnitte.append(" ".join(aktueller))

    if not abschnitte:
        return {"label": "sachlich", "score": 0.0}

    # 2. + 3. Pro Abschnitt klassifizieren, Scores mitteln
    summen = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    erfolgreiche_abschnitte = 0

    for abschnitt in abschnitte:
        if not abschnitt.strip():
            continue

        try:
            resultat = sentiment(abschnitt, truncation=True, max_length=512)
        except Exception as e:
            warn(f"Sentiment-Modell fehlgeschlagen für Abschnitt: "
                 f"{str(e)[:80]}")
            continue

        # Pipeline mit top_k=None gibt Liste aller Labels zurück.
        # Format kann [[{...}]] oder [{...}] sein — beide abfangen.
        if not resultat:
            continue

        if isinstance(resultat[0], list):
            eintraege = resultat[0]
        else:
            eintraege = resultat

        for eintrag in eintraege:
            # Robuste Label-Erkennung: manche Modelle geben deutsche
            # oder gemischte Labels zurück
            label = eintrag.get("label", "").lower()
            # Deutsche/englische Varianten mappen
            if label in ("positive", "positiv"):
                summen["positive"] += eintrag.get("score", 0.0)
            elif label in ("neutral",):
                summen["neutral"] += eintrag.get("score", 0.0)
            elif label in ("negative", "negativ"):
                summen["negative"] += eintrag.get("score", 0.0)

        erfolgreiche_abschnitte += 1

    if erfolgreiche_abschnitte == 0:
        warn("Kein Abschnitt konnte klassifiziert werden. Fallback: sachlich.")
        return {"label": "sachlich", "score": 0.0}

    mittel = {
        label: wert / erfolgreiche_abschnitte
        for label, wert in summen.items()
    }

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
      - 'wir', 'uns', 'euch', 'du', 'dir', 'dich' + Possessive
        → immer Ansprache (case-insensitive)
      - 'Sie', 'Ihnen', 'Ihr' + Possessive (grossgeschrieben)
        → nur Ansprache wenn NICHT am Satzanfang

    'ihr' (klein) wird bewusst NICHT erkannt (Mehrdeutigkeit,
    siehe Konfig-Kommentar oben).
    """
    publikumsbezug = []

    # Regex erlaubt Apostroph-Kontraktionen (wir's, du's, geht's...)
    # sowie normale Wörter mit deutschen Umlauten und ß
    token_regex = re.compile(r"[\wäöüÄÖÜß]+(?:'[\wäöüÄÖÜß]+)?")

    for satz in satzgrenzen:
        text = satz["text"]
        if not text:
            continue

        tokens = token_regex.findall(text)
        gefunden = False

        for pos, token in enumerate(tokens):
            # Bei Apostroph-Kontraktion: nur Hauptteil prüfen
            # ("wir's" → "wir")
            hauptteil = token.split("'")[0]

            # Immer-Ansprache (case-insensitive)
            if hauptteil.lower() in ANSPRACHE_IMMER:
                gefunden = True
                break

            # Formelle Ansprache: exakt grossgeschrieben + nicht Satzanfang
            if hauptteil in ANSPRACHE_FORMELL and pos > 0:
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
    try:
        os.makedirs(ordner, exist_ok=True)
    except OSError as e:
        print(f"  ✗ Ordner {ordner} konnte nicht erstellt werden: {e}")
        sys.exit(1)

    pfad = os.path.join(ordner, "inhalt_analyse_output.json")

    try:
        with open(pfad, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ JSON konnte nicht geschrieben werden: {e}")
        sys.exit(1)

    return pfad


def schreibe_bericht(output, root, transkript_name):
    """Schreibt den lesbaren TXT-Bericht nach reports/."""
    ordner = os.path.join(root, "reports")
    try:
        os.makedirs(ordner, exist_ok=True)
    except OSError as e:
        print(f"  ✗ Ordner {ordner} konnte nicht erstellt werden: {e}")
        sys.exit(1)

    basis = os.path.splitext(os.path.basename(transkript_name))[0]
    basis = sicherer_dateiname(basis)
    pfad = os.path.join(ordner, f"inhalt_analyse_bericht_{basis}.txt")

    sg = output["satzgrenzen"]
    kb = output["kernbotschaften"]
    ns = output["nebensaetze"]
    st = output["struktur"]
    rm = output["rhetorische_momente"]
    ss = output["satzstruktur"]
    et = output["emotionaler_ton"]
    pb = output["publikumsbezug"]
    warnungen = output.get("meta", {}).get("warnungen", [])

    z = []
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
        try:
            dauer_s = zeit_zu_sekunden(sg[-1]["end"])
            minuten = int(dauer_s // 60)
            sekunden = int(dauer_s % 60)
            z.append(f"  Dauer:       {minuten} Min {sekunden} Sek")
        except ValueError:
            z.append(f"  Dauer:       (nicht berechenbar)")
    z.append(f"  Sätze:       {len(sg)}")

    # ── Warnungen anzeigen (wenn vorhanden) ──
    if warnungen:
        titel(f"⚠ WARNUNGEN ({len(warnungen)})")
        for w in warnungen[:20]:
            z.append(f"  • {w}")
        if len(warnungen) > 20:
            z.append(f"  ... und {len(warnungen) - 20} weitere (siehe JSON)")

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
    z.append("  nebensätze, struktur, rhetorische_momente,")
    z.append("  satzstruktur, emotionaler_ton, publikumsbezug")
    z.append("═" * breite)

    try:
        with open(pfad, "w", encoding="utf-8") as f:
            f.write("\n".join(z))
    except OSError as e:
        print(f"  ✗ Bericht konnte nicht geschrieben werden: {e}")
        sys.exit(1)

    return pfad


# ════════════════════════════════════════════════════════════════════
# MAIN — Orchestrierung
# ════════════════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 60)
    print("  präsentation_ai — INHALTSANALYSE")
    print("═" * 60)

    # ── Schritt 1: Transkript einlesen ──
    print("\n[1/9] Transkript einlesen...")
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

    if not satzgrenzen:
        print("  ✗ Keine Sätze erkannt — Abbruch.")
        sys.exit(1)

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
            "warnungen": WARNUNGEN,
            "script_version": "v2-gehaertet",
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
    if WARNUNGEN:
        print(f"    Mit {len(WARNUNGEN)} Warnung(en) — Details im Bericht")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
