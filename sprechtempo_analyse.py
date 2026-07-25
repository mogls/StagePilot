"""sprechtempo_analyse.py
==============================================================================
Analyse des Sprechtempos einer Präsentation.

Misst die Sprechgeschwindigkeit in Silben pro Sekunde und bewertet:
  1. Gesamttempo         (Gewichtung 40%)  Optimum: 4.5-5.5 Silben/Sek
  2. Variation           (Gewichtung 30%)  Optimum: CV 0.20-0.35
  3. Kernbotschaften     (Gewichtung 30%)  Optimum: >= 10% Verlangsamung

Inputs:
  - Transkript (.txt) im Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm
  - inhalt_analyse_output.json aus zwischen_output/

Outputs:
  - TXT-Report in reports/sprechtempo/
  - JSON-Intermediate in zwischen_output/sprechtempo_analyse_output.json"""

import re
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import filedialog


# ============================================================================
# KONFIGURATION
# ============================================================================

GEWICHTUNG = {
    'tempo': 0.40,
    'variation': 0.30,
    'kernbotschaften': 0.30,
}

# Tempo-Schwellenwerte in Silben pro Sekunde
TEMPO_OPTIMUM_MIN = 4.5
TEMPO_OPTIMUM_MAX = 5.5
TEMPO_LEICHT_LANGSAM = 3.5
TEMPO_LEICHT_SCHNELL = 6.5
TEMPO_EXTREM_LANGSAM = 2.5
TEMPO_EXTREM_SCHNELL = 7.5

# Variationskoeffizient (Standardabweichung / Mittelwert)
CV_OPTIMUM_MIN = 0.20
CV_OPTIMUM_MAX = 0.35
CV_LEICHT_MIN = 0.10
CV_LEICHT_MAX = 0.50

# Kernbotschaften-Verlangsamung (als Anteil, 0.10 = 10%)
VERLANGSAMUNG_GUT = 0.10

# Mindestwerte für valide Tempo-Messung pro Satz
MIN_SATZ_DAUER = 1.0    # Sekunden
MIN_SATZ_SILBEN = 3

# Pfade (relativ zum Projekt-Root, angenommen script liegt in analyse/)
PROJEKT_ROOT = Path(__file__).resolve().parent.parent
JSON_INPUT_PFAD = PROJEKT_ROOT / "zwischen_output" / "inhalt_analyse_output.json"
JSON_OUTPUT_PFAD = PROJEKT_ROOT / "zwischen_output" / "sprechtempo_analyse_output.json"
REPORT_ORDNER = PROJEKT_ROOT / "reports" / "sprechtempo"


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def zeit_zu_sekunden(zeitstring):
    """Konvertiert HH:MM:SS.mmm zu Sekunden als float."""
    teile = zeitstring.strip().split(':')
    if len(teile) != 3:
        raise ValueError(f"Ungültiges Zeitformat: {zeitstring}")
    return int(teile[0]) * 3600 + int(teile[1]) * 60 + float(teile[2])


def hole_ende(eintrag):
    """Feldname 'end' bevorzugt, 'ende' als Fallback."""
    if 'end' in eintrag:
        return eintrag['end']
    if 'ende' in eintrag:
        return eintrag['ende']
    raise KeyError("Weder 'end' noch 'ende' im Eintrag")


# Ausnahmenliste für unregelmäßige Wörter (v2-Fix).
# Diese Wörter widersprechen der Diphthong- und Doppelvokal-Regel und
# werden deshalb direkt gemappt.
SILBEN_AUSNAHMEN = {
    'idee': 3, 'ideen': 3,
    'museum': 4, 'museums': 4, 'museen': 3,
    'familie': 4, 'familien': 4,
    'aktuell': 3, 'aktuelle': 4, 'aktuellen': 4,
    'individuell': 5, 'individuelle': 5,
    'situation': 4, 'situationen': 5,
    'nation': 3, 'nationen': 4, 'national': 4,
    'region': 3, 'regionen': 4, 'regional': 4,
    'union': 3, 'unionen': 4,
    'million': 3, 'millionen': 4,
    'milliarde': 4, 'milliarden': 4,
    'kreation': 4, 'kreationen': 5,
    'aktion': 3, 'aktionen': 4,
    'real': 2, 'reale': 3, 'realen': 3, 'realitaet': 4,
    'kreativitaet': 5, 'kreativität': 5,
    # -tion Wörter (Nation-Regel: -tion = 1 Silbe)
    'präsentation': 4, 'präsentationen': 5,
    'praesentation': 4, 'praesentationen': 5,
    'produktion': 3, 'produktionen': 4,
    'funktion': 3, 'funktionen': 4,
    'position': 3, 'positionen': 4,
    'diskussion': 3, 'diskussionen': 4,
    'motivation': 4, 'motivationen': 5,
    'information': 4, 'informationen': 5,
    'organisation': 5, 'organisationen': 6,
    # Fremdwoerter mit y am Ende (vokalisches y)
    'baby': 2, 'babys': 2, 'party': 2, 'partys': 2,
    'story': 2, 'stories': 2, 'hobby': 2, 'hobbys': 2,
    'city': 2, 'jury': 2, 'company': 3,
}


def zaehle_silben(wort):
    """
    Zählt Silben in einem deutschen Wort (v2-konform).

    Regeln:
      - Ausnahmenliste zuerst prüfen (Idee, Museum, Familie, ...)
      - Vokale/Umlaute zählen: a, e, i, o, u, ä, ö, ü
      - y-Sonderregel: y nur als Vokal wenn zwischen Konsonanten
        (z.B. 'System', 'Rhythmus'). Am Wortanfang/-ende oder vor Vokal
        ist y Konsonant (Yoga, Yacht).
      - Diphthonge als eine Silbe: ai, ei, au, äu, eu, ie, ui, ay, ey
      - Aufeinanderfolgende gleiche Vokale (See, Boot) = eine Silbe
      - Minimum: 1 Silbe pro Wort
    """
    wort_clean = re.sub(r'[^a-z\u00e4\u00f6\u00fc\u00dfy]', '', wort.lower())
    if not wort_clean:
        return 0

    # 1. Ausnahmenliste
    if wort_clean in SILBEN_AUSNAHMEN:
        return SILBEN_AUSNAHMEN[wort_clean]

    diphthonge = {'ai', 'ei', 'au', '\u00e4u', 'eu', 'ie', 'ui', 'ay', 'ey'}
    vokale_ohne_y = set('aeiou\u00e4\u00f6\u00fc')

    def ist_vokal(pos):
        """y ist nur Vokal wenn zwischen zwei Konsonanten steht."""
        c = wort_clean[pos]
        if c in vokale_ohne_y:
            return True
        if c == 'y':
            # y ist Konsonant am Anfang, am Ende oder neben einem Vokal
            hat_vokal_davor = pos > 0 and wort_clean[pos - 1] in vokale_ohne_y
            hat_vokal_danach = (pos + 1 < len(wort_clean)
                                and wort_clean[pos + 1] in vokale_ohne_y)
            am_rand = pos == 0 or pos == len(wort_clean) - 1
            if am_rand or hat_vokal_davor or hat_vokal_danach:
                return False
            return True
        return False

    silben = 0
    i = 0
    while i < len(wort_clean):
        if ist_vokal(i):
            # Diphthong-Prüfung (nutzt echte Zeichen — y ist hier nie Teil
            # eines Standard-Diphthongs ausser in ay/ey)
            zwei_zeichen = wort_clean[i:i + 2]
            if i + 1 < len(wort_clean) and zwei_zeichen in diphthonge:
                silben += 1
                i += 2
            else:
                silben += 1
                # Gleiche Doppelvokale überspringen (See, Boot)
                while i + 1 < len(wort_clean) and wort_clean[i + 1] == wort_clean[i]:
                    i += 1
                i += 1
        else:
            i += 1

    return max(1, silben)


def silben_zu_wpm(silben_pro_sek):
    """Rechnet Silben/Sek zu Wörtern/Min um (Deutsch: ~2.5 Silben pro Wort)."""
    return round((silben_pro_sek / 2.5) * 60, 1)


# ============================================================================
# EINLESEN
# ============================================================================

def lade_transkript(pfad):
    """
    Liest Transkript im Format: Wort HH:MM:SS.mmm HH:MM:SS.mmm

    Gibt Liste zurück: [{'wort', 'start_s', 'ende_s', 'silben'}, ...]
    """
    muster = re.compile(
        r'^(\S+)\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+(\d{2}:\d{2}:\d{2}\.\d+)\s*$'
    )

    def parse(datei):
        eintraege = []
        for zeile in datei:
            zeile = zeile.strip()
            if not zeile:
                continue
            treffer = muster.match(zeile)
            if not treffer:
                continue
            eintraege.append({
                'wort': treffer.group(1),
                'start_s': zeit_zu_sekunden(treffer.group(2)),
                'ende_s': zeit_zu_sekunden(treffer.group(3)),
                'silben': zaehle_silben(treffer.group(1)),
            })
        return eintraege

    try:
        with open(pfad, 'r', encoding='utf-8') as f:
            woerter = parse(f)
    except UnicodeDecodeError:
        # Fallback für Windows-Encodings
        with open(pfad, 'r', encoding='latin-1') as f:
            woerter = parse(f)

    if not woerter:
        raise ValueError("Keine gültigen Zeilen im Transkript gefunden.")

    return woerter


def lade_inhalt_json(pfad):
    """Liest die JSON-Ausgabe von inhalt_analyse.py ein."""
    with open(pfad, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# TEMPO-BERECHNUNG
# ============================================================================

def berechne_tempo_pro_satz(woerter, satzgrenzen):
    """
    Ordnet Wörter den Sätzen zu und berechnet Tempo pro Satz.

    Tempo = Netto-Artikulationsrate = Silben / Summe der Wort-Dauern
    (Pausen zählen nicht zur Sprechzeit)
    """
    ergebnis = []
    for satz in satzgrenzen:
        satz_start = zeit_zu_sekunden(satz['start'])
        satz_ende = zeit_zu_sekunden(hole_ende(satz))

        # Wort gehört zum Satz, wenn sein Start im Satz-Zeitraum liegt
        satz_woerter = [
            w for w in woerter
            if w['start_s'] >= satz_start - 0.05
            and w['start_s'] <= satz_ende + 0.05
        ]

        if not satz_woerter:
            continue

        silben_summe = sum(w['silben'] for w in satz_woerter)
        sprechzeit = sum(w['ende_s'] - w['start_s'] for w in satz_woerter)
        gesamt_dauer = satz_woerter[-1]['ende_s'] - satz_woerter[0]['start_s']

        if sprechzeit <= 0:
            continue

        tempo = silben_summe / sprechzeit

        ergebnis.append({
            'satz_id': satz['satz_id'],
            'text': satz.get('text', ''),
            'start_s': satz_start,
            'ende_s': satz_ende,
            'silben': silben_summe,
            'sprechzeit_s': round(sprechzeit, 3),
            'gesamt_dauer_s': round(gesamt_dauer, 3),
            'tempo': tempo,
            'valid': gesamt_dauer >= MIN_SATZ_DAUER and silben_summe >= MIN_SATZ_SILBEN,
        })

    return ergebnis


def berechne_gesamttempo(woerter):
    """Netto-Sprechrate über alle Wörter (Artikulationsrate)."""
    silben = sum(w['silben'] for w in woerter)
    sprechzeit = sum(w['ende_s'] - w['start_s'] for w in woerter)
    if sprechzeit <= 0:
        return 0.0
    return silben / sprechzeit


# ============================================================================
# BEWERTUNG (jeweils label + punkte)
# ============================================================================

def bewerte_gesamttempo(tempo):
    if TEMPO_OPTIMUM_MIN <= tempo <= TEMPO_OPTIMUM_MAX:
        return 'optimal', 100
    if TEMPO_LEICHT_LANGSAM <= tempo < TEMPO_OPTIMUM_MIN:
        return 'etwas_langsam', 80
    if TEMPO_OPTIMUM_MAX < tempo <= TEMPO_LEICHT_SCHNELL:
        return 'etwas_schnell', 80
    if TEMPO_EXTREM_LANGSAM <= tempo < TEMPO_LEICHT_LANGSAM:
        return 'zu_langsam', 50
    if TEMPO_LEICHT_SCHNELL < tempo <= TEMPO_EXTREM_SCHNELL:
        return 'zu_schnell', 50
    return 'extrem', 20


def bewerte_variation(cv):
    if CV_OPTIMUM_MIN <= cv <= CV_OPTIMUM_MAX:
        return 'optimal', 100
    if CV_LEICHT_MIN <= cv < CV_OPTIMUM_MIN:
        return 'leicht', 75
    if CV_OPTIMUM_MAX < cv <= CV_LEICHT_MAX:
        return 'stark', 75
    if cv < CV_LEICHT_MIN:
        return 'monoton', 40
    return 'chaotisch', 40


def bewerte_kernbotschaften(verlangsamung):
    """verlangsamung = (tempo_neben - tempo_kern) / tempo_neben"""
    if verlangsamung >= VERLANGSAMUNG_GUT:
        return 'gut', 100
    if verlangsamung >= 0:
        return 'neutral', 70
    return 'negativ', 30


def berechne_gesamtscore(punkte_tempo, punkte_variation, punkte_kern):
    return round(
        GEWICHTUNG['tempo'] * punkte_tempo
        + GEWICHTUNG['variation'] * punkte_variation
        + GEWICHTUNG['kernbotschaften'] * punkte_kern,
        1,
    )


# ============================================================================
# FEEDBACK-TEXTE (Hochdeutsch, 5 Levels pro Dimension)
# ============================================================================

TEMPO_FEEDBACK = {
    'optimal': (
        "Ihr Sprechtempo liegt im idealen Bereich für Präsentationen "
        "(4.5-5.5 Silben/Sek). Das Publikum kann Ihren Ausfuehrungen muehelos folgen."
    ),
    'etwas_langsam': (
        "Ihr Tempo ist ruhig und gut verständlich, könnte aber an einigen "
        "Stellen etwas mehr Energie vertragen. Bei komplexen Inhalten ist dieses "
        "Tempo dennoch angemessen."
    ),
    'etwas_schnell': (
        "Ihr Tempo ist zuegig und noch verständlich. Achten Sie darauf, "
        "wichtige Stellen bewusst zu verlangsamen, damit die Kernaussagen ankommen."
    ),
    'zu_langsam': (
        "Ihr Tempo ist deutlich zu langsam. Das Publikum verliert bei diesem "
        "Tempo schnell die Aufmerksamkeit. Steigern Sie die Sprechgeschwindigkeit, "
        "um lebendiger zu wirken."
    ),
    'zu_schnell': (
        "Ihr Tempo ist zu schnell. Studien zeigen, dass die Verständlichkeit "
        "ab 180 Wörtern pro Minute um bis zu 22 % sinkt. Verlangsamen Sie "
        "bewusst und setzen Sie mehr Pausen."
    ),
    'extrem': (
        "Ihr Tempo liegt weit ausserhalb des Referenzbereichs. Eine deutliche "
        "Anpassung ist notwendig, damit das Publikum Ihren Ausführungen folgen kann."
    ),
}

VARIATION_FEEDBACK = {
    'optimal': (
        "Ihr Tempo variiert in einem angenehmen Bereich. Diese Dynamik hält "
        "die Aufmerksamkeit des Publikums aufrecht."
    ),
    'leicht': (
        "Ihre Tempo-Variation ist vorhanden, könnte aber ausgepraegter sein. "
        "Setzen Sie bewusst schnellere und langsamere Passagen ein."
    ),
    'stark': (
        "Ihre Tempo-Variation ist sehr ausgepraegt. Achten Sie darauf, dass "
        "die Wechsel dem Inhalt dienen und nicht zufällig wirken."
    ),
    'monoton': (
        "Ihr Tempo ist zu konstant. Monotonie wirkt einschlaefernd. Bauen Sie "
        "bewusst Tempo-Wechsel ein, insbesondere bei Kernbotschaften und Übergaengen."
    ),
    'chaotisch': (
        "Ihr Tempo wechselt sehr sprunghaft. Das kann unruhig oder nervoes "
        "wirken. Streben Sie einen ruhigeren Grundrhythmus mit gezielten "
        "Variationen an."
    ),
}

KERNBOTSCHAFT_FEEDBACK = {
    'gut': (
        "Sie verlangsamen Ihr Tempo bei Kernbotschaften spürbar. Das betont "
        "diese Aussagen effektiv und lässt sie beim Publikum ankommen."
    ),
    'neutral': (
        "Ihre Kernbotschaften werden im gleichen Tempo wie der Rest gesprochen. "
        "Eine bewusste Verlangsamung um mindestens 10 % würde sie deutlich "
        "stärker hervorheben."
    ),
    'negativ': (
        "Sie sprechen bei Kernbotschaften schneller als im Durchschnitt. Das "
        "schwaecht ihre Wirkung erheblich. Verlangsamen Sie bewusst bei "
        "wichtigen Aussagen."
    ),
}


# ============================================================================
# STRUKTUR-CHECK (nur informativ, geht nicht in den Score)
# ============================================================================

def analysiere_struktur(saetze_mit_tempo, struktur_segmente, gesamttempo):
    """Prüft Tempo pro Struktur-Segment und ob die Erwartung erfüllt ist."""
    ergebnis = []
    for segment in struktur_segmente:
        seg_start = zeit_zu_sekunden(segment['start'])
        seg_ende = zeit_zu_sekunden(hole_ende(segment))

        segment_saetze = [
            s for s in saetze_mit_tempo
            if s['valid']
            and s['start_s'] >= seg_start - 0.5
            and s['ende_s'] <= seg_ende + 0.5
        ]
        if not segment_saetze:
            continue

        seg_tempo = statistics.mean(s['tempo'] for s in segment_saetze)
        typ = segment['typ']
        erwartet_langsamer = typ in ('Einleitung', 'Schluss', 'Uebergang', 'Zusammenfassung')
        abweichung = (seg_tempo - gesamttempo) / gesamttempo if gesamttempo > 0 else 0

        if erwartet_langsamer:
            erfuellt = seg_tempo <= gesamttempo * 1.05  # 5% Toleranz
        else:
            erfuellt = True

        ergebnis.append({
            'typ': typ,
            'start': segment['start'],
            'end': hole_ende(segment),
            'tempo': round(seg_tempo, 2),
            'abweichung_prozent': round(abweichung * 100, 1),
            'erwartung_erfuellt': erfuellt,
            'erwartet_langsamer': erwartet_langsamer,
        })
    return ergebnis


# ============================================================================
# OUTPUT
# ============================================================================

def erstelle_txt_report(analyse, pfad):
    """Erstellt den lesbaren TXT-Report."""
    z = []
    z.append("=" * 70)
    z.append("SPRECHTEMPO-ANALYSE")
    z.append("=" * 70)
    z.append(f"Erstellt:    {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    z.append(f"Transkript:  {analyse['transkript_datei']}")
    z.append("")
    z.append(f"GESAMTSCORE: {analyse['gesamtscore']}/100")
    z.append("")

    # --- Gesamttempo ---
    z.append("-" * 70)
    z.append("1. GESAMTTEMPO (Gewichtung 40 %)")
    z.append("-" * 70)
    z.append(f"Silben pro Sekunde:  {analyse['gesamttempo']:.2f}")
    z.append(f"Wörter pro Minute:  ~{analyse['gesamttempo_wpm']}")
    z.append(f"Teilscore:           {analyse['punkte_tempo']}/100")
    z.append("")
    z.append("Bewertung:")
    z.append(TEMPO_FEEDBACK[analyse['bewertung_tempo']])
    z.append("")

    # --- Variation ---
    z.append("-" * 70)
    z.append("2. VARIATION (Gewichtung 30 %)")
    z.append("-" * 70)
    z.append(f"Standardabweichung:      {analyse['variation_std']}")
    z.append(f"Variationskoeffizient:   {analyse['variation_cv']}")
    z.append(f"Teilscore:               {analyse['punkte_variation']}/100")
    z.append("")
    z.append("Bewertung:")
    if analyse['bewertung_variation'] in VARIATION_FEEDBACK:
        z.append(VARIATION_FEEDBACK[analyse['bewertung_variation']])
    else:
        z.append("Zu wenige valide Sätze für eine Variations-Bewertung.")
    z.append("")

    # --- Kernbotschaften ---
    z.append("-" * 70)
    z.append("3. KERNBOTSCHAFTEN (Gewichtung 30 %)")
    z.append("-" * 70)
    if analyse['kern_daten_vorhanden']:
        z.append(f"Tempo Kernbotschaften:   {analyse['tempo_kern']:.2f} Silben/Sek")
        z.append(f"Tempo Nebensätze:       {analyse['tempo_neben']:.2f} Silben/Sek")
        z.append(f"Verlangsamung:           {analyse['verlangsamung_prozent']} %")
        z.append(f"Teilscore:               {analyse['punkte_kernbotschaften']}/100")
        z.append("")
        z.append("Bewertung:")
        z.append(KERNBOTSCHAFT_FEEDBACK[analyse['bewertung_kernbotschaften']])
    else:
        z.append("Keine Kernbotschaften oder Nebensätze im Inhalt gefunden.")
        z.append("Dimension konnte nicht bewertet werden.")
    z.append("")

    # --- Struktur (informativ) ---
    if analyse.get('struktur_analyse'):
        z.append("-" * 70)
        z.append("STRUKTUR-ANALYSE (informativ, nicht im Score)")
        z.append("-" * 70)
        for seg in analyse['struktur_analyse']:
            marker = "OK " if seg['erwartung_erfuellt'] else "!! "
            erwartung = (
                "langsamer erwartet"
                if seg['erwartet_langsamer']
                else "keine Erwartung"
            )
            z.append(
                f"{marker}{seg['typ']:20s}  {seg['tempo']:.2f} Silben/Sek  "
                f"({erwartung}, Abweichung: {seg['abweichung_prozent']:+.1f} %)"
            )
        z.append("")

    # --- Referenzwerte ---
    z.append("=" * 70)
    z.append("REFERENZWERTE")
    z.append("=" * 70)
    z.append("Optimum Sprechtempo:            4.5 - 5.5 Silben/Sek  (90 - 120 WPM)")
    z.append("Optimum Variationskoeffizient:  0.20 - 0.35")
    z.append("Kernbotschaft-Verlangsamung:    >= 10 %")
    z.append("")
    z.append("Quellen: Kognitive Verarbeitungsforschung, University of Michigan,")
    z.append("Journal of Nonverbal Behavior 2024 (N=3958), Duden/Wikipedia.")

    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, 'w', encoding='utf-8') as f:
        f.write('\n'.join(z))


def erstelle_json_output(analyse, saetze_mit_tempo, pfad):
    """Erstellt die JSON-Ausgabe für nachgelagerte Scripts / gesamtscore.py."""
    output = {
        'gesamttempo_silben_sek': round(analyse['gesamttempo'], 2),
        'gesamttempo_wpm': analyse['gesamttempo_wpm'],
        'gesamtscore': analyse['gesamtscore'],
        'teilscores': {
            'tempo': analyse['punkte_tempo'],
            'variation': analyse['punkte_variation'],
            'kernbotschaften': analyse['punkte_kernbotschaften'],
        },
        'bewertungen': {
            'tempo': analyse['bewertung_tempo'],
            'variation': analyse['bewertung_variation'],
            'kernbotschaften': analyse['bewertung_kernbotschaften'],
        },
        'variation': {
            'standardabweichung': analyse['variation_std'],
            'variationskoeffizient': analyse['variation_cv'],
        },
        'kernbotschaften': {
            'tempo_kern': analyse.get('tempo_kern'),
            'tempo_neben': analyse.get('tempo_neben'),
            'verlangsamung_prozent': analyse.get('verlangsamung_prozent'),
        },
        'struktur': analyse.get('struktur_analyse', []),
        'saetze': [
            {
                'satz_id': s['satz_id'],
                'tempo': round(s['tempo'], 2),
                'silben': s['silben'],
                'sprechzeit_s': s['sprechzeit_s'],
                'valid': s['valid'],
            }
            for s in saetze_mit_tempo
        ],
    }
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# ============================================================================
# FILE-PICKER
# ============================================================================

def waehle_datei(titel, dateitypen):
    root = tk.Tk()
    root.withdraw()
    pfad = filedialog.askopenfilename(title=titel, filetypes=dateitypen)
    root.destroy()
    return pfad


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("SPRECHTEMPO-ANALYSE")
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # Schritt 1: Transkript wählen
    # -------------------------------------------------------------
    print("Schritt 1/9: Transkript waehlen...")
    if len(sys.argv) > 1:
        transkript_pfad = Path(sys.argv[1])
    else:
        pfad_str = waehle_datei(
            "Transkript waehlen (.txt)",
            [("Textdateien", "*.txt"), ("Alle Dateien", "*.*")],
        )
        if not pfad_str:
            print("  [x] Kein Transkript gewaehlt. Abbruch.")
            return
        transkript_pfad = Path(pfad_str)
    if not transkript_pfad.exists():
        print(f"  [x] Datei nicht gefunden: {transkript_pfad}")
        return
    print(f"  [OK] Transkript: {transkript_pfad.name}")

    # -------------------------------------------------------------
    # Schritt 2: Transkript einlesen + Silben zählen
    # -------------------------------------------------------------
    print("Schritt 2/9: Transkript einlesen und Silben zaehlen...")
    try:
        woerter = lade_transkript(transkript_pfad)
    except Exception as e:
        print(f"  [x] Fehler beim Einlesen: {e}")
        return
    silben_gesamt = sum(w['silben'] for w in woerter)
    print(f"  [OK] {len(woerter)} Wörter, {silben_gesamt} Silben insgesamt")

    # -------------------------------------------------------------
    # Schritt 3: Inhalt-Analyse JSON laden
    # -------------------------------------------------------------
    print("Schritt 3/9: Inhalt-Analyse JSON laden...")
    json_pfad = JSON_INPUT_PFAD
    if not json_pfad.exists():
        print(f"  [!] Nicht gefunden am Standardort: {json_pfad}")
        print("      Bitte inhalt_analyse_output.json manuell waehlen...")
        pfad_str = waehle_datei(
            "inhalt_analyse_output.json waehlen",
            [("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")],
        )
        if not pfad_str:
            print("  [x] Keine JSON gewaehlt. Abbruch.")
            return
        json_pfad = Path(pfad_str)
    try:
        inhalt = lade_inhalt_json(json_pfad)
    except Exception as e:
        print(f"  [x] Fehler beim Laden: {e}")
        return
    n_saetze = len(inhalt.get('satzgrenzen', []))
    n_kern = len(inhalt.get('kernbotschaften', []))
    n_struktur = len(inhalt.get('struktur', []))
    print(f"  [OK] {n_saetze} Sätze, {n_kern} Kernbotschaften, {n_struktur} Struktur-Segmente")

    # -------------------------------------------------------------
    # Schritt 4: Tempo pro Satz berechnen
    # -------------------------------------------------------------
    print("Schritt 4/9: Tempo pro Satz berechnen...")
    saetze_mit_tempo = berechne_tempo_pro_satz(woerter, inhalt.get('satzgrenzen', []))
    valide_saetze = [s for s in saetze_mit_tempo if s['valid']]
    print(f"  [OK] {len(saetze_mit_tempo)} Sätze analysiert ({len(valide_saetze)} valide)")

    # -------------------------------------------------------------
    # Schritt 5: Gesamttempo berechnen
    # -------------------------------------------------------------
    print("Schritt 5/9: Gesamttempo berechnen...")
    gesamttempo = berechne_gesamttempo(woerter)
    gesamttempo_wpm = silben_zu_wpm(gesamttempo)
    print(f"  [OK] {gesamttempo:.2f} Silben/Sek (~{gesamttempo_wpm} WPM)")

    # -------------------------------------------------------------
    # Schritt 6: Variation berechnen
    # -------------------------------------------------------------
    print("Schritt 6/9: Variation berechnen...")
    if len(valide_saetze) >= 2:
        tempi = [s['tempo'] for s in valide_saetze]
        std = statistics.stdev(tempi)
        mittelwert = statistics.mean(tempi)
        cv = std / mittelwert if mittelwert > 0 else 0
        variation_bewertbar = True
    else:
        std = 0.0
        cv = 0.0
        variation_bewertbar = False
    print(f"  [OK] Std: {std:.3f}, CV: {cv:.3f}")

    # -------------------------------------------------------------
    # Schritt 7: Kernbotschaften analysieren
    # -------------------------------------------------------------
    print("Schritt 7/9: Kernbotschaften analysieren...")
    kern_ids = {k['satz_id'] for k in inhalt.get('kernbotschaften', [])}
    kern_saetze = [s for s in valide_saetze if s['satz_id'] in kern_ids]
    neben_saetze = [s for s in valide_saetze if s['satz_id'] not in kern_ids]
    kern_daten_vorhanden = len(kern_saetze) > 0 and len(neben_saetze) > 0
    if kern_daten_vorhanden:
        tempo_kern = statistics.mean(s['tempo'] for s in kern_saetze)
        tempo_neben = statistics.mean(s['tempo'] for s in neben_saetze)
        verlangsamung = (tempo_neben - tempo_kern) / tempo_neben if tempo_neben > 0 else 0
    else:
        tempo_kern = None
        tempo_neben = None
        verlangsamung = 0
    print(f"  [OK] {len(kern_saetze)} Kernbotschaften, {len(neben_saetze)} Nebensätze")

    # -------------------------------------------------------------
    # Schritt 8: Bewertung und Score
    # -------------------------------------------------------------
    print("Schritt 8/9: Bewertung und Score berechnen...")
    bewertung_tempo, punkte_tempo = bewerte_gesamttempo(gesamttempo)

    if variation_bewertbar:
        bewertung_variation, punkte_variation = bewerte_variation(cv)
    else:
        bewertung_variation, punkte_variation = 'nicht_bewertbar', 50

    if kern_daten_vorhanden:
        bewertung_kern, punkte_kern = bewerte_kernbotschaften(verlangsamung)
    else:
        bewertung_kern, punkte_kern = 'nicht_bewertbar', 50

    gesamtscore = berechne_gesamtscore(punkte_tempo, punkte_variation, punkte_kern)
    struktur_analyse = analysiere_struktur(
        saetze_mit_tempo, inhalt.get('struktur', []), gesamttempo
    )
    print(f"  [OK] Gesamtscore: {gesamtscore}/100")

    # Analyse-Objekt zusammenbauen
    analyse = {
        'transkript_datei': transkript_pfad.name,
        'gesamttempo': gesamttempo,
        'gesamttempo_wpm': gesamttempo_wpm,
        'gesamtscore': gesamtscore,
        'punkte_tempo': punkte_tempo,
        'punkte_variation': punkte_variation,
        'punkte_kernbotschaften': punkte_kern,
        'bewertung_tempo': bewertung_tempo,
        'bewertung_variation': bewertung_variation,
        'bewertung_kernbotschaften': bewertung_kern,
        'variation_std': round(std, 3),
        'variation_cv': round(cv, 3),
        'kern_daten_vorhanden': kern_daten_vorhanden,
        'tempo_kern': round(tempo_kern, 2) if tempo_kern is not None else None,
        'tempo_neben': round(tempo_neben, 2) if tempo_neben is not None else None,
        'verlangsamung_prozent': round(verlangsamung * 100, 1) if kern_daten_vorhanden else None,
        'struktur_analyse': struktur_analyse,
    }

    # -------------------------------------------------------------
    # Schritt 9: Output schreiben
    # -------------------------------------------------------------
    print("Schritt 9/9: Output schreiben...")
    zeitstempel = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_pfad = REPORT_ORDNER / f"sprechtempo_{transkript_pfad.stem}_{zeitstempel}.txt"
    erstelle_txt_report(analyse, report_pfad)
    print(f"  [OK] TXT-Report:       {report_pfad}")
    erstelle_json_output(analyse, saetze_mit_tempo, JSON_OUTPUT_PFAD)
    print(f"  [OK] JSON-Intermediate: {JSON_OUTPUT_PFAD}")

    print()
    print("=" * 70)
    print(f"ANALYSE ABGESCHLOSSEN - Gesamtscore: {gesamtscore}/100")
    print("=" * 70)


if __name__ == "__main__":
    main()
