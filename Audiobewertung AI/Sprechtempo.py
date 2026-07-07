#!/usr/bin/env python3
"""
Sprechtempo-Messung fuer Praesentationsbewertungs-App
=====================================================
Nutzt OpenAI Whisper mit word_timestamps=True zur Berechnung
globaler und segmentweiser WPM (Words Per Minute).

Autor  : KI-generiert (produktionsreif)
Python : >= 3.9
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging-Konfiguration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sprechtempoanalyse")


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
SEGMENT_DAUER_SEK = 30          # Fenstergroesse fuer segmentweise Analyse
MINDEST_WOERTER   = 3           # Mindestwortanzahl fuer gueltiges Segment
UNTERSTUETZTE_FORMATE = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


# ---------------------------------------------------------------------------
# Bewertungsskala (Deutsch, 5 Kategorien)
# ---------------------------------------------------------------------------
class WpmKategorie(str, Enum):
    SEHR_LANGSAM  = "Sehr langsam"
    LANGSAM       = "Langsam"
    IDEAL         = "Ideal"
    SCHNELL       = "Schnell"
    SEHR_SCHNELL  = "Sehr schnell"


@dataclass
class WpmBewertung:
    kategorie : WpmKategorie
    wpm_von   : float
    wpm_bis   : float
    feedback  : str


# Skala fuer deutschsprachige Praesentationen (120-160 WPM = ideal)
WPM_SKALA: List[WpmBewertung] = [
    WpmBewertung(
        WpmKategorie.SEHR_LANGSAM, 0, 89,
        "Das Tempo ist zu langsam. Die Zuhoerer koennen den Faden verlieren. "
        "Etwas mehr Dynamik wuerde die Aufmerksamkeit steigern."
    ),
    WpmBewertung(
        WpmKategorie.LANGSAM, 90, 119,
        "Das Tempo ist etwas verhalten. Fuer komplexe Inhalte gut geeignet, "
        "bei einfacheren Themen darf es gerne etwas lebhafter sein."
    ),
    WpmBewertung(
        WpmKategorie.IDEAL, 120, 160,
        "Ausgezeichnetes Sprechtempo! Die Zielgeschwindigkeit fuer deutsche "
        "Praesentationen (120-160 WPM) wird perfekt getroffen. Weiter so!"
    ),
    WpmBewertung(
        WpmKategorie.SCHNELL, 161, 200,
        "Das Tempo ist leicht erhoet. Achten Sie darauf, Schluessel-"
        "aussagen durch kurze Pausen zu betonen. Ein Hauch mehr Ruhe hilft."
    ),
    WpmBewertung(
        WpmKategorie.SEHR_SCHNELL, 201, float("inf"),
        "Das Sprechtempo ist deutlich zu hoch. Zuhoerer koennen Inhalten "
        "kaum folgen. Bewusste Pausen und Verlangsamung sind dringend empfohlen."
    ),
]


def bewerte_wpm(wpm: float) -> WpmBewertung:
    """Ordnet einen WPM-Wert der passenden Bewertungskategorie zu."""
    for bewertung in WPM_SKALA:
        if bewertung.wpm_von <= wpm <= bewertung.wpm_bis:
            return bewertung
    # Fallback (sollte nicht eintreten)
    return WPM_SKALA[-1]


# ---------------------------------------------------------------------------
# Datenstrukturen fuer Ergebnisse
# ---------------------------------------------------------------------------
@dataclass
class WpmSegment:
    """Ergebnis eines 30-Sekunden-Analysefensters."""
    segment_nr       : int
    start_sek        : float
    end_sek          : float
    wortanzahl       : int
    wpm              : float
    kategorie        : str
    feedback         : str
    woerter          : List[str] = field(default_factory=list)


@dataclass
class SpeechAnalysisResult:
    """Gesamtergebnis der Sprechtempo-Analyse."""
    dateiname         : str
    audio_dauer_sek   : float
    gesamtwoerter     : int
    globale_wpm       : float
    globale_kategorie : str
    globales_feedback : str
    transkription     : str
    segmente          : List[WpmSegment]
    whisper_modell    : str
    sprache           : str

    def als_dict(self) -> dict:
        return asdict(self)

    def als_json(self, eingerueckt: bool = True) -> str:
        return json.dumps(self.als_dict(), ensure_ascii=False,
                          indent=2 if eingerueckt else None)


# ---------------------------------------------------------------------------
# Kern-Analyselogik
# ---------------------------------------------------------------------------
@dataclass
class WortStempel:
    """Einzelnes Wort mit Start- und Endzeitstempel aus Whisper."""
    wort  : str
    start : float
    end   : float


def extrahiere_wortstempel(transkriptions_ergebnis: dict) -> List[WortStempel]:
    """
    Extrahiert alle Worte mit Zeitstempeln aus dem Whisper-Ergebnis.

    Args:
        transkriptions_ergebnis: Rueckgabewert von whisper.transcribe()

    Returns:
        Liste von WortStempel-Objekten, chronologisch sortiert.

    Raises:
        ValueError: Falls keine Woerter mit Zeitstempeln gefunden werden.
    """
    wortstempel: List[WortStempel] = []

    segments = transkriptions_ergebnis.get("segments", [])
    if not segments:
        raise ValueError(
            "Whisper-Ergebnis enthaelt keine Segmente. "
            "Ist die Audiodatei leer oder beschaedigt?"
        )

    for seg in segments:
        woerter = seg.get("words", [])
        for w in woerter:
            text  = w.get("word", "").strip()
            start = w.get("start")
            end   = w.get("end")
            # Nur gueltige Eintraege uebernehmen
            if text and start is not None and end is not None:
                wortstempel.append(WortStempel(wort=text, start=start, end=end))

    if not wortstempel:
        raise ValueError(
            "Keine Woerter mit Zeitstempeln gefunden. "
            "Bitte word_timestamps=True sicherstellen."
        )

    wortstempel.sort(key=lambda w: w.start)
    logger.debug("Extrahiert: %d Woerter mit Zeitstempeln", len(wortstempel))
    return wortstempel


def berechne_globale_wpm(wortstempel: List[WortStempel]) -> tuple[float, float]:
    """
    Berechnet die globale WPM-Rate ueber die gesamte Aufnahme.

    Args:
        wortstempel: Chronologisch sortierte Liste der Worte.

    Returns:
        Tupel (globale_wpm, gesamtdauer_sekunden)
    """
    if len(wortstempel) < 2:
        logger.warning("Zu wenige Woerter fuer WPM-Berechnung (%d)", len(wortstempel))
        return 0.0, 0.0

    erste_start = wortstempel[0].start
    letztes_end = wortstempel[-1].end
    dauer_sek   = letztes_end - erste_start

    if dauer_sek <= 0:
        raise ValueError(
            f"Ungueltige Audiodauer berechnet: {dauer_sek:.2f}s. "
            "Zeitstempel korrumpiert?"
        )

    wpm = (len(wortstempel) / dauer_sek) * 60.0
    logger.info(
        "Globale WPM: %.1f | Woerter: %d | Dauer: %.1fs",
        wpm, len(wortstempel), dauer_sek
    )
    return round(wpm, 1), round(dauer_sek, 2)


def berechne_segmentweise_wpm(
    wortstempel       : List[WortStempel],
    segment_dauer_sek : int = SEGMENT_DAUER_SEK,
) -> List[WpmSegment]:
    """
    Berechnet WPM fuer aufeinanderfolgende Zeitfenster.

    Die Fenstereinteilung basiert auf absoluten Zeitmarken, nicht auf
    Wortanzahl. Pausen zwischen Woertern werden NICHT als Sprechzeit gezaehlt.

    Args:
        wortstempel       : Alle Woerter mit Zeitstempeln.
        segment_dauer_sek : Fenstergroesse in Sekunden (Standard: 30).

    Returns:
        Liste von WpmSegment-Objekten.
    """
    if not wortstempel:
        return []

    audio_start   = wortstempel[0].start
    audio_ende    = wortstempel[-1].end
    segmente      : List[WpmSegment] = []
    seg_nr        = 1

    fenster_start = audio_start
    while fenster_start < audio_ende:
        fenster_ende = fenster_start + segment_dauer_sek

        # Woerter in diesem Zeitfenster sammeln
        woerter_im_fenster = [
            w for w in wortstempel
            if w.start >= fenster_start and w.start < fenster_ende
        ]

        if len(woerter_im_fenster) >= MINDEST_WOERTER:
            # Tatsaechliche Sprechzeit = Summe der Wortdauern
            sprech_sek = sum(w.end - w.start for w in woerter_im_fenster)
            sprech_sek = max(sprech_sek, 0.1)  # Division durch 0 verhindern

            seg_wpm    = (len(woerter_im_fenster) / sprech_sek) * 60.0
            bewertung  = bewerte_wpm(seg_wpm)

            segment = WpmSegment(
                segment_nr  = seg_nr,
                start_sek   = round(fenster_start, 1),
                end_sek     = round(min(fenster_ende, audio_ende), 1),
                wortanzahl  = len(woerter_im_fenster),
                wpm         = round(seg_wpm, 1),
                kategorie   = bewertung.kategorie.value,
                feedback    = bewertung.feedback,
                woerter     = [w.wort for w in woerter_im_fenster],
            )
            segmente.append(segment)
            logger.debug(
                "Segment %d [%.0fs-%.0fs]: %d Woerter, WPM=%.1f (%s)",
                seg_nr, fenster_start, fenster_ende,
                len(woerter_im_fenster), seg_wpm, bewertung.kategorie.value
            )
        else:
            logger.debug(
                "Segment %d [%.0fs-%.0fs] uebersprungen "
                "(nur %d Woerter, Mindest: %d)",
                seg_nr, fenster_start, fenster_ende,
                len(woerter_im_fenster), MINDEST_WOERTER
            )

        fenster_start += segment_dauer_sek
        seg_nr        += 1

    return segmente


# ---------------------------------------------------------------------------
# Hauptanalyse-Funktion
# ---------------------------------------------------------------------------
def analysiere_sprechgeschwindigkeit(
    audiodatei      : str | Path,
    modell_groesse  : str = "base",
    sprache         : str = "de",
    segment_fenster : int = SEGMENT_DAUER_SEK,
) -> SpeechAnalysisResult:
    """
    Vollstaendige Sprechtempo-Analyse einer Audiodatei.

    Args:
        audiodatei      : Pfad zur Audiodatei (WAV, MP3, M4A, FLAC ...).
        modell_groesse  : Whisper-Modell (tiny, base, small, medium, large-v3).
        sprache         : ISO-639-1 Sprachcode (Standard: 'de' fuer Deutsch).
        segment_fenster : Fenstergroesse in Sekunden (Standard: 30).

    Returns:
        SpeechAnalysisResult mit allen Metriken.

    Raises:
        FileNotFoundError : Datei nicht gefunden.
        ValueError        : Ungueltige Datei oder Analysefehler.
        RuntimeError      : Whisper-Ladefehler oder Transkriptionsfehler.
    """
    # -- Importguard: whisper erst hier importieren (schwere Abhaengigkeit)
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Whisper nicht installiert. "
            "Bitte 'pip install openai-whisper' ausfuehren."
        ) from exc

    # 1. Dateipruefung
    pfad = Path(audiodatei)
    if not pfad.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {pfad}")
    if pfad.suffix.lower() not in UNTERSTUETZTE_FORMATE:
        raise ValueError(
            f"Dateiformat '{pfad.suffix}' nicht unterstuetzt. "
            f"Erlaubt: {', '.join(UNTERSTUETZTE_FORMATE)}"
        )
    if pfad.stat().st_size == 0:
        raise ValueError(f"Audiodatei ist leer: {pfad}")

    logger.info("Starte Analyse: '%s' | Modell: %s | Sprache: %s",
                pfad.name, modell_groesse, sprache)

    # 2. Whisper-Modell laden
    try:
        logger.info("Lade Whisper-Modell '%s' ...", modell_groesse)
        modell = whisper.load_model(modell_groesse)
    except Exception as exc:
        raise RuntimeError(
            f"Fehler beim Laden des Whisper-Modells '{modell_groesse}': {exc}"
        ) from exc

    # 3. Transkription mit Wortzeitstempeln
    try:
        logger.info("Transkribiere Audio ...")
        ergebnis = modell.transcribe(
            str(pfad),
            language          = sprache,
            word_timestamps   = True,
            verbose           = False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Whisper-Transkription fehlgeschlagen: {exc}"
        ) from exc

    transkriptions_text = ergebnis.get("text", "").strip()
    logger.info("Transkription abgeschlossen. Zeichen: %d", len(transkriptions_text))

    # 4. Wortstempel extrahieren
    wortstempel = extrahiere_wortstempel(ergebnis)

    # 5. Globale WPM berechnen
    globale_wpm, audio_dauer = berechne_globale_wpm(wortstempel)
    globale_bewertung        = bewerte_wpm(globale_wpm)

    # 6. Segmentweise WPM berechnen
    segmente = berechne_segmentweise_wpm(wortstempel, segment_fenster)

    # 7. Ergebnis zusammenstellen
    resultat = SpeechAnalysisResult(
        dateiname         = pfad.name,
        audio_dauer_sek   = audio_dauer,
        gesamtwoerter     = len(wortstempel),
        globale_wpm       = globale_wpm,
        globale_kategorie = globale_bewertung.kategorie.value,
        globales_feedback = globale_bewertung.feedback,
        transkription     = transkriptions_text,
        segmente          = segmente,
        whisper_modell    = modell_groesse,
        sprache           = sprache,
    )

    logger.info(
        "Analyse abgeschlossen: WPM=%.1f (%s) | Segmente: %d",
        globale_wpm, globale_bewertung.kategorie.value, len(segmente)
    )
    return resultat


# ---------------------------------------------------------------------------
# Report-Ausgabe
# ---------------------------------------------------------------------------
def drucke_report(resultat: SpeechAnalysisResult) -> None:
    """Gibt einen lesbaren Analysereport auf der Konsole aus."""
    trennlinie = "=" * 65

    print(trennlinie)
    print("  SPRECHTEMPO-ANALYSE - PRAESENTATION")
    print(trennlinie)
    print(f"  Datei          : {resultat.dateiname}")
    print(f"  Sprache        : {resultat.sprache.upper()}")
    print(f"  Whisper-Modell : {resultat.whisper_modell}")
    print(f"  Audio-Dauer    : {resultat.audio_dauer_sek:.1f} Sekunden "
          f"({resultat.audio_dauer_sek/60:.1f} Minuten)")
    print(f"  Gesamtwoerter  : {resultat.gesamtwoerter}")
    print(trennlinie)

    print(f"\n  GLOBALE BEWERTUNG")
    print(f"  WPM            : {resultat.globale_wpm:.1f}")
    print(f"  Kategorie      : {resultat.globale_kategorie}")
    print(f"  Feedback       :")
    # Zeilenumbruch bei langem Feedback
    for zeile in _umbreche_text(resultat.globales_feedback, 55):
        print(f"    {zeile}")

    if resultat.segmente:
        print(f"\n  SEGMENT-ANALYSE (Fenster: 30 Sekunden)")
        print(f"  {'Nr':>3}  {'Zeit':>14}  {'Woerter':>8}  {'WPM':>7}  Kategorie")
        print(f"  {'-'*60}")
        for seg in resultat.segmente:
            zeit_str = f"{seg.start_sek:.0f}s - {seg.end_sek:.0f}s"
            print(
                f"  {seg.segment_nr:>3}  {zeit_str:>14}  "
                f"{seg.wortanzahl:>8}  {seg.wpm:>7.1f}  {seg.kategorie}"
            )

    print(f"\n  TRANSKRIPTION (Ausschnitt):")
    ausschnitt = resultat.transkription[:300]
    if len(resultat.transkription) > 300:
        ausschnitt += " [...]"
    for zeile in _umbreche_text(ausschnitt, 60):
        print(f"    {zeile}")

    print("\n" + trennlinie)


def _umbreche_text(text: str, breite: int) -> List[str]:
    """Einfacher Zeilenumbruch fuer Konsolen-Ausgabe."""
    woerter = text.split()
    zeilen  = []
    aktuelle_zeile: List[str] = []
    laenge = 0

    for wort in woerter:
        if laenge + len(wort) + 1 > breite and aktuelle_zeile:
            zeilen.append(" ".join(aktuelle_zeile))
            aktuelle_zeile = [wort]
            laenge         = len(wort)
        else:
            aktuelle_zeile.append(wort)
            laenge += len(wort) + 1

    if aktuelle_zeile:
        zeilen.append(" ".join(aktuelle_zeile))
    return zeilen


# ---------------------------------------------------------------------------
# CLI-Interface
# ---------------------------------------------------------------------------
def erstelle_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "sprechtempoanalyse",
        description = "Sprechtempo-Messung fuer Praesentationsbewertungs-App "
                      "(OpenAI Whisper)",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audiodatei",
        type    = str,
        help    = "Pfad zur Audiodatei (WAV, MP3, M4A, FLAC, OGG, WEBM)",
    )
    parser.add_argument(
        "--modell", "-m",
        type    = str,
        default = "base",
        choices = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help    = "Whisper-Modellgroesse",
    )
    parser.add_argument(
        "--sprache", "-s",
        type    = str,
        default = "de",
        help    = "ISO-639-1 Sprachcode (z.B. 'de', 'en', 'fr')",
    )
    parser.add_argument(
        "--fenster", "-f",
        type    = int,
        default = SEGMENT_DAUER_SEK,
        help    = "Segmentfenster in Sekunden",
    )
    parser.add_argument(
        "--json-export", "-j",
        type    = str,
        default = None,
        metavar = "AUSGABEDATEI",
        help    = "Ergebnis als JSON exportieren (z.B. ergebnis.json)",
    )
    parser.add_argument(
        "--debug", "-d",
        action  = "store_true",
        help    = "Debug-Logging aktivieren",
    )
    return parser


def main() -> int:
    """Haupteinstiegspunkt fuer CLI-Nutzung."""
    parser = erstelle_argument_parser()
    args   = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug-Modus aktiviert")

    try:
        resultat = analysiere_sprechgeschwindigkeit(
            audiodatei      = args.audiodatei,
            modell_groesse  = args.modell,
            sprache         = args.sprache,
            segment_fenster = args.fenster,
        )

        drucke_report(resultat)

        if args.json_export:
            export_pfad = Path(args.json_export)
            export_pfad.write_text(
                resultat.als_json(), encoding="utf-8"
            )
            logger.info("JSON-Export gespeichert: %s", export_pfad)
            print(f"\n  JSON-Export: {export_pfad}")

        return 0

    except FileNotFoundError as exc:
        logger.error("Datei nicht gefunden: %s", exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 2

    except ValueError as exc:
        logger.error("Validierungsfehler: %s", exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 3

    except RuntimeError as exc:
        logger.error("Laufzeitfehler: %s", exc)
        print(f"\nFEHLER: {exc}", file=sys.stderr)
        return 4

    except KeyboardInterrupt:
        print("\nAbgebrochen.", file=sys.stderr)
        return 130


# ---------------------------------------------------------------------------
# Programmeinstieg
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
