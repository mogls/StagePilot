"""gesamtscore.py
==============================================================================
Aggregiert alle Modul-Scores zu einem konsistenten Gesamtergebnis und führt
die 5 zentralen Konsistenz-Checks aus Abschnitt 12.5 der Planung durch.

Modalitaets-Gewichtung (Abschnitt 12.2):
    Inhalt/Sprache  25 %
      - Füllwörter        15 %  (15 % des Gesamt = 60 % innerhalb der 25)
      - Sprechfluss         10 %  (10 % des Gesamt = 40 % innerhalb der 25)
    Prosodie        45 %
      - Sprechtempo         20 % von 45 =  9.00 %
      - Pausen              25 % von 45 = 11.25 %
      - Lautstaerke         15 % von 45 =  6.75 %
      - Pitch-Variation     20 % von 45 =  9.00 %
      - Emotionale Var.     20 % von 45 =  9.00 %
    Video           30 %
      - Video-Analyse       100 % von 30 = 30 %

Inputs:
    zwischen_output/*.json  (alle Modul-Outputs)

Outputs:
    zwischen_output/gesamtscore_output.json
    reports/gesamt/gesamt_report_<TIMESTAMP>.txt

Aufruf:
    python gesamtscore.py [--input-dir zwischen_output] [--output-dir reports/gesamt]
=============================================================================="""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple


# ============================================================================
# GEWICHTUNGEN — Abschnitt 12.2--12.4 der Planung
# ============================================================================

GEWICHT_MODALITAET = {
    "inhalt_sprache": 0.25,
    "prosodie": 0.45,
    "video": 0.30,
}

# Anteile innerhalb Inhalt/Sprache (v2-Fix: Füllwörter 15%, Sprechfluss 10%)
GEWICHT_INHALT = {
    "fuellwoerter":  0.15 / 0.25,   # 0.60
    "sprechfluss":   0.10 / 0.25,   # 0.40
}

# Anteile innerhalb Prosodie
GEWICHT_PROSODIE = {
    "sprechtempo":            0.20,
    "pausen":                 0.25,
    "lautstaerke":            0.15,
    "pitch_variation":        0.20,
    "emotionale_variation":   0.20,
}

# Anteile innerhalb Video
GEWICHT_VIDEO = {
    "video": 1.00,
}


# ============================================================================
# MODUL-JSONs -> SCORE-EXTRAKTION
# ============================================================================

# Erwartete Dateinamen im zwischen_output-Verzeichnis
MODUL_DATEIEN = {
    "fuellwoerter":          "fuellwoerter_analyse_output.json",
    "sprechtempo":           "sprechtempo_analyse_output.json",
    "pausen":                "pausen_analyse_output.json",
    "sprechfluss":           "sprechfluss_analyse_output.json",
    "lautstaerke":           "lautstaerke_analyse_output.json",
    "pitch_variation":       "pitch_variation_analyse_output.json",
    "emotionale_variation":  "emotionale_variation_analyse_output.json",
    "video":                 "video_analyse_output.json",
}


def lade_modul_json(pfad: Path) -> Optional[Dict]:
    """Liefert dict oder None wenn Datei fehlt/kaputt ist."""
    if not pfad.exists():
        return None
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[gesamtscore][WARN] Konnte {pfad.name} nicht laden: {e}")
        return None


def extrahiere_score(daten: Dict) -> Optional[float]:
    """
    Liefert den Gesamtscore eines Moduls.
    Kompatibel mit beiden Formaten:
      - {"scoring": {"gesamtscore": 87}}     (pausen, sprechfluss, lautstaerke, ...)
      - {"gesamt_score": 87}                 (füllwörter, sprechtempo)
    """
    if daten is None:
        return None
    if "scoring" in daten and isinstance(daten["scoring"], dict):
        s = daten["scoring"].get("gesamtscore")
        if s is not None:
            return float(s)
    for key in ("gesamt_score", "gesamtscore", "gesamt_punkte", "score"):
        if key in daten:
            try:
                return float(daten[key])
            except (TypeError, ValueError):
                pass
    return None


def sammle_modul_scores(input_dir: Path) -> Dict[str, Optional[float]]:
    """
    Liest für jedes Modul aus MODUL_DATEIEN den Gesamtscore und liefert
    ein dict { modul_key -> score oder None }.
    """
    scores = {}
    for modul_key, dateiname in MODUL_DATEIEN.items():
        pfad = input_dir / dateiname
        daten = lade_modul_json(pfad)
        scores[modul_key] = extrahiere_score(daten)
        if scores[modul_key] is None:
            print(f"[gesamtscore] {modul_key:25s} nicht bewertet "
                  f"(Datei: {dateiname})")
        else:
            print(f"[gesamtscore] {modul_key:25s} Score: {scores[modul_key]:.1f}")
    return scores


# ============================================================================
# GEWICHTETE AGGREGATION
# ============================================================================

def gewichte_gruppe(scores: Dict[str, Optional[float]],
                    gewichte: Dict[str, float]) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Gewichtete Aggregation mit graceful degradation:
    - Wenn einzelne Module fehlen, werden die Gewichte der vorhandenen normiert.
    - Wenn ALLE Module einer Gruppe fehlen, wird None zurückgegeben.

    Returns:
        (gruppen_score, effektive_gewichte_dict)
    """
    vorhanden = {k: v for k, v in gewichte.items()
                 if k in scores and scores[k] is not None}
    if not vorhanden:
        return None, {}
    # Normieren
    summe = sum(vorhanden.values())
    effektiv = {k: v / summe for k, v in vorhanden.items()}
    aggregiert = sum(scores[k] * w for k, w in effektiv.items())
    return round(aggregiert, 2), effektiv


def berechne_gesamtscore(scores: Dict[str, Optional[float]]) -> Dict:
    """
    Rechnet die drei Modalitaets-Scores und den Gesamtscore.
    """
    inhalt_score, inhalt_w = gewichte_gruppe(scores, GEWICHT_INHALT)
    prosodie_score, prosodie_w = gewichte_gruppe(scores, GEWICHT_PROSODIE)
    video_score, video_w = gewichte_gruppe(scores, GEWICHT_VIDEO)

    modalitaet_scores = {
        "inhalt_sprache": inhalt_score,
        "prosodie": prosodie_score,
        "video": video_score,
    }
    # Gesamt: normieren wenn Modalitaeten fehlen
    vorhandene = {k: v for k, v in modalitaet_scores.items() if v is not None}
    if not vorhandene:
        gesamt = None
    else:
        gewichte = {k: GEWICHT_MODALITAET[k] for k in vorhandene}
        summe = sum(gewichte.values())
        gewichte = {k: v / summe for k, v in gewichte.items()}
        gesamt = round(sum(vorhandene[k] * gewichte[k] for k in vorhandene), 2)

    return {
        "gesamtscore": gesamt,
        "modalitaeten": modalitaet_scores,
        "effektive_gewichte": {
            "inhalt_sprache": inhalt_w,
            "prosodie": prosodie_w,
            "video": video_w,
        },
        "modul_scores": scores,
    }


# ============================================================================
# KONSISTENZ-CHECKS — Abschnitt 12.5 der Planung
# ============================================================================

def _hole_kennzahl(daten: Optional[Dict], *keys) -> Optional[float]:
    """
    Navigiert einen verschachtelten dict-Pfad. Robust gegen None.
    _hole_kennzahl(d, "statistiken", "stocker_rate_pro_min")
    """
    if daten is None:
        return None
    current = daten
    for k in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(k)
        if current is None:
            return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def check_1_nervositaet(sprechtempo, pausen, sprechfluss):
    """
    Trigger: Sprechtempo < 3.5 Silben/Sek UND
             (Wiederholungs-Rate > 1/Min ODER Stocker-Rate > 3/Min).
    """
    tempo = _hole_kennzahl(sprechtempo, "gesamttempo_silben_sek")\
        or _hole_kennzahl(sprechtempo, "gesamttempo")
    stocker_rate = _hole_kennzahl(pausen, "statistiken", "stocker_rate_pro_min")
    disfluenz_rate = _hole_kennzahl(sprechfluss, "statistiken", "disfluenz_rate_pro_min")

    if tempo is None:
        return None
    if tempo >= 3.5:
        return None
    if (disfluenz_rate is not None and disfluenz_rate > 1.0) or\
       (stocker_rate is not None and stocker_rate > 3.0):
        return {
            "check": "nervositaets_muster",
            "titel": "Nervositaets-Muster (Clark & Fox Tree 2002)",
            "text": ("Langsame Sprecher zeigen mehr Disfluenzen aufgrund von "
                     "Sprachplanungs-Verzögerungen. Empfehlung: Text vor der "
                     "Präsentation stärker verinnerlichen."),
            "kennzahlen": {
                "sprechtempo_silben_sek": tempo,
                "disfluenz_rate": disfluenz_rate,
                "stocker_rate": stocker_rate,
            },
        }
    return None


def check_2_kernbotschaft_unbetont(lautstaerke, sprechtempo, pitch_variation):
    """
    Trigger: Kernbotschaften weder lauter (Lautstaerke-D1 < 50) noch langsamer
             (Sprechtempo-D3 < 50) noch pitch-moduliert (Pitch-D3 < 50).
    """
    l_d1 = _hole_kennzahl(lautstaerke, "scoring", "d1_kernbotschaftsbetonung", "punkte")
    t_d3 = _hole_kennzahl(sprechtempo, "punkte_kernbotschaften")\
        or _hole_kennzahl(sprechtempo, "scoring", "d3_kernbotschaften", "punkte")
    p_d3 = _hole_kennzahl(pitch_variation, "scoring", "d3_kernbotschaftsvariation", "punkte")\
        or _hole_kennzahl(pitch_variation, "scoring", "d3_kernbotschafts_variation", "punkte")

    # Mindestens 2 der 3 Werte müssen verfügbar sein
    verfuegbar = [x for x in (l_d1, t_d3, p_d3) if x is not None]
    if len(verfuegbar) < 2:
        return None
    unbetont = all(x is None or x < 50 for x in (l_d1, t_d3, p_d3))
    if unbetont:
        return {
            "check": "kernbotschaft_unbetont",
            "titel": "Inkonsistente Kernbotschafts-Betonung",
            "text": ("Kernbotschaften werden in keiner Prosodie-Dimension "
                     "besonders hervorgehoben. Wähle mindestens eine bewusste "
                     "Betonungs-Strategie (lauter, langsamer oder pitch-moduliert)."),
            "kennzahlen": {
                "lautstaerke_d1": l_d1,
                "sprechtempo_d3": t_d3,
                "pitch_d3": p_d3,
            },
        }
    return None


def check_3_video_audio_diskrepanz(video, gesamtscores):
    """
    Trigger: Video-Score > 80 UND Audio-Score (Prosodie) < 50 (oder umgekehrt).
    """
    video_score = gesamtscores["modalitaeten"].get("video")
    prosodie_score = gesamtscores["modalitaeten"].get("prosodie")
    if video_score is None or prosodie_score is None:
        return None
    diff = abs(video_score - prosodie_score)
    if diff >= 30 and ((video_score > 80 and prosodie_score < 50) or
                       (prosodie_score > 80 and video_score < 50)):
        return {
            "check": "video_audio_diskrepanz",
            "titel": "Video-Audio-Diskrepanz",
            "text": ("Starke Diskrepanz zwischen Körpersprache und Stimme — "
                     "die AI-Module bewerten die Modalitaeten sehr unterschiedlich. "
                     "Prüfe, ob eine der beiden nachbearbeitet werden muss."),
            "kennzahlen": {
                "video_score": video_score,
                "prosodie_score": prosodie_score,
                "differenz": diff,
            },
        }
    return None


def check_4_modalpartikel_hektik(sprechtempo, fuellwoerter):
    """
    Trigger: Sprechtempo > 6 Silben/Sek UND Modalpartikeln > 12/Min.
    """
    tempo = _hole_kennzahl(sprechtempo, "gesamttempo_silben_sek")\
        or _hole_kennzahl(sprechtempo, "gesamttempo")
    modal = _hole_kennzahl(fuellwoerter, "modal", "pro_min")\
        or _hole_kennzahl(fuellwoerter, "statistiken", "modal_pro_min")
    if tempo is None or modal is None:
        return None
    if tempo > 6.0 and modal > 12.0:
        return {
            "check": "modalpartikel_hektik",
            "titel": "Modalpartikel-Sprechtempo-Muster",
            "text": ("Hastiges Sprechen kombiniert mit vielen Modalpartikeln — "
                     "typisches Hektik-Muster. Verlangsamung und bewusste "
                     "Pausen würden die Wirkung stärken."),
            "kennzahlen": {
                "sprechtempo_silben_sek": tempo,
                "modalpartikeln_pro_min": modal,
            },
        }
    return None


def check_5_fehlende_rhet_pausen(pausen):
    """
    Trigger: > 50 % der Kernbotschaften ohne rhetorische Pause davor/danach.
    """
    kb_info = pausen.get("kernbotschaft_check") if isinstance(pausen, dict) else None
    if not kb_info:
        return None
    gesamt = kb_info.get("gesamt", 0)
    ohne = kb_info.get("ohne_rhetorische_pause", 0)
    if gesamt < 2:
        return None
    anteil = ohne / gesamt
    if anteil > 0.5:
        return {
            "check": "fehlende_rhetorische_pausen",
            "titel": "Fehlende rhetorische Pausen an Kernbotschaften",
            "text": ("Kernbotschaften brauchen Wirkung durch Pausen. Aktuell "
                     f"werden {ohne} von {gesamt} Kernbotschaften ohne rhetorische "
                     "Pause davor/danach gesprochen."),
            "kennzahlen": {
                "kernbotschaften_gesamt": gesamt,
                "ohne_pause": ohne,
                "anteil": round(anteil, 2),
            },
        }
    return None


def alle_konsistenz_checks(daten: Dict[str, Optional[Dict]],
                           gesamtscores: Dict) -> List[Dict]:
    """Führt alle 5 Checks aus und liefert die Treffer als Liste."""
    treffer = []
    for check_fn, args in [
        (check_1_nervositaet,
            (daten["sprechtempo"], daten["pausen"], daten["sprechfluss"])),
        (check_2_kernbotschaft_unbetont,
            (daten["lautstaerke"], daten["sprechtempo"], daten["pitch_variation"])),
        (check_3_video_audio_diskrepanz,
            (daten["video"], gesamtscores)),
        (check_4_modalpartikel_hektik,
            (daten["sprechtempo"], daten["fuellwoerter"])),
        (check_5_fehlende_rhet_pausen,
            (daten["pausen"],)),
    ]:
        try:
            result = check_fn(*args)
            if result:
                treffer.append(result)
        except Exception as e:
            print(f"[gesamtscore][WARN] Check {check_fn.__name__} "
                  f"crashed: {e}")
    return treffer


# ============================================================================
# TOP-3 VERBESSERUNGSVORSCHLAEGE
# ============================================================================

def top3_schwaechste(scores: Dict[str, Optional[float]]) -> List[Tuple[str, float]]:
    """Liefert die 3 Module mit dem niedrigsten Score."""
    vorhanden = [(k, v) for k, v in scores.items() if v is not None]
    vorhanden.sort(key=lambda x: x[1])
    return vorhanden[:3]


# ============================================================================
# REPORT
# ============================================================================

def generiere_report(gesamtscores: Dict,
                     scores: Dict[str, Optional[float]],
                     konsistenz: List[Dict],
                     input_dir: Path) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("PRAESENTATIONS-GESAMTBEWERTUNG")
    lines.append("=" * 70)
    lines.append(f"Erzeugt: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Quelle:  {input_dir}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("GESAMTSCORE")
    lines.append("-" * 70)
    gesamt = gesamtscores["gesamtscore"]
    lines.append(f"  {gesamt if gesamt is not None else '—'} / 100")
    lines.append("")

    lines.append("-" * 70)
    lines.append("MODALITAETEN")
    lines.append("-" * 70)
    for name, key, gewicht in [
        ("Inhalt / Sprache", "inhalt_sprache", 0.25),
        ("Prosodie",         "prosodie",       0.45),
        ("Video",            "video",          0.30),
    ]:
        val = gesamtscores["modalitaeten"].get(key)
        val_s = f"{val:5.1f}" if val is not None else "  —  "
        lines.append(f"  {name:20s} ({gewicht:>4.0%})   {val_s} / 100")
    lines.append("")

    lines.append("-" * 70)
    lines.append("EINZEL-MODULE")
    lines.append("-" * 70)
    for k, v in scores.items():
        v_s = f"{v:5.1f}" if v is not None else " n/a "
        lines.append(f"  {k:25s} {v_s} / 100")
    lines.append("")

    if konsistenz:
        lines.append("-" * 70)
        lines.append(f"KONSISTENZ-HINWEISE ({len(konsistenz)})")
        lines.append("-" * 70)
        for i, c in enumerate(konsistenz, 1):
            lines.append(f"  {i}. {c['titel']}")
            lines.append(f"     {c['text']}")
            lines.append("")
    else:
        lines.append("-" * 70)
        lines.append("KONSISTENZ-HINWEISE")
        lines.append("-" * 70)
        lines.append("  Keine kritischen Muster erkannt.")
        lines.append("")

    top3 = top3_schwaechste(scores)
    if top3:
        lines.append("-" * 70)
        lines.append("TOP-3 VERBESSERUNGSPOTENZIAL")
        lines.append("-" * 70)
        for i, (modul, score) in enumerate(top3, 1):
            lines.append(f"  {i}. {modul:25s} {score:5.1f} / 100 "
                         f"— siehe Detail-Report {modul}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("ENDE REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

def aggregiere(input_dir: Path,
               output_json: Path,
               output_txt: Path) -> Dict:
    """Führt die komplette Aggregation durch und schreibt beide Outputs."""
    print(f"[gesamtscore] Lese Modul-JSONs aus {input_dir}")

    # 1. Rohdaten laden
    modul_daten = {
        k: lade_modul_json(input_dir / v)
        for k, v in MODUL_DATEIEN.items()
    }

    # 2. Scores extrahieren
    scores = {k: extrahiere_score(v) for k, v in modul_daten.items()}

    # 3. Gewichtete Aggregation
    gesamtscores = berechne_gesamtscore(scores)

    # 4. Konsistenz-Checks
    konsistenz = alle_konsistenz_checks(modul_daten, gesamtscores)

    # 5. JSON-Output
    output = {
        "modul": "gesamtscore",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "modul_scores": scores,
        "modalitaets_scores": gesamtscores["modalitaeten"],
        "effektive_gewichte": gesamtscores["effektive_gewichte"],
        "gesamtscore": gesamtscores["gesamtscore"],
        "konsistenz_hinweise": konsistenz,
        "top3_schwaechste": [
            {"modul": m, "score": s} for m, s in top3_schwaechste(scores)
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[gesamtscore] JSON: {output_json}")

    # 6. TXT-Report
    report = generiere_report(gesamtscores, scores, konsistenz, input_dir)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[gesamtscore] Report: {output_txt}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Aggregiert alle Modul-Scores + Konsistenz-Checks."
    )
    parser.add_argument("--input-dir", default="zwischen_output",
                        help="Verzeichnis mit den Modul-JSONs")
    parser.add_argument("--output-json", default="zwischen_output/gesamtscore_output.json")
    parser.add_argument("--output-report-dir", default="reports/gesamt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt = Path(args.output_report_dir) / f"gesamt_report_{ts}.txt"

    aggregiere(input_dir, output_json, output_txt)


if __name__ == "__main__":
    main()
