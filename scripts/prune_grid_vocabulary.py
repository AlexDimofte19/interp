#!/usr/bin/env python3
"""Prune the grid signal vocabulary down to tokens that actually carry grid information.

`data/jlens/grid_tokens_full.json` was produced by `notebooks/grid_tokens.ipynb` before the
model-token-only rule (see `data/jlens/README.md`), and it shows: of its 1258 tokens, 830
never reach a jlens top-20 at all, and a large share of the mass that *is* captured comes
from words with no grid sense -- ' cannot', ' index', ' case', ' field', ' line', bare 'a'
and bare 'g'.

This script does not re-run the notebook and does not touch the deployed vocabulary. It
reads the committed file, applies a set of **named, auditable rules**, and writes a pruned
copy plus a per-token report saying which rule dropped what. Every rule is a lexical
judgement about the token -- "is this a word for a thing in a grid world?" -- never a
measurement of how loud the token is in j-space, because the j-space is what the vocabulary
is used to measure (`notebooks/grid_tokens.ipynb`, cell 0).

Dry-run by default, like the other destructive scripts here; pass --write to emit files.

    python scripts/prune_grid_vocabulary.py
    python scripts/prune_grid_vocabulary.py --write --out data/jlens/grid_tokens_pruned.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

SEP_RE = re.compile(r"""[_\-./\\:,;()\[\]{}<>"'`|!?*+=#@$%^&~\s]+""")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "data" / "jlens" / "grid_tokens_full.json"
DEFAULT_OUT = REPO / "data" / "jlens" / "grid_tokens_pruned.json"
DEFAULT_REPORT = REPO / "data" / "jlens" / "grid_tokens_prune_report.csv"


def bare(tok: str) -> str:
    """Surface form with punctuation and whitespace shaved off, as the notebook strips it.

    >>> bare("_LEFT"), bare(" col"), bare("(parsed")
    ('LEFT', 'col', 'parsed')
    """
    return SEP_RE.sub("", tok.strip())


# --------------------------------------------------------------------------------------
# The rules. Each is (reason, {bare forms}), matched case-sensitively against bare(token)
# unless the name ends in "_ci", which lowercases both sides.
#
# Case matters more than usual here: the grid legend's glyphs are 'A' and 'G', so the
# uppercase single letters are the agent and the goal, while the lowercase ones are the
# English article and a stray letter. Folding case would delete the two real symbols.
# --------------------------------------------------------------------------------------

RULES: dict[str, set[str]] = {
    # ---- single letters and articles -------------------------------------------------
    # 'A'/'G' are the legend glyphs and stay. 'a'/'an' are the commonest words in English
    # and 'g' is not a symbol this grid uses; every one of them fires broadly on ordinary
    # text and contributes a noise floor to every token's score.
    "bare_lowercase_letter": {"a", "g", "gg", "gv", "gw", "gha", "ġ"},
    "english_article": {"an", "An", "AN"},
    # letter bigrams difflib pulled in beside 'G'
    "letter_bigram": {"GF", "GC", "Gh", "Gy", "Gl", "Gol", "Glob", "GLES", "Ｇ", "Â", "Ã", "â", "ª"},
    # ---- generic modality: STATUS is a negation class, not a grid class ---------------
    # ' cannot' alone is the seventh-largest mass absorber in the whole vocabulary. None of
    # these say anything about a grid; the grid sense lives in blocked/impassable/
    # unreachable, which are kept.
    "generic_modality_ci": {
        "cannot",
        "fail",
        "fails",
        "unable",
        "unnable",
        "incapable",
        "inability",
        "impotence",
        "impair",
        "unavoidable",
        "uncontrolled",
        "undesirable",
        "incompetent",
        "uncomplicated",
        "unsuitable",
        "incompatible",
        "unsupported",
        "unprocessable",
        "unmodifiable",
        "unshift",
        "unchecked",
        "Unhandled",
        "UNRELATED",
        "detachable",
        "refuses",
        "unavailable",
        "onvoldoende",
    },
    # ---- code senses of pass / through / block ---------------------------------------
    # 'blocked'/'blocking'/'blockage'/'blockade' keep the grid sense; the bare stems are
    # control-flow and HTML words the lens emits constantly.
    "code_control_flow_ci": {"pass", "Passing", "through", "block", "BLOCK", "blockquote", "undable"},
    # ---- the "case" family: from the French anchor "case vide" ------------------------
    # MiniLM matched the English programming 'case'. Every one of these is switch/case.
    "anchor_drift_case_ci": {
        "case",
        "cases",
        "Cases",
        "kasus",
        "cazul",
        "případ",
        "prípade",
        "slučaju",
        "случаи",
        "ケース",
        "मामले",
        "casos",
        "案件",
        "CASE",
    },
    # ---- the "field" family: from the German anchor "freies Feld" ---------------------
    "anchor_drift_field_ci": {"field", "fields", "Feld", "veld", "fält", "สนาม", "fieldset", "FIELD", "FIELDS"},
    # ---- the "free" anchor, which reached gambling spam --------------------------------
    "anchor_drift_free_ci": {"เงินฟรี", "бесплат", "spotless", "barren", "deserted", "vacant"},
    # ---- the "index" family: a programming word, not a grid word ----------------------
    # The model says 'row' and 'col'; over 268k sampled reasoning tokens it says ' index'
    # 123 times and 'indexes' never.
    "code_index_ci": {"index", "indexes", "indexed", "indexing", "Indexer", "Index", "Indexes", "Indexed"},
    # ---- the English "line" family ----------------------------------------------------
    # 'ligne'/'linha'/'línea'/'Reihe'/'rij'/'række' are the real row-words in other
    # languages and stay. English 'line' is not a word this model uses for a grid row.
    "english_line_ci": {"line", "线", "線", "ライン", "lijn", "linien", "लाइन", "লাইন"},
    # ---- difflib bridging 'col'/'colonne' to 'colon' -----------------------------------
    "difflib_colon_ci": {"colon", "kolon", "colo"},
    # ---- geodesy: latitude/longitude/geometry/GPS are not grid coordinates ------------
    "geo_drift_ci": {
        "latitude",
        "longitude",
        "Geometry",
        "geometry",
        "ometr",
        "GPS",
        "projection",
        "Matrices",
        "matrices",
        "Stack",
        "Longitude",
        "Latitude",
    },
    # ---- 'travel' drift out of the GOAL anchor 'destination' --------------------------
    "goal_drift_travel_ci": {
        "travel",
        "Travel",
        "viajar",
        "viagem",
        "viagens",
        "viaggio",
        "reizen",
        "reisen",
        "reise",
        "यात्रा",
        "goalie",
        "quadrant",
    },
    # ---- generic code 'location' ------------------------------------------------------
    "code_location_ci": {"LOCATION", "Location"},
    # ---- 'challenge'/'hurdle': the semantic neighbourhood of 'obstacle', not a wall ----
    "wall_drift_challenge_ci": {
        "challenge",
        "desafio",
        "desafío",
        "défi",
        "défis",
        "Herausforderung",
        "uitdaging",
        "hurdle",
        "hurdles",
        "shielding",
        "щит",
        "hinder",
        "gard",
    },
    # ---- 'concrete'/'cement': material words the traces never use ----------------------
    "wall_drift_material_ci": {"concrete", "Concrete", "concreto", "cement", "cements", "cemento", "الأسمنت", "duct"},
    # ---- difflib garbage around the Spanish/French wall seeds --------------------------
    # 'pared' -> spared/parsed/pred/parked/Parte; 'barrier' -> carrera/Karriere;
    # 'mur' -> Mahl/Maus/mambo/maw. These are character neighbours, not meanings.
    "difflib_wall_neighbour_ci": {
        "spared",
        "Parte",
        "parte",
        "parked",
        "Parsed",
        "parsed",
        "pred",
        "parentes",
        "prete",
        "Karriere",
        "carrera",
        "Carrera",
        "arrera",
        "auer",
        "ARRIER",
        "alls",
        "stacles",
        "stacle",
        "Mahl",
        "mahl",
        "Mahon",
        "Mahm",
        "Maw",
        "maw",
        "mav",
        "maml",
        "maual",
        "Maus",
        "mahdoll",
        "mahimong",
        "mafai",
        "mambo",
        "lombok",
        "maqu",
        "ítés",
        "ūra",
        "τεί",
        "хана",
        "láthair",
        "brú",
        "costat",
        "ഇട",
    },
    # ---- difflib garbage elsewhere: fragments that are not words -----------------------
    "difflib_fragment_ci": {
        "osition",
        "posito",
        "positivo",
        "positi",
        "lige",
        "ilas",
        "igne",
        "OLUMNS",
        "OLUMN",
        "ordinates",
        "inha",
        "linh",
        "fias",
        "eile",
        "rowse",
        "rowth",
        "rowser",
        "zed",
        "zech",
        "zept",
        "zerano",
        "zeuge",
        "zell",
        "Zell",
        "Zel",
        "Zet",
        "Zert",
        "Zeb",
        "zeb",
        "zeal",
        "zette",
        "slapen",
        "oplasm",
        "uyant",
        "tract",
        "uegos",
        "inson",
        "cellence",
        "cellent",
        "perto",
        "voto",
        "acio",
        "azio",
        "ilibre",
        "ouver",
        "roffen",
        "ofen",
        "ffen",
        "ibre",
        "ivre",
        "berto",
        "hoffen",
        "couvert",
        "offens",
        "Libro",
        "libro",
        "llibre",
        "Libr",
        "libr",
        "offent",
        "sapertos",
        "pust",
        "amespace",
        "SPATH",
        "agments",
        "agment",
        "AGMENT",
        "agens",
        "angent",
        "aget",
        "genoten",
        "augmente",
        "gente",
        "agena",
        "estination",
        "iettivo",
        "arget",
        "zele",
        "zile",
        "hede",
        "taget",
        "gals",
        "iele",
        "edef",
        "ARGET",
        "achable",
        "Objeto",
        "Destino",
        "objeto",
        "squareup",
        "cellspacing",
        "Cellular",
        "cellular",
        "squared",
        "Squared",
    },
    # ---- pure noise: other-script tokens with no grid reading --------------------------
    "foreign_noise": {
        "полиция",
        "colonia",
        "exposición",
        "oposición",
        "Coordinator",
        "coordinator",
        "coordinated",
        "срока",
        "сне",
        "грив",
        "развіц",
        "wykon",
        "struč",
        "tärke",
        "tungaanut",
        "конеч",
        "стандар",
        "Zitat",
        "جدول",
        "garis",
        "خط",
        "קו",
        "חלט",
        "ძი",
        "ાળો",
        "ილის",
        "քներ",
        "քները",
        "հերթ",
        "ಕೋಟ",
        "ுகள",
        "sovere",
        "cavity",
        "spæ",
        "адкры",
        "тракт",
        "остоя",
        "сятся",
        "сып",
        "очник",
        "реді",
        "Сп",
        "sıra",
        "tuyến",
        "мур",
        "总代理",
        "做代理",
        "平台代理",
        "总代理联系",
        "当前位置",
        "აგენტ",
        "ա",
        "不能提现",
        "不了怎么办",
        "不了了",
        "不上",
        "ไม่ได้",
        "领域",
        "平方",
        "Tunnel",
        "pavement",
        "terrein",
    },
    # ---- the direction vocabulary's own words, which must not be in this one -----------
    # Keeping them makes grid loudness and direction loudness non-independent, and the
    # whole point of two signals is that they can be compared.
    "direction_contamination_ci": {"directional", "Directional"},
}

# Tokens kept on purpose that a reader will want to query. Not applied -- documentation.
DELIBERATELY_KEPT = {
    "A": "the agent glyph in the grid legend (uppercase only; bare 'a' is pruned)",
    "G": "the goal glyph in the grid legend (uppercase only; bare 'g' is pruned)",
    "square": "misfiled under GOAL rather than OPEN, but a real cell word -- reclassify, do not drop",
    "walkway": "from the 'walkable tile' anchor; a real open-cell reading the model never emits",
}


def build_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """Split RULES into a case-sensitive and a case-folded form -> reason map."""
    exact: dict[str, str] = {}
    folded: dict[str, str] = {}
    for reason, forms in RULES.items():
        target, key = (folded, str.lower) if reason.endswith("_ci") else (exact, str)
        for form in forms:
            target.setdefault(key(form), reason)
    return exact, folded


def verdict(token: str, exact: dict[str, str], folded: dict[str, str]) -> str | None:
    """The rule that drops `token`, or None to keep it.

    >>> e, f = build_lookup()
    >>> verdict(" cannot", e, f)
    'generic_modality_ci'
    >>> verdict(" A", e, f) is None, verdict(" a", e, f)
    (True, 'bare_lowercase_letter')
    >>> verdict(" row", e, f) is None
    True
    """
    form = bare(token)
    if not form:
        return None
    return exact.get(form) or folded.get(form.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--write", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args()

    vocab: dict[str, list[str]] = json.loads(args.src.read_text(encoding="utf-8"))
    exact, folded = build_lookup()

    pruned: dict[str, list[str]] = {}
    rows: list[tuple[str, str, str, str]] = []
    dropped: Counter[str] = Counter()
    for klass, tokens in vocab.items():
        keep = []
        for tok in tokens:
            why = verdict(tok, exact, folded)
            rows.append((tok, klass, "drop" if why else "keep", why or ""))
            if why:
                dropped[why] += 1
            else:
                keep.append(tok)
        pruned[klass] = keep

    n_in = sum(len(v) for v in vocab.values())
    n_out = sum(len(v) for v in pruned.values())
    print(f"{args.src}\n  {n_in} tokens in -> {n_out} kept, {n_in - n_out} dropped\n")
    print(f"  {'class':8s} {'in':>5s} {'kept':>5s} {'dropped':>8s}")
    for klass, tokens in vocab.items():
        gone = len(tokens) - len(pruned[klass])
        print(f"  {klass:8s} {len(tokens):5d} {len(pruned[klass]):5d} {gone:8d}")
    print("\n  by rule:")
    for reason, n in dropped.most_common():
        print(f"    {reason:32s} {n:4d}")

    if not args.write:
        print("\n  dry run -- pass --write to emit the pruned vocabulary and the report")
        return

    args.out.write_text(json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.report, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "class", "verdict", "rule"])
        w.writerows(rows)
    print(f"\n  wrote {args.out}\n  wrote {args.report}")


if __name__ == "__main__":
    main()
