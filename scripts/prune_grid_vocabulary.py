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

**One rule set, both models.** `--in` also takes the Qwen twin
`data/grid_tokens_full_qwen3-6-35b-a3b.json` (1316 tokens), and `--out`/`--report` are derived
from it, so the MODEL_SLUG survives into the products instead of a hardcoded default writing a
Qwen prune over the gpt-oss one. The rules stay shared on purpose: `MODEL_ID` is the only thing
that differs between the two notebooks, so a shared rule set keeps the only difference between
the two *pruned* files the model as well. Every rule below that the Qwen file motivated is a
judgement about the string, not about Qwen.

    gpt-oss-20b     1258 -> 633
    Qwen3.6-35B-A3B 1316 -> 533

The Qwen tokenizer carries far more code identifiers, which is what `code_api_compound` and
`difflib_objectif_object_ci` are for: ` GameObject`, `.DataGridViewTextBoxColumn` and
`.isNullOrEmpty` each contain a real grid word and none is a word a model writes in prose.

Dry-run by default, like the other destructive scripts here; pass --write to emit files.

    python scripts/prune_grid_vocabulary.py
    python scripts/prune_grid_vocabulary.py --write
    python scripts/prune_grid_vocabulary.py --in data/grid_tokens_full_qwen3-6-35b-a3b.json --write
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

SEP_RE = re.compile(r"""[_\-./\\:,;()\[\]{}<>"'`|!?*+=#@$%^&~\s]+""")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "data" / "jlens" / "grid_tokens_full.json"


def derived_paths(src: Path) -> tuple[Path, Path]:
    """Output and report beside `src`, named off it, so --in alone picks a vocabulary.

    The Qwen twin carries a MODEL_SLUG that has to survive into its products, and hardcoded
    defaults silently wrote a Qwen prune over the gpt-oss one.

    >>> out, rep = derived_paths(Path("data/grid_tokens_full_qwen3-6-35b-a3b.json"))
    >>> out.name, rep.name
    ('grid_tokens_pruned_qwen3-6-35b-a3b.json', 'grid_tokens_prune_report_qwen3-6-35b-a3b.csv')
    >>> [p.name for p in derived_paths(Path("data/jlens/grid_tokens_full.json"))]
    ['grid_tokens_pruned.json', 'grid_tokens_prune_report.csv']
    """
    stem = src.stem
    return (
        src.with_name(stem.replace("_full", "_pruned", 1) + ".json"),
        src.with_name(stem.replace("_full", "_prune_report", 1) + ".csv"),
    )


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
    "bare_lowercase_letter": {"a", "g", "gg", "gv", "gw", "gha", "ġ", "ａ", "gz"},
    # '\u2019a' / '\u201cA' / '\u2019an' carry a curly quote, which SEP_RE does not strip, so they
    # survive bare() as their own forms and have to be named.
    "english_article": {"an", "An", "AN", "\u2019an", "\u2014an", "aN", "\u00e0n"},
    # letter bigrams difflib pulled in beside 'G'
    "letter_bigram": {
        "GF",
        "GC",
        "Gh",
        "Gy",
        "Gl",
        "Gol",
        "Glob",
        "GLES",
        "Ｇ",
        "Â",
        "Ã",
        "â",
        "ª",
        "ｇ",
        "Ğ",
        "ГК",
        "ГҐ",
        "Glo",
        "GGLE",
        "ã",
        "à",
        "ân",
        "ânt",
        "’a",
        "“A",
        "’A",
    },
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
        "unsustainable",
        "untranslated",
        "unstoppable",
        "uninterrupted",
        "unwitting",
        "unrecognized",
        "unsigned",
        "undefined",
        "unistd",
        "restricted",
    },
    # ---- code senses of pass / through / block ---------------------------------------
    # 'blocked'/'blocking'/'blockage'/'blockade' keep the grid sense; the bare stems are
    # control-flow and HTML words the lens emits constantly.
    "code_control_flow_ci": {
        "pass",
        "Passing",
        "through",
        "block",
        "BLOCK",
        "blockquote",
        "undable",
        "blocks",
        "blockly",
        "블록체인",
        "ブロック",
    },
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
    # 'unused' is the programming sense; the rest are abstract emptiness, not an empty cell.
    "anchor_drift_free_ci": {
        "เงินฟรี",
        "бесплат",
        "spotless",
        "barren",
        "deserted",
        "vacant",
        "nothing",
        "devoid",
        "vacancy",
        "abandoned",
        "unused",
        "空虚",
        "虚空",
        "беско",
        "ร้าง",
    },
    # ---- the "index" family: a programming word, not a grid word ----------------------
    # The model says 'row' and 'col'; over 268k sampled reasoning tokens it says ' index'
    # 123 times and 'indexes' never.
    "code_index_ci": {"index", "indexes", "indexed", "indexing", "Indexer", "Index", "Indexes", "Indexed"},
    # ---- the English "line" family ----------------------------------------------------
    # 'ligne'/'linha'/'línea'/'Reihe'/'rij'/'række' are the real row-words in other
    # languages and stay. English 'line' is not a word this model uses for a grid row.
    "english_line_ci": {
        "line",
        "线",
        "線",
        "ライン",
        "lijn",
        "linien",
        "लाइन",
        "লাইন",
        "lineto",
        "线条",
        "线的",
        "線的",
        "線の",
        "線で",
        "線を",
        "划线",
        "เส้น",
    },
    # ---- difflib bridging 'col'/'colonne' to 'colon' -----------------------------------
    # 'colonne'/'columna' also bridge to Colonel / Cologne / colonia / columnist.
    "difflib_colon_ci": {"colon", "kolon", "colo", "colonel", "cologne", "colonia", "columnist"},
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
        "纬度",
        "几何",
        "геометри",
        "geometr",
        "ometry",
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
        "trip",
        "viaggi",
        "السفر",
        "旅游目的地",
        "여행을",
        "destiny",
        "进球",
        "goalt",
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
        "sfide",
        "挑戦",
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
        "carrier",
        "pare",
        "pard",
        "paed",
        "parted",
        "paired",
        "arriver",
        "ared",
        "тена",
        "сторо",
        "obsta",
        "преки",
        "блица",
        "indép",
        "murn",
        "spra",
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
        "licant",
        "overt",
        "offend",
        "abiert",
        "empt",
        "liber",
        "toffen",
        "ouverture",
        "agements",
        "ragments",
        "gent",
        "argent",
        "ージェント",
        "jectif",
        "estino",
        "destinati",
        "designation",
        "estimation",
        "ціаль",
        "мети",
        "glob",
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
        "тракта",
        "委托代理人",
        "委托诉讼代理人",
        "代理机构",
        "代理服务",
        "經紀",
    },
    # ---- difflib bridging the GOAL seeds objectif/objetivo/obiettivo to "object" -------
    # 24 tokens in the Qwen vocabulary, before the camelCase forms (GameObject, PyObject,
    # ObjectMapper, ...) that `code_api_compound` takes. The programming "object" is the
    # single largest junk family there, and it is a character neighbour, not a meaning.
    "difflib_objectif_object_ci": {"object", "objeto", "objet", "objekt", "hobject"},
    # ---- difflib bridging the OPEN seeds libre/livre to "live" / "lire" ----------------
    # ' live' and its ten punctuation arms, plus the French "to read" and the two
    # place names the embedder pulled in beside them.
    "difflib_libre_live_ci": {"live", "lire", "alberta", "alberto"},
    # ---- difflib bridging position/colonne/fila/ligne to their character neighbours ----
    # The -osition rhymes (opposition, imposition, deposition, exposition) and the Italian
    # -sizione forms are not row or column words in any language.
    "difflib_axis_neighbour_ci": {
        "opposition",
        "imposition",
        "deposition",
        "exposition",
        "esposizione",
        "composizione",
        "disposizione",
        "posizion",
        "potion",
        "ordinate",
        "lign",
        "colum",
        "lina",
        "flas",
        "fils",
    },
    # ---- a queue is not a grid row -----------------------------------------------------
    "queue_drift_ci": {"队列", "排隊", "очереди", "очередь"},
    # ---- the direction vocabulary's own words, which must not be in this one -----------
    # Keeping them makes grid loudness and direction loudness non-independent, and the
    # whole point of two signals is that they can be compared.
    "direction_contamination_ci": {"directional", "Directional"},
}

# --------------------------------------------------------------------------------------
# Two rules that cannot be written as a set, because what they match is a *shape*: they are
# open-ended families that a word list would have to chase token by token. Both are matched
# against the raw token, before bare() -- `bare()` strips the delimiters they are defined by.
#
# They drop nothing from the gpt-oss vocabulary (no camelCase identifier and no special token
# survives the rules above there), and 221 tokens from the Qwen twin, whose tokenizer carries
# far more code identifiers. One rule set still describes both files.
# --------------------------------------------------------------------------------------

CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def is_code_api_compound(token: str) -> bool:
    """A camelCase or PascalCase identifier: a method or class name, not a word.

    The grid seeds sit inside hundreds of them -- ` GameObject`, `.DataGridViewTextBoxColumn`,
    `.isNullOrEmpty`, ` cellForRowAtIndexPath`. Each contains a real grid word, and none is a
    word the model writes when it reasons about a grid in prose.

    >>> is_code_api_compound(" GameObject"), is_code_api_compound("getRow")
    (True, True)
    >>> is_code_api_compound(" column"), is_code_api_compound("ROW"), is_code_api_compound("aN")
    (False, False, False)
    """
    form = bare(token)
    if len(form) < 5:
        return False
    parts = CAMEL_RE.split(form)
    return len(parts) > 1 and all(len(p) >= 2 for p in parts)


def is_special_token(token: str) -> bool:
    """A `<|...|>` control token. Qwen has two that carry the string "object".

    >>> is_special_token("<|object_ref_start|>"), is_special_token(" object")
    (True, False)
    """
    stripped = token.strip()
    return stripped.startswith("<|") and stripped.endswith("|>")


PREDICATE_RULES: dict[str, Callable[[str], bool]] = {
    "code_api_compound": is_code_api_compound,
    "special_token": is_special_token,
}

# Tokens kept on purpose that a reader will want to query. Not applied -- documentation.
DELIBERATELY_KEPT = {
    "A": "the agent glyph in the grid legend (uppercase only; bare 'a' is pruned)",
    "G": "the goal glyph in the grid legend (uppercase only; bare 'g' is pruned)",
    "square": "misfiled under GOAL rather than OPEN, but a real cell word -- reclassify, do not drop",
    "walkway": "from the 'walkable tile' anchor; a real open-cell reading the model never emits",
    "locked": "arrived by difflib off 'blocked', but D/K are in the legend, so a door state is a grid state",
    "accessible": "the positive half of the passability sense STATUS is meant to carry",
    "#": "the wall glyph; SEP_RE eats it, so neither notebook stage can score it and bare() leaves it empty",
    "_": "the open-cell glyph, same reason -- both are appended by hand in cell 10 of the notebook",
    "Ａ": "the fullwidth agent glyph. Note the fullwidth 'Ｇ' is dropped by letter_bigram, which"
    " predates this and is left alone rather than silently changing the gpt-oss prune",
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
    >>> verdict(" GameObject", e, f)
    'code_api_compound'
    """
    form = bare(token)
    if not form:
        return None
    hit = exact.get(form) or folded.get(form.lower())
    if hit:
        return hit
    return next((name for name, matches in PREDICATE_RULES.items() if matches(token)), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=None, help="default: derived from --in")
    ap.add_argument("--report", type=Path, default=None, help="default: derived from --in")
    ap.add_argument("--write", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args()
    default_out, default_report = derived_paths(args.src)
    args.out = args.out or default_out
    args.report = args.report or default_report

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
