"""
Local fighter database access.

Loads data/fighters.json once at import time, exposes fuzzy-name lookup, and
computes per-weight-class percentiles for each numeric stat. The percentiles
are injected into the Stage 1 prompt so the LLM doesn't have to guess whether
a given number is "good" — it's told whether the number is elite or weak
relative to the fighter's division.

V4 adds build_closeness_context(), which deterministically measures how
similar two fighters are across all stat categories and returns a calibration
hint so Stage 2 doesn't have to guess confidence level.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process


DATA_PATH = os.path.join(os.path.dirname(__file__), "fighters.json")

# Stats we percentile-rank. Higher is better for all of these except sapm.
PERCENTILE_STATS = [
    "slpm", "str_acc_pct", "str_def_pct",
    "td_avg_per15", "td_acc_pct", "td_def_pct",
    "sub_avg_per15",
]
LOWER_IS_BETTER = {"sapm"}


def _load_raw() -> Tuple[List[dict], str]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["fighters"], data.get("snapshot_note", "")


_FIGHTERS, SNAPSHOT_NOTE = _load_raw()
_BY_NAME: Dict[str, dict] = {f["name"].lower(): f for f in _FIGHTERS}


def _percentile_rank(value: float, population: List[float], lower_is_better: bool = False) -> int:
    """Return the percentile rank of `value` within `population` (0-100, ints).

    A value at the top of the distribution gets 100; bottom gets 0.
    For `lower_is_better` stats (e.g., strikes absorbed), the scale is inverted
    so the percentile still reads as "higher = better fighter."
    """
    if not population:
        return 50
    below = sum(1 for v in population if v < value)
    equal = sum(1 for v in population if v == value)
    rank = (below + 0.5 * equal) / len(population)
    pct = int(round(rank * 100))
    if lower_is_better:
        pct = 100 - pct
    return max(0, min(100, pct))


def _pct_to_tier(pct: int) -> str:
    if pct >= 85:
        return "elite"
    if pct >= 65:
        return "above-average"
    if pct >= 35:
        return "average"
    if pct >= 15:
        return "below-average"
    return "weak"


def get_fighter_names() -> List[str]:
    """All fighter names. Used by /api/fighters for autocomplete."""
    return sorted(f["name"] for f in _FIGHTERS)


def search_names(query: str, limit: int = 8) -> List[str]:
    """Autocomplete suggestions.

    Substring matches come first (the user is usually typing a real name);
    fuzzy matches are appended only if we still have room, to catch typos
    like "makachev" -> "Makhachev".
    """
    if not query or not query.strip():
        return []
    q = query.strip().lower()
    all_names = [f["name"] for f in _FIGHTERS]

    substring = [n for n in all_names if q in n.lower()]
    if len(substring) >= limit:
        return substring[:limit]

    fuzzy = process.extract(query, all_names, scorer=fuzz.WRatio, limit=limit * 2)
    extras = [name for name, score, _ in fuzzy if score >= 72 and name not in substring]
    return (substring + extras)[:limit]


def resolve_fighter(name: str) -> Tuple[Optional[dict], List[str]]:
    """Look up a fighter by name with fuzzy fallback.

    Returns (fighter_dict, suggestions). If found, suggestions is empty. If
    not found, fighter_dict is None and suggestions contains up to 5 plausible
    alternatives so the UI can show "did you mean...?".
    """
    if not name:
        return None, []
    key = name.lower().strip()
    if key in _BY_NAME:
        return _BY_NAME[key], []

    # Fuzzy fallback. A high score means we can treat it as a direct hit.
    matches = process.extract(
        name, [f["name"] for f in _FIGHTERS],
        scorer=fuzz.WRatio, limit=5,
    )
    if matches and matches[0][1] >= 90:
        return _BY_NAME[matches[0][0].lower()], []
    suggestions = [m[0] for m in matches if m[1] >= 60]
    return None, suggestions


def build_percentile_context(fighter: dict) -> Dict[str, dict]:
    """For each percentile stat, return {value, percentile, tier} computed
    against fighters in the same weight class.

    This is what makes the Stage 1 prompt deterministic — the LLM never has to
    decide "is 65% takedown defense good?" because we've already told it that
    65% is, say, the 30th percentile at lightweight (below-average).
    """
    wc = fighter["weight_class"]
    peers = [f for f in _FIGHTERS if f["weight_class"] == wc]
    out: Dict[str, dict] = {}

    for stat in PERCENTILE_STATS:
        peer_values = [f[stat] for f in peers]
        value = fighter[stat]
        pct = _percentile_rank(value, peer_values, lower_is_better=False)
        out[stat] = {"value": value, "percentile": pct, "tier": _pct_to_tier(pct)}

    sapm_peers = [f["sapm"] for f in peers]
    sapm_pct = _percentile_rank(fighter["sapm"], sapm_peers, lower_is_better=True)
    out["sapm"] = {"value": fighter["sapm"], "percentile": sapm_pct, "tier": _pct_to_tier(sapm_pct)}

    return out


# ---------------------------------------------------------------------------
# V4: Closeness context for calibration
# ---------------------------------------------------------------------------

# Stats grouped into the four matchup categories. Each group's percentile
# gap is averaged to get a per-category closeness score.
_CATEGORY_STATS = {
    "striking": ["slpm", "str_acc_pct", "sapm", "str_def_pct"],
    "grappling": ["td_avg_per15", "td_acc_pct", "td_def_pct", "sub_avg_per15"],
    "cardio_durability": ["sapm", "str_def_pct"],
    "physical": ["slpm", "td_avg_per15"],
}


def _closeness_label(mean_gap: float) -> str:
    """Convert a mean percentile gap (0-100) into a human-readable label."""
    if mean_gap <= 10:
        return "very close"
    if mean_gap <= 25:
        return "competitive"
    if mean_gap <= 45:
        return "clear edge"
    return "dominant"


def build_closeness_context(ctx_a: Dict[str, dict], ctx_b: Dict[str, dict],
                            spread: bool = False) -> dict:
    """Deterministically compute how close two fighters are across all
    stat categories. Returns a dict with:
      - overall_closeness: label (very close / competitive / clear edge / dominant)
      - overall_gap: mean percentile gap across all shared stats (0-100)
      - recommended_confidence: low / medium / high
      - category_gaps: per-category gap labels for the four matchup categories
      - note: plain-English instruction for the LLM

    The LLM is instructed to treat recommended_confidence as a strong prior
    and only override it if the profiles contain decisive qualitative evidence
    (e.g., a fighter is on a 5-fight KO streak against elite competition).
    """
    all_stats = list(ctx_a.keys())
    gaps = [abs(ctx_a[s]["percentile"] - ctx_b[s]["percentile"]) for s in all_stats]
    overall_gap = sum(gaps) / len(gaps) if gaps else 0

    category_gaps: Dict[str, str] = {}
    for cat, stats in _CATEGORY_STATS.items():
        shared = [s for s in stats if s in ctx_a and s in ctx_b]
        if shared:
            cat_gap = sum(abs(ctx_a[s]["percentile"] - ctx_b[s]["percentile"])
                          for s in shared) / len(shared)
            category_gaps[cat] = _closeness_label(cat_gap)
        else:
            category_gaps[cat] = "competitive"

    # Recommended confidence: driven by overall gap with a slight boost
    # if multiple categories are decisive.
    decisive_cats = sum(1 for v in category_gaps.values() if v in ("clear edge", "dominant"))
    if spread:
        # V5 banding. The V4 logic below contains a bug - both arms of its
        # middle branch return "medium" - and since most matchups land in
        # that band, V4 could almost never emit anything but "medium". That
        # is why REPORT.md records V4 calibration as no better than V3.
        # Kept intact under `spread=False` so the V1-V4 sweep still reproduces.
        if overall_gap <= 10:
            rec_conf = "low"
        elif overall_gap <= 22:
            rec_conf = "low" if decisive_cats == 0 else "medium"
        elif overall_gap <= 34:
            rec_conf = "high" if decisive_cats >= 2 else "medium"
        else:
            rec_conf = "high"
    elif overall_gap <= 12:
        rec_conf = "low"
    elif overall_gap <= 28:
        rec_conf = "medium" if decisive_cats < 2 else "medium"
    else:
        rec_conf = "high" if decisive_cats >= 2 else "medium"

    note = (
        f"Deterministic closeness analysis: overall mean percentile gap = "
        f"{overall_gap:.1f} ({_closeness_label(overall_gap)}). "
        f"Recommended confidence: {rec_conf}. "
        f"Treat this as a strong prior — only override to 'high' if the "
        f"profiles show decisive qualitative advantages (dominant finishing "
        f"rate, elite chin vs. known KO losses, etc.). "
        f"Only override to 'low' if there are significant x-factors "
        f"(ring rust, injury history, weight class move) not captured in stats."
    )

    return {
        "overall_closeness": _closeness_label(overall_gap),
        "overall_gap": round(overall_gap, 1),
        "recommended_confidence": rec_conf,
        "category_gaps": category_gaps,
        "note": note,
    }


# ---------------------------------------------------------------------------
# V5: Sample-size shrinkage, physical context, statistical favorite
# ---------------------------------------------------------------------------
#
# Motivated by the UFC 331 miss analysis (eval/ufc331_predictions.json).
# Two of three wrong picks were fighters whose per-minute rates were computed
# over a handful of first-round finishes: Gable Steveson graded "elite
# striking" off 16.29 SLpM across 4 fights, Iwo Baraniewski off 11.35 SLpM
# across 9. Both got beaten by more experienced opponents. Raw rate stats over
# tiny samples are noise, but the percentile ranking treated them as signal.

CONFIDENCE_PRIOR = 8.0  # pseudo-fights; controls how hard thin samples regress


def fight_count(fighter: dict) -> int:
    """Total professional bouts parsed from the 'W-L-D' record string."""
    try:
        return sum(int(p) for p in fighter["record"].split("-")[:3])
    except (ValueError, KeyError):
        return 0


def reliability_label(n: int) -> str:
    if n >= 20:
        return "high"
    if n >= 12:
        return "moderate"
    if n >= 7:
        return "low"
    return "very low"


def _compress(pct: int, n: int, k: float = CONFIDENCE_PRIOR) -> int:
    """Regress a percentile toward 50 in proportion to sample size.

    weight = n / (n + k). A 4-fight fighter keeps a third of their distance
    from the median; a 29-fight veteran keeps nearly four fifths. This is
    symmetric on purpose: with few fights we do not know a fighter is elite,
    and we equally do not know they are terrible.

    Shrinking the raw values instead does not work - the peer population
    shrinks with them, so an outlier keeps its rank.
    """
    if n <= 0:
        return 50
    w = n / (n + k)
    return int(round(50 + (pct - 50) * w))


def build_percentile_context_shrunk(fighter: dict) -> Dict[str, dict]:
    """Like build_percentile_context, but each percentile is regressed toward
    the median according to how many fights it was computed over, and the
    tier label is re-derived from the regressed value.

    The tier is what the Stage 1 prompt is told to treat as ground truth, so
    capping it here is what actually stops a 4-fight record from being graded
    "elite". V5 only - V3/V4 behaviour is untouched.
    """
    raw = build_percentile_context(fighter)
    n = fight_count(fighter)
    out: Dict[str, dict] = {}
    for stat, entry in raw.items():
        pct = _compress(entry["percentile"], n)
        out[stat] = {
            "value": entry["value"],
            "percentile": pct,
            "raw_percentile": entry["percentile"],
            "tier": _pct_to_tier(pct),
        }
    out["_sample"] = {"fights": n, "reliability": reliability_label(n)}
    return out


def build_physical_context(fighter: dict) -> dict:
    """Physical attributes for Stage 2.

    Stage 2 is asked to judge a 'physical' advantage category but the
    FighterProfile schema carries no height, reach, age or stance, so before
    V5 that category was decided with no physical data at all. Despaigne beat
    Tuivasa at UFC 331 with a nine-inch reach edge that never entered the
    reasoning.
    """
    return {
        "name": fighter["name"],
        "height_in": fighter["height_in"],
        "reach_in": fighter["reach_in"],
        "age": fighter["age"],
        "stance": fighter.get("stance", "Unknown"),
        "fights": fight_count(fighter),
    }


def build_statistical_favorite(ctx_a: Dict[str, dict], ctx_b: Dict[str, dict],
                               name_a: str, name_b: str) -> dict:
    """Which fighter leads on more percentile categories, and by how much.

    Handed to Stage 2 so that picking the other fighter becomes an explicit,
    justified deviation rather than an unnoticed one.
    """
    stats = [s for s in PERCENTILE_STATS + ["sapm"] if s in ctx_a and s in ctx_b]
    a_wins = sum(1 for s in stats if ctx_a[s]["percentile"] > ctx_b[s]["percentile"])
    b_wins = sum(1 for s in stats if ctx_b[s]["percentile"] > ctx_a[s]["percentile"])
    mean_a = sum(ctx_a[s]["percentile"] for s in stats) / len(stats)
    mean_b = sum(ctx_b[s]["percentile"] for s in stats) / len(stats)

    if abs(mean_a - mean_b) < 4:
        favorite = "neither (statistically even)"
    else:
        favorite = name_a if mean_a > mean_b else name_b

    return {
        "favorite": favorite,
        "category_wins": {name_a: a_wins, name_b: b_wins},
        "mean_percentile": {name_a: round(mean_a, 1), name_b: round(mean_b, 1)},
    }
