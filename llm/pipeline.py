"""
Pipeline dispatcher.

PIPELINE_VERSION env var selects which version runs end-to-end:
  - v1: single LLM call producing the full MatchupResponse from raw stats.
        The Project-4-style baseline; expected to be the weakest.
  - v2: two-stage (distill -> matchup) but Stage 1 gets RAW stats only,
        no percentile context. Tests whether structuring alone is enough.
  - v3: two-stage WITH deterministic percentile/tier context injected into
        Stage 1 (the default; the version described in REPORT.md).
  - v4: v3 PLUS deterministic closeness context injected into Stage 2.
        Gives the model a calibration hint (recommended_confidence) so it
        doesn't have to guess confidence level from scratch.
        See REPORT.md for results — calibration did not improve as expected.
  - v5: v4 plus three fixes derived from the UFC 331 post-mortem (7/10),
        all deterministic, so cost per matchup is unchanged at three LLM
        calls. (1) Stage 1 percentiles are regressed toward the divisional
        median by sample size, so a 4-fight record can no longer be graded
        "elite". (2) Stage 2 receives height/reach/age/stance, which it
        previously never saw despite being asked to judge a "physical"
        advantage. (3) Stage 2 receives the aggregate statistical favorite
        and must justify picking against it. Also enables the corrected
        confidence banding (see build_closeness_context's `spread` flag).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from data.load_fighters import (build_closeness_context, build_percentile_context,
                                build_percentile_context_shrunk, build_physical_context,
                                build_statistical_favorite)
from llm.matchup import generate_matchup
from llm.profile import distill_profile
from llm.schemas import MatchupResponse
from llm.v1_single_prompt import single_shot_matchup


VERSION = os.environ.get("PIPELINE_VERSION", "v5").lower()


def run_pipeline(fighter_a: dict, fighter_b: dict) -> MatchupResponse:
    if VERSION == "v1":
        return single_shot_matchup(fighter_a, fighter_b)

    use_percentiles = VERSION in ("v3", "v4", "v5")
    v5 = VERSION == "v5"

    if v5:
        ctx_a = build_percentile_context_shrunk(fighter_a)
        ctx_b = build_percentile_context_shrunk(fighter_b)
    else:
        ctx_a = build_percentile_context(fighter_a)
        ctx_b = build_percentile_context(fighter_b)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(distill_profile, fighter_a, ctx_a, use_percentiles)
        fut_b = pool.submit(distill_profile, fighter_b, ctx_b, use_percentiles)
        profile_a = fut_a.result()
        profile_b = fut_b.result()

    if VERSION in ("v4", "v5"):
        # `_sample` is V5 bookkeeping, not a stat; keep it out of the gap math.
        gap_a = {k: v for k, v in ctx_a.items() if k != "_sample"}
        gap_b = {k: v for k, v in ctx_b.items() if k != "_sample"}
        closeness = build_closeness_context(gap_a, gap_b, spread=v5)
    else:
        closeness = None

    physical = favorite = None
    if v5:
        physical = (build_physical_context(fighter_a), build_physical_context(fighter_b))
        favorite = build_statistical_favorite(gap_a, gap_b,
                                              fighter_a["name"], fighter_b["name"])

    analysis = generate_matchup(profile_a, profile_b, closeness_context=closeness,
                                physical=physical, statistical_favorite=favorite)
    return MatchupResponse(profile_a=profile_a, profile_b=profile_b, matchup=analysis)
