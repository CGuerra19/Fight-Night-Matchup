"""
Stage 2: head-to-head matchup analysis.

Consumes two FighterProfile objects (NOT raw stats) and produces a
MatchupAnalysis. The point of the two-stage design is that Stage 2 reasons
over a normalized vocabulary — style labels, grades, strengths, weaknesses —
instead of trying to compare 16 numeric stats in one shot.

V4 adds an optional closeness_context dict (from build_closeness_context)
that injects a deterministic recommended_confidence into the prompt so the
model doesn't have to guess calibration from scratch.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from llm.schemas import FighterProfile, MatchupAnalysis


MATCHUP_MODEL = os.environ.get("MATCHUP_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = """You are an MMA analyst writing a head-to-head matchup
breakdown for two UFC fighters. Both fighters have already been distilled
into structured FighterProfile objects by a prior step.

You will produce a MatchupAnalysis with four parts:

1. advantages[] — exactly four entries, one per category:
     striking, grappling, cardio_durability, physical
   For each, name the favored fighter (or "even"), pick a magnitude
   (even/slight/clear/decisive), and give a one-sentence reason that
   references the profiles' grades, style_label, strengths, or weaknesses.

2. stylistic_clash — 2-3 sentences on how the two styles interact. Examples:
   "A pressure wrestler against a counter-striker tends to ..." Speak in
   terms of style_label, not raw stats.

3. key_factors — 3-5 short bullets naming the variables most likely to
   decide this specific fight (range management, takedown threat, finish
   rate, cardio, ring rust, etc.). Each bullet should connect to something
   in the profiles.

4. predicted_winner + confidence + prediction_rationale — pick a winner by
   name (must match one of the two input names exactly), choose a confidence
   level, and write 2-3 sentences justifying the pick using the advantages
   and stylistic_clash you already named. Confidence should be "low" when
   the matchup is close or has many x-factors, "high" only when one fighter
   has decisive advantages in multiple categories.

Hard rules:
  - Do not invent stats, fights, or biographical info not in the profiles.
  - The predicted_winner name MUST be one of the two fighters provided.
  - Stay grounded in the profile fields. If neither profile mentions a
    "glass chin", do not claim one fighter has one.
  - If a CALIBRATION HINT is provided below the fighter profiles, treat the
    recommended_confidence as a strong prior. Only override it if the profiles
    contain clear qualitative evidence that justifies a different level.
"""


def _render_profile(profile: FighterProfile, label: str) -> str:
    lines = [
        f"=== {label}: {profile.name} ===",
        f"Weight class: {profile.weight_class}",
        f"Record: {profile.record}",
        f"Style: {profile.style_label}",
        f"Grades  —  striking: {profile.striking_grade}, "
        f"grappling: {profile.grappling_grade}, "
        f"cardio/durability: {profile.cardio_durability_grade}",
        f"Recent form: {profile.recent_form}",
        "Top strengths:",
    ]
    for s in profile.top_strengths:
        lines.append(f"  - {s}")
    lines.append("Top weaknesses:")
    for w in profile.top_weaknesses:
        lines.append(f"  - {w}")
    if profile.x_factors:
        lines.append("X-factors:")
        for x in profile.x_factors:
            lines.append(f"  - {x}")
    return "\n".join(lines)


def generate_matchup(profile_a: FighterProfile, profile_b: FighterProfile,
                     closeness_context: Optional[dict] = None) -> MatchupAnalysis:
    """Call the LLM and return a validated MatchupAnalysis.

    Args:
        profile_a: Stage 1 profile for fighter A.
        profile_b: Stage 1 profile for fighter B.
        closeness_context: Optional dict from build_closeness_context().
            When provided (V4+), injects a deterministic calibration hint
            so the model doesn't have to guess confidence level from scratch.
    """
    client = OpenAI()
    user_prompt = (
        _render_profile(profile_a, "FIGHTER A") + "\n\n"
        + _render_profile(profile_b, "FIGHTER B") + "\n\n"
        + "Produce the MatchupAnalysis now. "
        + f"predicted_winner must be exactly '{profile_a.name}' or '{profile_b.name}'."
    )

    if closeness_context:
        user_prompt += (
            "\n\n--- CALIBRATION HINT (deterministic, treat as ground truth) ---\n"
            + closeness_context["note"]
            + f"\nCategory closeness: {closeness_context['category_gaps']}"
        )

    completion = client.beta.chat.completions.parse(
        model=MATCHUP_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=MatchupAnalysis,
        temperature=0.3,
    )
    msg = completion.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"Model refused to produce matchup: {msg.refusal}")
    analysis = msg.parsed
    if analysis is None:
        raise RuntimeError(
            f"Stage 2 returned unparseable content: {msg.content!r}"
        )

    # Sanity: predicted_winner must be one of the two fighters. If the model
    # paraphrased ("Tom" instead of "Tom Aspinall"), snap it back.
    valid_names = {profile_a.name, profile_b.name}
    if analysis.predicted_winner not in valid_names:
        a_match = profile_a.name.lower().startswith(analysis.predicted_winner.lower())
        b_match = profile_b.name.lower().startswith(analysis.predicted_winner.lower())
        if a_match and not b_match:
            analysis.predicted_winner = profile_a.name
        elif b_match and not a_match:
            analysis.predicted_winner = profile_b.name
        else:
            # Last resort: pick by predicted confidence default
            analysis.predicted_winner = profile_a.name

    return analysis
