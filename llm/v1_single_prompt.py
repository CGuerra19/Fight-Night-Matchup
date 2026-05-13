"""
V1 baseline pipeline — single LLM call.

This is the "Project 4 pattern" that the assignment asks us to surpass:
hand raw stats for both fighters to the model in one prompt and force it
to produce the full MatchupResponse (profiles + matchup) in a single
structured output.

V1 is kept around so the eval harness can measure how much the multi-stage
V3 pipeline actually improves things.
"""

from __future__ import annotations

import os

from openai import OpenAI

from llm.schemas import MatchupResponse


V1_MODEL = os.environ.get("V1_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = """You are an MMA analyst. You will be given raw career
statistics for two UFC fighters. In a single response, produce a
MatchupResponse that contains:
  - A FighterProfile for fighter A (style_label, grades, strengths,
    weaknesses, recent form, x_factors).
  - A FighterProfile for fighter B (same fields).
  - A MatchupAnalysis (category advantages, stylistic clash, key factors,
    predicted winner, confidence, rationale).

Rules:
  - Use only the data provided.
  - The predicted_winner must be the exact name of one of the two fighters.
  - Grades must use one of: elite, above-average, average, below-average, weak.
"""


def _render_raw(fighter: dict, label: str) -> str:
    lines = [
        f"=== {label}: {fighter['name']} ===",
        f"Weight class: {fighter['weight_class']}",
        f"Record: {fighter['record']}",
        f"Stance: {fighter['stance']}",
        f"Height: {fighter['height_in']} in   Reach: {fighter['reach_in']} in   Age: {fighter['age']}",
        f"SLpM: {fighter['slpm']}    Str.Acc: {fighter['str_acc_pct']}%",
        f"SApM: {fighter['sapm']}    Str.Def: {fighter['str_def_pct']}%",
        f"TD/15: {fighter['td_avg_per15']}   TD.Acc: {fighter['td_acc_pct']}%   "
        f"TD.Def: {fighter['td_def_pct']}%",
        f"Sub/15: {fighter['sub_avg_per15']}",
        "Last 5:"
    ]
    for fight in fighter["last_5"]:
        lines.append(f"  - {fight}")
    return "\n".join(lines)


def single_shot_matchup(fighter_a: dict, fighter_b: dict) -> MatchupResponse:
    client = OpenAI()
    user = (
        _render_raw(fighter_a, "FIGHTER A") + "\n\n"
        + _render_raw(fighter_b, "FIGHTER B") + "\n\n"
        + f"predicted_winner must be exactly '{fighter_a['name']}' "
        + f"or '{fighter_b['name']}'."
    )
    completion = client.beta.chat.completions.parse(
        model=V1_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=MatchupResponse,
        temperature=0.3,
    )
    msg = completion.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"V1 refused: {msg.refusal}")
    resp = msg.parsed
    if resp is None:
        raise RuntimeError(f"V1 returned unparseable content: {msg.content!r}")
    # Snap names back to canonical (the model sometimes paraphrases).
    resp.profile_a.name = fighter_a["name"]
    resp.profile_a.weight_class = fighter_a["weight_class"]
    resp.profile_a.record = fighter_a["record"]
    resp.profile_b.name = fighter_b["name"]
    resp.profile_b.weight_class = fighter_b["weight_class"]
    resp.profile_b.record = fighter_b["record"]
    if resp.matchup.predicted_winner not in {fighter_a["name"], fighter_b["name"]}:
        resp.matchup.predicted_winner = fighter_a["name"]
    return resp
