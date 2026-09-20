"""
Stage 1: distill a raw fighter record into a FighterProfile.

This is the step the instructor recommended: take raw stats + deterministically
computed percentile context, hand them to the LLM, and force JSON output
matching the FighterProfile schema. Everything downstream (Stage 2, UI, eval)
consumes only the resulting FighterProfile — never the raw numbers.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from llm.schemas import FighterProfile


# Model is overridable via env var so the eval harness can sweep versions.
PROFILE_MODEL = os.environ.get("PROFILE_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = """You are an MMA analyst who specializes in turning raw UFC
statistics into concise, predictable fighter profiles for downstream matchup
analysis.

You will be given:
  - A fighter's biographical/record info.
  - Their career striking, grappling, and durability stats.
  - A deterministically computed percentile + tier for each stat, ranked
    against other fighters in the same weight class. Treat these tiers as
    ground truth — do NOT contradict them. If a fighter's TD defense is
    labeled "elite", call that a strength.
  - Their last five fight results.

Produce a FighterProfile object that other downstream prompts can consume.
Hard rules:
  - Use only the data provided. Do not invent fights, finishes, or stats.
  - Strengths and weaknesses must be grounded in the tiered stats or recent
    form. "Elite chin" is fine if their losses are decisions, not KOs;
    otherwise avoid it.
  - style_label must be 2-4 words (e.g., "pressure wrestler", "volume kickboxer",
    "submission grappler", "counter-striker").
  - recent_form must reference the last_5 list — streak length, finish rate,
    or notable losses.
  - Be concise. Strengths/weaknesses are short phrases, not paragraphs.
"""


def _build_user_prompt(fighter: dict, percentile_context: Dict[str, dict],
                       use_percentiles: bool) -> str:
    """Render the fighter (optionally with percentile context) into a prompt.

    When `use_percentiles` is False (the V2 ablation), we hand the model
    raw numbers only — it has to decide on its own whether 65% TD defense
    is good or bad. When True (V3, default), we attach a deterministic tier
    label to every stat so the model never has to guess.
    """
    lines = [
        f"Fighter: {fighter['name']}",
        f"Nickname: {fighter.get('nickname') or '(none)'}",
        f"Weight class: {fighter['weight_class']}",
        f"Record: {fighter['record']}",
        f"Stance: {fighter['stance']}",
        f"Height: {fighter['height_in']} in   Reach: {fighter['reach_in']} in   Age: {fighter['age']}",
        "",
        ("Career stats (with percentile + tier vs. same-division peers):"
         if use_percentiles else "Career stats:"),
    ]
    stat_labels = {
        "slpm": "Significant strikes landed / min",
        "str_acc_pct": "Striking accuracy %",
        "sapm": "Significant strikes absorbed / min (lower=better)",
        "str_def_pct": "Striking defense %",
        "td_avg_per15": "Takedowns / 15 min",
        "td_acc_pct": "Takedown accuracy %",
        "td_def_pct": "Takedown defense %",
        "sub_avg_per15": "Submission attempts / 15 min",
    }
    for stat_key, label in stat_labels.items():
        if use_percentiles:
            ctx = percentile_context[stat_key]
            lines.append(
                f"  - {label}: {ctx['value']} "
                f"(p{ctx['percentile']}, {ctx['tier']})"
            )
        else:
            lines.append(f"  - {label}: {fighter[stat_key]}")

    lines.append("")
    lines.append("Last 5 fights (most recent first):")
    for fight in fighter["last_5"]:
        lines.append(f"  - {fight}")

    sample = percentile_context.get("_sample") if use_percentiles else None
    if sample:
        lines.append("")
        lines.append(
            f"Sample size: {sample['fights']} professional fights "
            f"(statistical reliability: {sample['reliability']}). "
            "The percentiles above have already been regressed toward the "
            "divisional median to account for this, so take them at face "
            "value. Do not re-inflate a thin record into elite grades on the "
            "strength of a few quick finishes, and do not write off an "
            "inexperienced fighter either - a short record means unknown, "
            "not bad. Say so in x_factors when reliability is low."
        )

    lines.append("")
    lines.append("Now produce the FighterProfile.")
    return "\n".join(lines)


def distill_profile(fighter: dict, percentile_context: Dict[str, dict],
                    use_percentiles: bool = True) -> FighterProfile:
    """Call the LLM and return a validated FighterProfile.

    Raises:
        openai.OpenAIError on API failure.
        pydantic.ValidationError if the LLM returns a malformed object even
        after the SDK's structured-output enforcement (very rare).
    """
    client = OpenAI()
    user_prompt = _build_user_prompt(fighter, percentile_context, use_percentiles)

    completion = client.beta.chat.completions.parse(
        model=PROFILE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=FighterProfile,
        temperature=0.2,
    )
    msg = completion.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"Model refused to produce profile: {msg.refusal}")
    profile = msg.parsed
    if profile is None:
        # SDK couldn't parse; fall back to manual JSON parse for a useful error.
        raise RuntimeError(
            f"Stage 1 returned unparseable content: {msg.content!r}"
        )
    # Enforce that the LLM kept the canonical name (it sometimes paraphrases).
    profile.name = fighter["name"]
    profile.weight_class = fighter["weight_class"]
    profile.record = fighter["record"]
    return profile
