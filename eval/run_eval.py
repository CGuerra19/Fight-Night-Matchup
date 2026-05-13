"""
Evaluation harness for the Fight Night Matchup pipeline.

Runs every labeled case in eval/test_cases.json through the pipeline, scores
each one with a hybrid metric (deterministic fact coverage + an LLM-judge
rubric + a forbidden-claim check), aggregates to a single 0–1 score, and
writes the per-case results to eval/results/<version>_results.json.

Usage:
    python eval/run_eval.py                    # tags run as "current"
    python eval/run_eval.py --version v2       # tags + saves as v2
    python eval/run_eval.py --limit 3          # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# Make the repo root importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

from data.load_fighters import build_percentile_context, resolve_fighter
from llm.pipeline import VERSION as PIPELINE_VERSION, run_pipeline
from llm.schemas import MatchupResponse


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")


# --- Scoring components ---------------------------------------------------

class RubricScore(BaseModel):
    """LLM-judge structured rubric. Each axis is 1-5; reasoning is free text."""
    factual_accuracy: Literal[1, 2, 3, 4, 5] = Field(
        description="Are the claims in the analysis grounded in the provided "
                    "profile data? 5 = no errors; 1 = multiple fabrications."
    )
    identifies_correct_advantages: Literal[1, 2, 3, 4, 5] = Field(
        description="Does the analysis name the genuine advantages each "
                    "fighter has based on the data? 5 = nails the central "
                    "dynamic; 1 = misses or inverts key advantages."
    )
    calibration: Literal[1, 2, 3, 4, 5] = Field(
        description="Does the confidence level match how clear the matchup "
                    "actually is? 5 = appropriately calibrated; 1 = wildly over- "
                    "or under-confident."
    )
    reasoning: str = Field(
        description="One paragraph explaining the scores. Reference specific "
                    "parts of the analysis."
    )


JUDGE_SYSTEM = """You are a precise grader scoring an automated MMA matchup
analysis. You will be given:
  - The two fighters' input stats (raw + percentile context).
  - Free-text notes describing what an ideal analysis should surface.
  - The full analysis output produced by the system.

Score on three axes, 1-5 each:
  - factual_accuracy: are all stated claims grounded in the input data?
  - identifies_correct_advantages: does it surface the genuine advantages
    suggested by the data and the notes?
  - calibration: is the confidence level appropriate to how clear the
    matchup actually is?

Be strict but fair. A 3 means "acceptable", a 5 means "this is what an
analyst would write." Penalize fabrication harshly.
"""


def _build_text_blob(resp: MatchupResponse) -> str:
    """Concatenate every text field in the response for keyword matching."""
    parts = []
    for prof in (resp.profile_a, resp.profile_b):
        parts.extend([
            prof.name, prof.style_label, prof.recent_form,
            *prof.top_strengths, *prof.top_weaknesses, *prof.x_factors,
            prof.striking_grade, prof.grappling_grade, prof.cardio_durability_grade,
        ])
    m = resp.matchup
    parts.append(m.stylistic_clash)
    parts.append(m.prediction_rationale)
    parts.extend(m.key_factors)
    for a in m.advantages:
        parts.extend([a.category, a.favored_fighter, a.magnitude, a.reasoning])
    return " ".join(parts).lower()


def _score_fact_coverage(blob: str, must_mention: List[List[str]]) -> tuple[float, List[bool]]:
    """For each must_mention group (OR-keywords), check ANY appears in blob."""
    if not must_mention:
        return 1.0, []
    hits = [any(kw.lower() in blob for kw in group) for group in must_mention]
    return sum(hits) / len(hits), hits


def _score_forbidden_claims(blob: str, must_not_claim: List[List[str]]) -> tuple[float, List[bool]]:
    """1.0 if no forbidden keywords appear, else fractional penalty.

    Each must_not_claim group is a list of variants of the same forbidden
    claim; if ANY variant appears, the group counts as a violation.
    """
    if not must_not_claim:
        return 1.0, []
    violations = [any(kw.lower() in blob for kw in group) for group in must_not_claim]
    clean_fraction = 1.0 - (sum(violations) / len(violations))
    return clean_fraction, violations


def _render_for_judge(fighter_a_raw: dict, fighter_b_raw: dict,
                      ctx_a: dict, ctx_b: dict, notes: str,
                      resp: MatchupResponse) -> str:
    def render_raw(f, ctx):
        lines = [f"{f['name']} ({f['weight_class']}, {f['record']}):"]
        for k, c in ctx.items():
            lines.append(f"  {k} = {c['value']} (p{c['percentile']}, {c['tier']})")
        lines.append(f"  last_5 = {f['last_5']}")
        return "\n".join(lines)

    out = [
        "Input data:",
        render_raw(fighter_a_raw, ctx_a),
        render_raw(fighter_b_raw, ctx_b),
        "",
        "Notes on what an ideal analysis should surface:",
        notes,
        "",
        "Analysis to grade (JSON):",
        json.dumps(resp.model_dump(), indent=2),
    ]
    return "\n".join(out)


def _score_judge(fighter_a_raw, fighter_b_raw, ctx_a, ctx_b, notes, resp) -> RubricScore:
    client = OpenAI()
    user = _render_for_judge(fighter_a_raw, fighter_b_raw, ctx_a, ctx_b, notes, resp)
    completion = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=RubricScore,
        temperature=0.0,
    )
    return completion.choices[0].message.parsed


# --- Runner ----------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    fact_coverage: float
    fact_hits: List[bool]
    forbidden_clean: float
    forbidden_violations: List[bool]
    judge_factual: int
    judge_advantages: int
    judge_calibration: int
    judge_reasoning: str
    aggregate: float
    response: Optional[dict]
    error: Optional[str] = None


WEIGHTS = {"facts": 0.4, "judge": 0.5, "forbidden": 0.1}


def _aggregate(fact_cov: float, forbidden_clean: float, judge: RubricScore) -> float:
    judge_norm = (judge.factual_accuracy + judge.identifies_correct_advantages
                  + judge.calibration) / 15.0
    return (WEIGHTS["facts"] * fact_cov
            + WEIGHTS["judge"] * judge_norm
            + WEIGHTS["forbidden"] * forbidden_clean)


def run_case(case: dict) -> CaseResult:
    fa, _ = resolve_fighter(case["fighter_a"])
    fb, _ = resolve_fighter(case["fighter_b"])
    if not fa or not fb:
        return CaseResult(case["id"], 0.0, [], 0.0, [], 0, 0, 0,
                          "fighter not found", 0.0, None,
                          error="fighter not found in DB")
    ctx_a = build_percentile_context(fa)
    ctx_b = build_percentile_context(fb)

    try:
        resp = run_pipeline(fa, fb)
    except (OpenAIError, RuntimeError) as e:
        return CaseResult(case["id"], 0.0, [], 0.0, [], 0, 0, 0,
                          f"pipeline error: {e}", 0.0, None, error=str(e))

    blob = _build_text_blob(resp)
    fact_cov, fact_hits = _score_fact_coverage(blob, case.get("must_mention", []))
    forbidden_clean, viols = _score_forbidden_claims(blob, case.get("must_not_claim", []))
    judge = _score_judge(fa, fb, ctx_a, ctx_b, case.get("notes", ""), resp)
    agg = _aggregate(fact_cov, forbidden_clean, judge)
    return CaseResult(
        case_id=case["id"],
        fact_coverage=fact_cov,
        fact_hits=fact_hits,
        forbidden_clean=forbidden_clean,
        forbidden_violations=viols,
        judge_factual=judge.factual_accuracy,
        judge_advantages=judge.identifies_correct_advantages,
        judge_calibration=judge.calibration,
        judge_reasoning=judge.reasoning,
        aggregate=agg,
        response=resp.model_dump(),
    )


def main():
    p = argparse.ArgumentParser(
        description="Run the eval suite. Set PIPELINE_VERSION=v1|v2|v3 to pick "
                    "which pipeline version is graded (default v3).",
    )
    p.add_argument("--version", default=PIPELINE_VERSION,
                   help="Tag for this run; saved to eval/results/<version>_results.json. "
                        "Defaults to whatever PIPELINE_VERSION resolves to.")
    p.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    args = p.parse_args()

    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    if args.limit > 0:
        cases = cases[:args.limit]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results: List[CaseResult] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        r = run_case(case)
        results.append(r)
        if r.error:
            print(f"ERROR ({r.error})")
        else:
            print(f"agg={r.aggregate:.2f}  facts={r.fact_coverage:.2f}  "
                  f"judge={r.judge_factual}/{r.judge_advantages}/{r.judge_calibration}")

    aggregate = sum(r.aggregate for r in results) / max(1, len(results))
    elapsed = time.time() - t0
    print(f"\n=== Version: {args.version} ===")
    print(f"Mean aggregate score: {aggregate:.3f}  ({len(results)} cases, {elapsed:.1f}s)")
    print(f"Mean fact coverage:   {sum(r.fact_coverage for r in results)/len(results):.3f}")
    print(f"Mean forbidden clean: {sum(r.forbidden_clean for r in results)/len(results):.3f}")
    print(f"Mean judge factual:   {sum(r.judge_factual for r in results)/len(results):.2f}/5")
    print(f"Mean judge advantage: {sum(r.judge_advantages for r in results)/len(results):.2f}/5")
    print(f"Mean judge calib.:    {sum(r.judge_calibration for r in results)/len(results):.2f}/5")

    out = {
        "version": args.version,
        "pipeline_version": PIPELINE_VERSION,
        "model_profile": os.environ.get("PROFILE_MODEL", "gpt-4o-mini"),
        "model_matchup": os.environ.get("MATCHUP_MODEL", "gpt-4o-mini"),
        "model_judge": JUDGE_MODEL,
        "mean_aggregate": aggregate,
        "cases": [r.__dict__ for r in results],
    }
    out_path = os.path.join(RESULTS_DIR, f"{args.version}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
