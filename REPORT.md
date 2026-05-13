# REPORT.md — Fight Night Matchup

## 1. What & why

Fight Night Matchup is a web app that takes the names of two UFC fighters,
pulls their career statistics from a local database, and produces a
structured head-to-head matchup analysis: per-category advantages, a
stylistic-clash paragraph, the key factors most likely to decide the fight,
and a calibrated winner pick. It is built for casual MMA fans who follow
the sport, see two names announced for an upcoming card, and want a
quick analytical read instead of having to compare stat tables across
several sites themselves.

The hard part of the AI behavior is *honest, calibrated comparison*.
Each fighter has roughly a dozen meaningful stats, and "good" is relative
to the weight class (70% takedown defense is mediocre at heavyweight but
unremarkable at flyweight). The model also has to keep stats it can see
separate from claims it would love to invent — "glass chin", "ring rust",
"mental edge". The two-stage pipeline distills each fighter into a strict `FighterProfile`, then comparing the profiles — is the load-bearing design choice for 
both problems.

## 2. Iterations

> Metric: hybrid aggregate score in `[0, 1]`. Components: deterministic
> fact-coverage on `must_mention` keyword groups (40%), an LLM-judge
> rubric over factual accuracy / advantage identification / calibration
> averaged from a 1–5 scale (50%), and a forbidden-claim check (10%).
> The eval set is the 12 labeled matchups in `eval/test_cases.json`.
> Numbers below are from runs against `gpt-4o-mini` on the updated
> 2024-2025 fighter dataset; rerun the eval
> (`PIPELINE_VERSION=v1|v2|v3|v4 python eval/run_eval.py`) to reproduce.
> Pre-run result files are in `eval/results/`.

### V1 — Single-prompt baseline

- **Change.** One LLM call (`llm/v1_single_prompt.py`) gets raw stats for
  both fighters and must emit the full `MatchupResponse` (two profiles +
  matchup) in a single structured-output shot. The Project 4 pattern,
  included as the floor.
- **Motivating example.** `bw-omalley-dvalishvili`: the model graded
  O'Malley's wrestling "average" despite his 0.2 TDs / 15 min at the
  10th percentile. One prompt grading 16 stats and writing the matchup
  is too many decisions at once.
- **Delta.** N/A → aggregate **0.847**. Fact coverage 0.917, judge mean
  3.80 / 5, 0 of 12 forbidden-claim violations.
- **Conclusion.** Baseline performs reasonably on a well-structured
  dataset. Main weakness is calibration (3.33/5) — the model is
  overconfident when all context is crammed into one prompt. Next:
  split off a distillation step with a fixed schema.

### V2 — Two-stage, raw stats only

- **Change.** Pipeline split into Stage 1 (`distill_profile`: raw stats
  → `FighterProfile`) and Stage 2 (`generate_matchup`: two profiles →
  `MatchupAnalysis`). Stage 1 sees raw numbers only — no percentile context.
- **Motivating example.** V1 `lw-makhachev-gaethje` graded Gaethje's
  90% takedown defense as "above-average" and framed it as a grappling
  stat instead of the central wall against Islam's TDs. A dedicated
  distillation step with an explicit `top_strengths` slot was the fix.
- **Delta.** 0.847 → **0.864**. Fact coverage 0.944, calibration 3.50/5,
  advantage identification 4.08/5. Zero forbidden-claim violations.
- **Conclusion.** Structure alone closes most of the gap. Remaining
  weakness: the model still guesses whether a raw number is "good"
  without divisional context. That motivates V3.

### V3 — Two-stage + deterministic percentile context

- **Change.** `build_percentile_context` in `data/load_fighters.py`
  computes each stat's percentile and tier (`elite`/`above-average`/
  `average`/`below-average`/`weak`) within the fighter's weight class
  and appends it to every stat in the Stage 1 prompt. The model is
  instructed to treat tiers as ground truth.
- **Motivating example.** V2 `hw-pavlovich-blaydes` called Blaydes's
  wrestling "above-average" when it sits at the 90th percentile at HW.
  Injecting the tier label removes that guess.
- **Delta.** 0.864 → **0.858**. Advantage identification peaked at
  4.25/5. Marginal aggregate dip is within noise — a slight drop in
  factual accuracy (3.83/5) suggests the model occasionally
  over-interprets tier labels in edge cases.
- **Conclusion.** Percentile context measurably improves advantage
  identification and keeps forbidden-claim violations at zero. Scores
  are tightly clustered (0.847–0.864) across all versions, reflecting
  strong schema enforcement rather than a weak baseline. Calibration
  (3.33/5) remains the weakest axis — that motivates V4.

### V4 — V3 + deterministic closeness/calibration context

- **Change.** `build_closeness_context` in `data/load_fighters.py`
  computes the mean percentile gap between the two fighters and derives
  a `recommended_confidence` label injected into Stage 2 as a
  "CALIBRATION HINT" the model is told to treat as a strong prior.
- **Motivating example.** V3 calibration scored 3.33/5 — the model was
  picking "high" confidence on genuinely close matchups. A deterministic
  closeness score was intended to anchor it.
- **Delta.** 0.858 → **0.850**. Calibration stayed flat at 3.33/5.
  Advantage identification dropped slightly to 4.17/5.
- **Conclusion.** The hint didn't move calibration. Two likely reasons:
  (1) the LLM-judge scores based on matchup content, not the confidence
  label chosen; (2) the model overrides the hint when it has strong
  qualitative signal. The added context also introduced minor noise,
  dropping advantage ID. The right next step is constraining `confidence`
  at the schema level — deriving it from the advantage magnitudes already
  computed rather than leaving it as a free LLM choice.

## 3. Code walkthrough

A user types "Islam Makhachev" and "Justin Gaethje" and clicks **Analyze
Matchup**. `static/app.js:127` is the submit handler — it disables the
button, shows the loading state, and POSTs `{fighter_a, fighter_b}` to
`/api/matchup`. On the server, `app.py:49-75` validates the body and
calls `resolve_fighter` in `data/load_fighters.py:101` for each name; that
function tries an exact match first, falls back to a rapidfuzz `WRatio`
score ≥ 90 for typo tolerance, and otherwise returns "did you mean?"
suggestions. With both fighters resolved, `app.py:75` calls
`run_pipeline` (`llm/pipeline.py:32`), which dispatches on
`PIPELINE_VERSION`. The default V3 path computes percentile contexts
(`load_fighters.py:125`), submits both Stage 1 distillations to a
`ThreadPoolExecutor` (`pipeline.py:40`) so the two OpenAI calls run in
parallel, then chains `generate_matchup` (`llm/matchup.py:90`). Each
LLM call uses `client.beta.chat.completions.parse` with a Pydantic
`response_format`, so the SDK enforces the JSON schema before returning.
The validated `MatchupResponse` is `model_dump()`-ed into the HTTP
response, and `static/app.js:152` renders three cards — two fighter
profiles, one matchup breakdown.

**Design decision: parallel Stage 1 calls instead of batching both
fighters into one Stage 1 prompt.** I considered a "render both fighters
in one prompt and ask for two profiles back" design — it would be one
fewer API call. I rejected it because (a) profiles are independent, so
batching adds zero quality signal, and (b) two short isolated prompts
exhibit measurably less prompt-token bleed in pilot runs (the model
sometimes copied phrasing across profiles when batched). Parallelizing
two calls recovers the latency cost.

## 4. AI disclosure & safety

I used Claude (via Kiro) to structure and develop this project end-to-end.
Below are specific moments where I caught problems and pushed back.

1. **Wrong OpenAI SDK shape.** Claude generated `openai.ChatCompletion.create(...)`
   (pre-1.0 syntax). I caught it during a dry import and directed it to use
   `client.beta.chat.completions.parse` with a Pydantic `response_format`.

2. **Hallucinated stats.** The initial fighter dataset gave Khamzat Chimaev
   a 90% takedown defense that didn't match his public record. I
   checked against ufcstats.com and corrected the values.

3. **httpx version conflict.** First run crashed with `TypeError: Client.__init__()
   got an unexpected keyword argument 'proxies'`. I saw that
   `httpx==0.28.1` had been installed while `openai==1.54.4` expected
   `httpx<0.28`. I directed the downgrade to `0.27.2`.

4. **Stale fighter data.** The original dataset was a late-2023 snapshot.
   I noticed Jon Jones showed 27-1 and Ilia Topuria was still listed at
   Featherweight after moving to Lightweight. I directed Claude to update
   all 44 existing fighters and add 10 new ones using current stats from
   ufcstats.com and ufc.com.

5. **V4 negative result.** I noticed calibration was the weakest axis
   (3.33/5) and directed Claude to build a V4 injecting a deterministic
   closeness score as a calibration hint. The eval showed no improvement
   (0.850, calibration still 3.33/5). I accepted the negative result —
   the report documents why it failed and what the next possible step is.

**Safety risk: hallucinated fight facts.** The model is tempted to invent
genre-appropriate details that could unfairly damage a real fighter's reputation. Mitigation: every output field flows through a Pydantic schema, both system prompts explicitly forbid inventing fights or attributes, and deterministic percentile/tier
context anchors claims to the data. The UI footer labels output as analysis,
not a betting recommendation.
