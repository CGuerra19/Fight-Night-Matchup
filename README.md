# Fight Night Matchup

An AI-powered web app for analyzing hypothetical UFC fights. Pick any two of
the 86 fighters in the local database; the app distills each fighter's
career stats into a structured profile, then runs a head-to-head
matchup analysis with predicted advantages, stylistic clash, key factors,
and a calibrated winner pick.

The AI pipeline runs in two stages so each LLM call has a narrow,
well-defined job:

1. **Stage 1 — Distill.** For each fighter, raw stats + deterministically
   computed percentile/tier context are handed to the model, which returns
   a strict `FighterProfile` (style label, striking/grappling/cardio grades,
   strengths, weaknesses, recent form, x-factors).
2. **Stage 2 — Compare.** The two `FighterProfile` objects (not the raw
   numbers) feed a second model call that returns a `MatchupAnalysis` with
   four per-category advantages, a stylistic-clash paragraph, key factors,
   predicted winner, and confidence.

Everything is OpenAI-only — no other API keys, no hosted services. The
fighter database is a local JSON file, the eval harness writes results to
disk.

---

## Quick start (copy/paste)

From an empty directory, run these commands in order. Replace
`<your-repo-url>` with the GitHub URL of this repo and `sk-...` with
your OpenAI API key.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> fight-night-matchup
cd fight-night-matchup

# 2. Create and activate a virtualenv (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Create your .env and paste your OpenAI API key into it
cp .env.example .env
echo 'OPENAI_API_KEY=sk-...' > .env     # or open .env in an editor

# 5. Start the web app
python app.py
```

Now open <http://127.0.0.1:5000> in your browser. Type two fighter names
into the inputs (autocomplete suggests matches as you type), click
**Analyze Matchup**, and you should see two profile cards plus a
predicted winner within ~5 seconds.

To stop the server: press `Ctrl+C` in the terminal where `python app.py`
is running.

The sections below repeat these steps in more detail and add the eval
harness commands.

---

## Requirements

- Python **3.10+** (developed and smoke-tested on 3.10.12).
- An OpenAI API key with access to `gpt-4o-mini`.
- Internet access (only to reach `api.openai.com`).

The repo has been smoke-tested on macOS and Linux.

---

## Setup

Run these commands from the repo root, in order. Any Python 3.10 or newer
will work; substitute `python3.11`/`python3.12` if `python3` on your machine
points at something older.

```bash
# 1. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Create your .env from the template, then paste your OpenAI key in
cp .env.example .env
# open .env in your editor and set OPENAI_API_KEY=sk-...
```

---

## Run the web app

```bash
python app.py
```

The server logs `Starting Fight Night Matchup on port 5000 (pipeline=v5)`
and listens on http://127.0.0.1:5000. Open that URL in any modern browser.

### Using the UI

1. Type a fighter name into each input. As you type, an autocomplete
   dropdown shows fighters from the local database.
2. Click **Analyze Matchup**.
3. After a few seconds (two parallel Stage 1 calls + one Stage 2 call),
   the page shows two fighter profile cards followed by the matchup
   breakdown and predicted winner.

### Example fighters to try

- Islam Makhachev vs. Justin Gaethje  *(wrestler vs. striker)*
- Jon Jones vs. Tom Aspinall  *(generational heavyweight matchup)*
- Alex Pereira vs. Jiri Prochazka  *(power kickboxing vs. unorthodox striking)*
- Max Holloway vs. Yair Rodriguez  *(volume vs. creativity at featherweight)*

The full fighter list is in `data/fighters.json`.

### Error handling

The app surfaces a clear error banner if a fighter isn't recognized
(with "did you mean…?" suggestions), if the two inputs are identical, or
if the OpenAI API call fails. Submission stays disabled during the API
call so the user can see the loading state.

---

## Run the evaluation

The eval harness scores the pipeline on 12 labeled matchups using a hybrid
metric: deterministic fact-coverage checks (40%), an LLM-judge rubric for
factual accuracy / advantage identification / calibration (50%), and a
forbidden-claim check (10%). Results land in `eval/results/`.

```bash
# Default: runs version v5
python eval/run_eval.py
```

### Reproducing the V1 → V2 → V3 → V4 sweep from REPORT.md

Set `PIPELINE_VERSION` to switch which version is graded. Each run writes
to a separately named results file.

```bash
PIPELINE_VERSION=v1 python eval/run_eval.py     # single-prompt baseline
PIPELINE_VERSION=v2 python eval/run_eval.py     # two-stage, no percentile context
PIPELINE_VERSION=v3 python eval/run_eval.py     # two-stage with percentile context
PIPELINE_VERSION=v4 python eval/run_eval.py     # v3 + deterministic closeness/calibration hint
PIPELINE_VERSION=v5 python eval/run_eval.py     # v4 + sample-size shrinkage, physical context,
                                                #      statistical-favorite guard (default)
```

### A note on V5

V5 was built from the UFC 331 post-mortem (`eval/ufc331_predictions.json`,
7/10 correct). It makes three deterministic changes - no extra LLM calls, so
cost per matchup is unchanged:

1. **Sample-size shrinkage.** Stage 1 percentiles are regressed toward the
   divisional median in proportion to how many fights they were computed
   over. Gable Steveson was graded "elite" striking off 16.29 SLpM across
   four first-round finishes, and lost; he now grades "above-average". The
   correction is symmetric - a thin record means unknown, not bad.
2. **Physical attributes reach Stage 2.** The `FighterProfile` schema carries
   no height, reach, age or stance, so Stage 2 had been judging its
   "physical" advantage category with no physical data at all. Despaigne beat
   Tuivasa with a nine-inch reach edge that never entered the reasoning.
3. **Statistical-favorite guard.** Stage 2 is told which fighter leads on
   aggregate percentiles and must justify picking against it. This targets a
   specific observed failure where the model named its opponent's advantages
   and then picked the other fighter anyway on "power and experience".

V5 also enables corrected confidence banding. The V4 banding contains a bug -
both arms of its middle branch return `"medium"` - which is why REPORT.md
records V4 calibration as no better than V3. The bug is left intact under
`spread=False` so the V1-V4 sweep still reproduces exactly.

**Honest caveat on results.** V5 scores 0.850 aggregate on the 12-case
harness, against 0.864 for V2 and 0.858 for V3. It does not win. The whole
V1-V5 spread is 0.847-0.864 across 12 cases, which is well inside noise, and
the harness does not score prediction accuracy at all - it measures fact
coverage, forbidden claims, and judge ratings. The changes above target
picking winners, which this harness cannot see. On the only head-to-head
evidence available, the seven decided UFC 331 bouts, V5 went 5/7 where V3
went 4/7 - but those fixes were designed knowing those outcomes, so treat
that number as a sanity check, not a result.

A run takes ~1–3 minutes and ~$0.05 in OpenAI credit on `gpt-4o-mini`.

Optional flags:

- `--limit 3` — run only the first three cases (smoke test).
- `--version <tag>` — override the output filename tag.

---

## Project layout

```
fight-night-matchup/
├── app.py                    Flask server + routes
├── data/
│   ├── fighters.json         Fighter database (86 fighters, stats through Sep 2026)
│   └── load_fighters.py      Lookup, fuzzy match, percentile computation
├── llm/
│   ├── schemas.py            Pydantic models — the "predictable format"
│   ├── profile.py            Stage 1 — distill raw stats into a profile
│   ├── matchup.py            Stage 2 — compare two profiles
│   ├── v1_single_prompt.py   V1 baseline (single LLM call)
│   └── pipeline.py           Dispatcher selecting V1/V2/V3/V4
├── templates/index.html      Single-page UI
├── static/
│   ├── style.css
│   └── app.js                Autocomplete, fetch, render
├── eval/
│   ├── test_cases.json       12 labeled matchups
│   ├── run_eval.py           Hybrid scorer + LLM judge
│   └── results/              (auto-generated)
├── requirements.txt
├── .env.example
├── README.md
└── REPORT.md
```

---

## Data note

`data/fighters.json` is a snapshot of public UFC career statistics current
through **19 September 2026 (UFC 331)**, covering 86 fighters across all weight
classes including women's divisions. Records and recent results were sourced
from ufc.com athlete profiles. See `snapshot_note` inside the JSON for known
gaps — stance is marked `Unknown` where it could not be confirmed, ufc.com
takedown-accuracy values are unreliable, and fighters with very few bouts have
per-minute rates inflated by small samples. The stats are used for analysis
only — this app does not claim to have live data.
