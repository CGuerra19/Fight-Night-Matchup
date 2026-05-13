"""
Fight Night Matchup — Flask backend.

Wires together the local fighter database (data.load_fighters) and the
two-stage LLM pipeline (llm.profile then llm.matchup), and serves a small
single-page UI from templates/.
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAIError
from pydantic import ValidationError

# Load .env BEFORE importing the LLM modules, so OPENAI_API_KEY is set when
# the OpenAI client is instantiated.
load_dotenv()

from data.load_fighters import (
    SNAPSHOT_NOTE,
    resolve_fighter,
    search_names,
)
from llm.pipeline import VERSION as PIPELINE_VERSION, run_pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fight-night")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", snapshot_note=SNAPSHOT_NOTE)


@app.route("/api/fighters")
def fighters():
    """Autocomplete suggestions for the fighter name inputs."""
    q = request.args.get("q", "").strip()
    return jsonify({"suggestions": search_names(q, limit=8)})


@app.route("/api/matchup", methods=["POST"])
def matchup():
    """Resolve two fighters, run Stage 1 in parallel, then Stage 2."""
    body = request.get_json(silent=True) or {}
    name_a = (body.get("fighter_a") or "").strip()
    name_b = (body.get("fighter_b") or "").strip()

    if not name_a or not name_b:
        return jsonify({"error": "Both fighter_a and fighter_b are required."}), 400
    if name_a.lower() == name_b.lower():
        return jsonify({"error": "Pick two different fighters."}), 400

    fighter_a, suggestions_a = resolve_fighter(name_a)
    fighter_b, suggestions_b = resolve_fighter(name_b)
    missing = []
    if not fighter_a:
        missing.append({"input": name_a, "suggestions": suggestions_a})
    if not fighter_b:
        missing.append({"input": name_b, "suggestions": suggestions_b})
    if missing:
        return jsonify({
            "error": "One or more fighters were not found in the local database.",
            "missing": missing,
        }), 404

    # Run whichever pipeline version PIPELINE_VERSION selects (default v3).
    try:
        response = run_pipeline(fighter_a, fighter_b)
    except OpenAIError as e:
        log.exception("OpenAI error during pipeline")
        return jsonify({"error": f"OpenAI API error: {e}"}), 502
    except (ValidationError, RuntimeError) as e:
        log.exception("Pipeline validation error")
        return jsonify({"error": f"Pipeline error: {e}"}), 500

    return jsonify(response.model_dump())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    log.info("Starting Fight Night Matchup on port %d (pipeline=%s)", port, PIPELINE_VERSION)
    app.run(host="127.0.0.1", port=port, debug=False)
