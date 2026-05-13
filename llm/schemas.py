"""
Pydantic schemas for the Fight Night Matchup pipeline.

These are the "predictable format" the project is built around: Stage 1 distills
raw fighter stats into a FighterProfile, and everything downstream (Stage 2,
the API response, the frontend, the eval harness) consumes only these shapes.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


Grade = Literal["elite", "above-average", "average", "below-average", "weak"]
Magnitude = Literal["even", "slight", "clear", "decisive"]
Confidence = Literal["low", "medium", "high"]


class StatWithContext(BaseModel):
    """A raw stat carried forward to Stage 2 with its tier in the dataset."""
    value: float
    percentile: int = Field(ge=0, le=100)
    tier: Grade


class FighterProfile(BaseModel):
    """Stage 1 output — a distilled, predictable summary of one fighter."""
    name: str
    weight_class: str
    record: str
    style_label: str = Field(
        description="Concise style tag, e.g. 'pressure wrestler', "
                    "'volume striker', 'counter-puncher', 'BJJ specialist'."
    )
    striking_grade: Grade
    grappling_grade: Grade
    cardio_durability_grade: Grade
    top_strengths: List[str] = Field(
        min_length=2, max_length=4,
        description="2-4 short phrases highlighting what this fighter does best."
    )
    top_weaknesses: List[str] = Field(
        min_length=1, max_length=3,
        description="1-3 short phrases highlighting exploitable weaknesses."
    )
    recent_form: str = Field(
        description="One sentence summarizing momentum from the last 5 fights "
                    "(streaks, finishes, level of competition, layoffs)."
    )
    x_factors: List[str] = Field(
        min_length=0, max_length=3,
        description="0-3 unusual variables: chin, finishing rate, durability, "
                    "ring rust, fight IQ, etc."
    )


class CategoryAdvantage(BaseModel):
    """Per-category advantage in the matchup."""
    category: Literal["striking", "grappling", "cardio_durability", "physical"]
    favored_fighter: str = Field(description="Fighter name, or 'even'.")
    magnitude: Magnitude
    reasoning: str = Field(
        description="One sentence grounded in the profile data, not raw stats."
    )


class MatchupAnalysis(BaseModel):
    """Stage 2 output — head-to-head comparison consuming two FighterProfiles."""
    advantages: List[CategoryAdvantage] = Field(
        min_length=4, max_length=4,
        description="Exactly one entry per category: striking, grappling, "
                    "cardio_durability, physical."
    )
    stylistic_clash: str = Field(
        description="2-3 sentences on how the two styles interact "
                    "(e.g., wrestler vs. striker, pressure vs. counter)."
    )
    key_factors: List[str] = Field(
        min_length=3, max_length=5,
        description="3-5 bullet points naming the things most likely to "
                    "decide the fight."
    )
    predicted_winner: str = Field(
        description="Name of the predicted winner. Must be one of the two fighters."
    )
    confidence: Confidence
    prediction_rationale: str = Field(
        description="2-3 sentences explaining the pick, referencing the "
                    "advantages and stylistic clash above."
    )


class MatchupResponse(BaseModel):
    """The full payload returned by /api/matchup."""
    profile_a: FighterProfile
    profile_b: FighterProfile
    matchup: MatchupAnalysis
