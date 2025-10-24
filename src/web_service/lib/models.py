# Pydantic models for the web service
from typing import Literal

from pydantic import BaseModel, Field


class AbaloneInput(BaseModel):
    """Input schema for abalone size prediction."""

    Sex: Literal["M", "F", "I"] = Field(
        ..., description="Sex of the abalone: M (male), F (female), or I (infant)."
    )
    Length: float = Field(..., description="Longest shell measurement (mm).")
    Diameter: float = Field(..., description="Diameter perpendicular to length (mm).")
    Height: float = Field(..., description="Height with meat in shell (mm).")
    Whole_weight: float = Field(
        ..., alias="Whole weight", description="Weight of whole abalone (grams)."
    )
    Shucked_weight: float = Field(
        ..., alias="Shucked weight", description="Weight of meat (grams)."
    )
    Viscera_weight: float = Field(
        ..., alias="Viscera weight", description="Gut weight after bleeding (grams)."
    )
    Shell_weight: float = Field(
        ..., alias="Shell weight", description="Weight of dried shell (grams)."
    )
    Age: float = Field(..., description="Age of abalone in years.")


class PredObj(BaseModel):
    """Output schema for abalone age prediction"""

    Age: float
