from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class MammographyStudy(BaseModel):
    study_id: str
    patient_id: str
    dataset_source: str
    ground_truth: int | None = Field(default=None, ge=0, le=1)
    l_cc: str
    r_cc: str
    l_mlo: str
    r_mlo: str
    left_ground_truth: int | None = Field(default=None, ge=0, le=1)
    right_ground_truth: int | None = Field(default=None, ge=0, le=1)
    horizontal_flip: Literal["NO", "YES"] = "NO"

class ModelPrediction(BaseModel):
    model: str
    model_version: str
    malignancy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: Literal["SUCCESS", "FAILED", "INCOMPATIBLE"]
    inference_time_ms: float = 0.0
    views_used: list[str] = []
    xai_artifacts: list[str] = []
    native_outputs: dict = {}
    aggregation_method: str | None = None
    error_code: str | None = None
    error_message: str | None = None

class EnsembleResult(BaseModel):
    ensemble_malignancy_score: float = Field(ge=0.0, le=1.0)
    classification: Literal["CANCER", "NO_CANCER"]
    threshold: float
    weights: dict[str, float]
    model_range: float
    model_std: float
    discordance: bool

    @model_validator(mode="after")
    def weights_sum(self):
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1")
        return self
