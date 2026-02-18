from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union

from cockatoo_ml.registry import APIConfig


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=APIConfig.MIN_TEXT_LENGTH, description="Text to classify")
    threshold: Optional[Union[float, Dict[str, float]]] = Field(None, description="Confidence threshold for positive labels. Can be a single float for all labels or a dict of per-label thresholds.")


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, float]           # label -> probability
    positive_labels: List[str]              # labels above threshold
    top_label: Optional[str] = None
    max_score: Optional[float] = None
    error: Optional[str] = None