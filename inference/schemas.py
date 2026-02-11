from pydantic import BaseModel, Field
from typing import Dict, Optional, List

from cockatoo_ml.registry import APIConfig


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=APIConfig.MIN_TEXT_LENGTH, description="Text to classify")
    threshold: Optional[float] = Field(APIConfig.DEFAULT_THRESHOLD, ge=APIConfig.MIN_THRESHOLD, le=APIConfig.MAX_THRESHOLD, description="Confidence threshold for positive labels")


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, float]           # label -> probability
    positive_labels: List[str]              # labels above threshold
    top_label: Optional[str] = None
    max_score: Optional[float] = None
    error: Optional[str] = None