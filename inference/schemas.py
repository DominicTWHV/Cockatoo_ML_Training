from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for positive labels")


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, float]           # label -> probability
    positive_labels: List[str]              # labels above threshold
    top_label: Optional[str] = None
    max_score: Optional[float] = None
    error: Optional[str] = None