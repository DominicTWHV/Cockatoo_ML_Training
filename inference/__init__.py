"""Inference module for Constellation One Text model"""

from .model import ThreatClassifier
from .schemas import PredictionRequest, PredictionResponse

__all__ = ["ThreatClassifier", "PredictionRequest", "PredictionResponse"]
