"""API module for churn prediction inference."""

from src.api.main import app
from src.api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
)

__all__ = [
    "app",
    "CustomerInput",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ExplanationResponse",
]
