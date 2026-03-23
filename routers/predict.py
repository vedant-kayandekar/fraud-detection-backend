"""
Predict Router — POST /api/v1/predict-single

Accepts a single transaction row and returns fraud prediction.
Uses the model trained on the last uploaded dataset.
"""

import logging
from fastapi import APIRouter, HTTPException

from models.schemas import SingleTransactionRequest, SinglePredictionResponse
from routers.analyze import get_detector

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict-single", response_model=SinglePredictionResponse)
async def predict_single(request: SingleTransactionRequest):
    """
    Predict fraud probability for a single transaction.

    Uses the model trained during the last /analyze call.

    Args:
        request: SingleTransactionRequest with transaction fields.

    Returns:
        SinglePredictionResponse with fraud_probability, is_fraud,
        confidence, and reasons.

    Raises:
        HTTPException: 500 on prediction error.
    """
    try:
        detector = get_detector()
        row_dict = request.model_dump()
        result = detector.predict_single(row_dict)
        return SinglePredictionResponse(**result)
    except Exception as e:
        logger.error(f"Single prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
