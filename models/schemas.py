"""
Pydantic Models for FraudGuard API.

Defines request/response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SingleTransactionRequest(BaseModel):
    """Request body for single transaction fraud prediction."""
    transaction_id: Optional[str] = "TXN-TEST"
    user_id: str = Field(..., description="User ID for the transaction")
    transaction_amount: Optional[str] = None
    transaction_timestamp: Optional[str] = None
    user_location: Optional[str] = None
    merchant_location: Optional[str] = None
    merchant_category: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    payment_method: Optional[str] = None
    account_balance: Optional[float] = None
    transaction_status: Optional[str] = "success"
    ip_address: Optional[str] = None
    amt: Optional[str] = None


class SinglePredictionResponse(BaseModel):
    """Response body for single transaction prediction."""
    fraud_probability: float
    is_fraud: bool
    confidence: str
    reasons: List[Any]


class ShapReason(BaseModel):
    """Individual SHAP explanation reason."""
    feature: str
    value: float = 0.0
    impact: str


class FraudRow(BaseModel):
    """Single fraud row in the results."""
    transaction_id: str
    user_id: str
    clean_amount: float
    clean_timestamp: str
    user_city: str
    merchant_category: str
    fraud_probability: float
    fraud_rank: int
    shap_reasons: List[Dict[str, Any]]
    device_id: str = ""
    hour_of_day: int = 0
    device_type: str = ""


class AnalysisResponse(BaseModel):
    """Full analysis response for dashboard."""
    data_quality: Dict[str, Any]
    summary_stats: Dict[str, Any]
    fraud_results: Dict[str, Any]
    chart_data: Dict[str, Any]
    filename: str
    total_rows: int


class HistoryEntry(BaseModel):
    """One entry in analysis history."""
    id: Optional[str] = None
    user_id: Optional[str] = None
    filename: str
    upload_time: Optional[str] = None
    total_rows: int
    fraud_count: int
    fraud_rate: float
    f1_score: float
    result_json: Optional[Dict[str, Any]] = None
