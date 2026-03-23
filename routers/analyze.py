"""
Analyze Router — POST /api/v1/analyze

Accepts CSV upload, runs full fraud detection pipeline,
and returns complete dashboard JSON.
"""

import io
import time
import math
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Optional

from pipeline.cleaner import DataCleaner
from pipeline.features import FeatureEngineer
from pipeline.model import FraudDetector
from pipeline.analyzer import EDAAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared model cache (persists between requests)
_detector: Optional[FraudDetector] = None


def get_detector() -> FraudDetector:
    """Get or create the shared FraudDetector instance."""
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector


def sanitize_for_json(obj):
    """
    Recursively sanitize a Python object for JSON serialization.
    Replaces NaN, inf, -inf with None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Full fraud detection pipeline endpoint.

    Accepts: multipart/form-data with CSV file + optional user_id.
    Runs: cleaner → features → model → analyzer.
    Returns: complete dashboard JSON.

    Args:
        file: Uploaded CSV file.
        user_id: Optional authenticated user ID for saving to history.

    Returns:
        AnalysisResponse dict with data_quality, summary_stats,
        fraud_results, chart_data, filename, total_rows.

    Raises:
        HTTPException: 400 if not CSV, 422 if parsing fails, 500 on error.
    """
    start_time = time.time()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Check file size (50MB limit)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit.")

    try:
        # Parse CSV
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=422, detail="CSV file is empty.")

        logger.info(f"Processing {file.filename}: {len(df)} rows, {len(df.columns)} columns")

        # Step 1: Clean
        cleaner = DataCleaner()
        df_clean, quality_report = cleaner.clean(df)

        # Step 2: Feature engineering
        fe = FeatureEngineer()
        df_features = fe.engineer_features(df_clean)

        # Step 3: Fraud detection
        detector = get_detector()
        fraud_results = detector.detect(df_features)

        # Step 4: EDA
        analyzer = EDAAnalyzer()
        summary_stats = analyzer.get_summary_stats(df_clean)
        chart_data = analyzer.get_chart_data(df_clean, df_features)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Pipeline complete in {elapsed}s")

        # Calculate data quality score
        total = quality_report['total_rows']
        issues = (
            quality_report['amount_parse_failures'] +
            quality_report['timestamp_parse_failures'] +
            quality_report['invalid_ips'] +
            quality_report['duplicate_rows_removed']
        )
        quality_score = round(max(0, (1 - issues / max(total, 1)) * 100), 1)

        response = {
            "data_quality": {
                **quality_report,
                "quality_score": quality_score,
            },
            "summary_stats": summary_stats,
            "fraud_results": fraud_results,
            "chart_data": chart_data,
            "filename": file.filename,
            "total_rows": total,
            "processing_time_seconds": elapsed,
        }

        # Save to Supabase if user_id provided
        if user_id:
            try:
                from db.supabase_client import save_analysis
                save_analysis(
                    user_id=user_id,
                    filename=file.filename,
                    total_rows=total,
                    fraud_count=fraud_results['fraud_count'],
                    fraud_rate=fraud_results['fraud_rate'],
                    f1=fraud_results['f1_score'],
                    result_json=response,
                )
            except Exception as e:
                logger.warning(f"Failed to save to Supabase: {e}")

        return JSONResponse(content=sanitize_for_json(response))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
