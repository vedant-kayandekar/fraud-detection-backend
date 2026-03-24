"""
Supabase Client for FraudGuard.

Initializes the Supabase client from environment variables.
Provides helper methods for database operations.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy client initialization
_supabase_client = None


def get_supabase():
    """
    Get or create the Supabase client singleton.

    Returns:
        Supabase client instance, or None if not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        logger.warning(
            "SUPABASE_URL or SUPABASE_SERVICE_KEY not set. "
            "Database features disabled."
        )
        return None

    try:
        from supabase import create_client, ClientOptions

        # Increase DB query timeout to 60s (free-tier can be slow)
        options = ClientOptions(postgrest_client_timeout=60)
        
        _supabase_client = create_client(url, key, options=options)
        logger.info("Supabase client initialized successfully.")
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None


def save_analysis(user_id: str, filename: str, total_rows: int,
                  fraud_count: int, fraud_rate: float, f1: float,
                  result_json: dict) -> Optional[dict]:
    """
    Save analysis results to Supabase.

    Args:
        user_id: UUID of the authenticated user.
        filename: Uploaded CSV filename.
        total_rows: Total rows processed.
        fraud_count: Number of fraud rows detected.
        fraud_rate: Fraud rate as percentage.
        f1: Model F1 score.
        result_json: Full result JSON to store.

    Returns:
        Inserted row dict or None on failure.
    """
    client = get_supabase()
    if client is None:
        return None
    try:
        data = {
            "user_id": user_id,
            "filename": filename,
            "total_rows": total_rows,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "f1_score": f1,
            "result_json": result_json,
        }
        result = client.table("analysis_history").insert(data).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")
        return None


def get_history(user_id: str) -> list:
    """
    Get analysis history for a user.

    Args:
        user_id: UUID of the authenticated user.

    Returns:
        List of history entries.
    """
    client = get_supabase()
    if client is None:
        return []
    try:
        result = (
            client.table("analysis_history")
            .select("id, filename, upload_time, total_rows, fraud_count, fraud_rate, f1_score, result_json")
            .eq("user_id", user_id)
            .order("upload_time", desc=True)
            .limit(20)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []
