"""
History Router — GET /api/v1/history/{user_id}

Returns past analysis history from Supabase for the authenticated user.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from db.supabase_client import get_history
from routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/history")
async def get_user_history(user_id: str = Depends(get_current_user)):
    """
    Get analysis history for a user.

    Args:
        user_id: UUID of the authenticated user.

    Returns:
        List of history entries with filename, date, counts, etc.

    Raises:
        HTTPException: 500 on database error.
    """
    try:
        history = get_history(user_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"History fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")
