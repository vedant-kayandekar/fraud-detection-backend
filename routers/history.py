"""
History Router — GET /api/v1/history

Returns past analysis history from Supabase for the authenticated user.
Returns empty list for guest/unauthenticated users.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional

from db.supabase_client import get_history

logger = logging.getLogger(__name__)
router = APIRouter()

_security = HTTPBearer(auto_error=False)


async def _get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> Optional[str]:
    """Return user_id if valid token, else None."""
    if credentials is None:
        return None
    try:
        from db.supabase_client import get_supabase
        client = get_supabase()
        if not client:
            return None
        res = client.auth.get_user(credentials.credentials)
        if res and res.user:
            return res.user.id
    except Exception:
        pass
    return None


@router.get("/history")
async def get_user_history(user_id: Optional[str] = Depends(_get_optional_user)):
    """
    Get analysis history for a user.
    Returns empty list if not authenticated (guest mode).
    """
    if not user_id:
        return {"history": []}

    try:
        history = get_history(user_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"History fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")
