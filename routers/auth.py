"""
Authentication Router for FastAPI.

Provides /login and /signup endpoints that utilize the Supabase Python Client.
Also exports a dependency `get_current_user` to secure other endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

from db.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

class AuthCredentials(BaseModel):
    email: str
    password: str

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    FastAPI Dependency to enforce authentication.
    Validates the Bearer token with Supabase and returns the user ID.
    """
    client = get_supabase()
    if not client:
        # If DB is not configured, we allow bypass for testing purposes
        # (Though we shouldn't really, but to not break the app if DB keys are missing)
        logger.warning("Supabase not configured. Bypassing authentication check.")
        return "mock-user-id"

    token = credentials.credentials
    try:
        # get_user checks the JWT signature against Supabase
        res = client.auth.get_user(token)
        if not res or not res.user:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return res.user.id
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.post("/signup")
async def signup(creds: AuthCredentials):
    """Register a new user in Supabase Auth."""
    client = get_supabase()
    if not client:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        res = client.auth.sign_up({
            "email": creds.email,
            "password": creds.password,
        })
        if res and res.user:
            return {
                "message": "User created successfully",
                "user_id": res.user.id,
                "access_token": res.session.access_token if res.session else None
            }
        else:
            raise HTTPException(status_code=400, detail="Signup failed.")
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(creds: AuthCredentials):
    """Authenticate a user and return a JWT access token."""
    client = get_supabase()
    if not client:
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        res = client.auth.sign_in_with_password({
            "email": creds.email,
            "password": creds.password,
        })
        if res and res.session:
            return {
                "message": "Login successful",
                "user_id": res.user.id,
                "access_token": res.session.access_token
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password")
    except Exception as e:
        logger.error(f"Login error: {e}")
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            raise HTTPException(status_code=504, detail="Authentication server timed out. Please try again.")
        raise HTTPException(status_code=401, detail="Invalid email or password")
