"""
FraudGuard FastAPI Application.

Main entry point for the backend API. Configures CORS,
mounts routers, and sets up logging.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import analyze, predict, history, auth

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("FraudGuard API starting up...")
    yield
    logger.info("FraudGuard API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="FraudGuard API",
    description="FinTech Fraud Detection Pipeline API — IsolationForest + XGBoost + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow Vercel frontend and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(history.router, prefix="/api/v1", tags=["history"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FraudGuard API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "FraudGuard API",
        "version": "1.0.0",
        "supabase_configured": bool(os.getenv("SUPABASE_URL")),
    }
