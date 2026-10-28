"""FastAPI Application Entry Point for CineAI."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.v1 import (
    health,
    models,
    movies,
    users,
    recommendations,
    metrics,
    compare,
    music,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-performance REST API for CineAI: 10 AI Movie Recommendation Models on MovieLens 100K",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware (Supports explicitly whitelisted origins and all Vercel frontend domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 Routers (CineAI Movie Recommendation)
api_v1 = settings.API_V1_STR
app.include_router(health.router, prefix=api_v1)
app.include_router(models.router, prefix=api_v1)
app.include_router(movies.router, prefix=api_v1)
app.include_router(users.router, prefix=api_v1)
app.include_router(recommendations.router, prefix=api_v1)
app.include_router(metrics.router, prefix=api_v1)
app.include_router(compare.router, prefix=api_v1)

# Include MusicAI Routers (Mounted under both /api/music and /api/v1/music for multi-domain support)
app.include_router(music.router, prefix="/api/music")
app.include_router(music.router, prefix=f"{api_v1}/music")


@app.get("/")
def root():
    return {
        "app": "Multi-Domain AI Recommendation System (CineAI + MusicAI)",
        "version": settings.VERSION,
        "docs": "/docs",
        "domains": {
            "movies": {"api_v1": api_v1, "service": "CineAI"},
            "music": {
                "api_direct": "/api/music",
                "api_v1": f"{api_v1}/music",
                "service": "MusicAI",
            },
        },
        "status": "online",
    }
