"""FastAPI Application Entry Point for CineAI."""

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
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-performance REST API for CineAI: 10 AI Movie Recommendation Models on MovieLens 100K",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 Routers
api_v1 = settings.API_V1_STR
app.include_router(health.router, prefix=api_v1)
app.include_router(models.router, prefix=api_v1)
app.include_router(movies.router, prefix=api_v1)
app.include_router(users.router, prefix=api_v1)
app.include_router(recommendations.router, prefix=api_v1)
app.include_router(metrics.router, prefix=api_v1)
app.include_router(compare.router, prefix=api_v1)


@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": api_v1,
        "status": "online",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safely catch unhandled server errors without leaking internals."""
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
