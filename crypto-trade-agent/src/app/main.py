from fastapi import FastAPI
from typing import Any

from .config import settings
from .routes import agent_routes, user_routes, portfolio_routes, market_routes, trade_routes

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for interacting with the crypto trading agent and related services.",
    version="0.1.0",
)

# Include API routers
app.include_router(agent_routes.router, prefix=settings.API_V1_STR)
app.include_router(user_routes.router, prefix=settings.API_V1_STR)
app.include_router(portfolio_routes.router, prefix=settings.API_V1_STR)
app.include_router(market_routes.router, prefix=settings.API_V1_STR)
app.include_router(trade_routes.router, prefix=settings.API_V1_STR)

@app.get("/")
async def read_root():
    """Root endpoint providing basic API information."""
    return {
        "message": f"Welcome to the {settings.APP_NAME}",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_STR
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }

# If running directly using uvicorn for development:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
