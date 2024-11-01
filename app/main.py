from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api.routes import router
from .core.config import get_settings
import logging
from time import time
from typing import Callable, Dict
import uvicorn
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Medical Text to JSON Parser API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Callable):
    start_time = time()

    # Basic rate limiting
    if hasattr(request.state, 'requests'):
        request.state.requests += 1
        if request.state.requests > settings.RATE_LIMIT_REQUESTS:
            raise HTTPException(status_code=429, detail="Too many requests")
    else:
        request.state.requests = 1

    response = await call_next(request)
    process_time = time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": str(type(exc).__name__)
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Check API health status and basic system information.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "service": settings.APP_NAME,
        "timestamp": str(time())
    }


# Include API routes
app.include_router(
    router,
    prefix=settings.API_V1_STR,
    tags=["Medical Text Processing"]
)


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME}")
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found in environment variables!")
    logger.info("Server is ready to accept requests")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.APP_NAME}")


def start_server(host: str = "0.0.0.0",
                 port: int = 8000,
                 reload: bool = True,
                 workers: int = 1,
                 log_level: str = "info"):
    """Start the FastAPI server with the specified configuration."""
    try:
        logger.info(f"Starting server on http://{host}:{port}")
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser(description='Run the Medical Text Processing API server')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on code changes')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--log-level', type=str, default="info",
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')

    args = parser.parse_args()

    # Start server with command line arguments
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )