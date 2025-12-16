import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from factible_api.api.v1.router import api_router
from factible_api.core.config import get_settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    _logger.info(f"Starting {settings.project_name} v{settings.version}")
    yield
    _logger.info(f"Shutting down {settings.project_name}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description="AI-powered YouTube video fact-checking API with real-time streaming",
        docs_url="/docs",
        redoc_url=None,  # Disable ReDoc
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        expose_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # Root endpoint
    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "message": f"Welcome to {settings.project_name}",
                "version": settings.version,
                "docs": "/docs",
                "api": settings.api_v1_prefix,
                "health": f"{settings.api_v1_prefix}/health",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "factible_api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.debug,
    )
