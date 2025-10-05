from fastapi import APIRouter, Depends

from factible_api.core.config import Settings, get_settings
from factible_api.schemas.v1.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=settings.version)
