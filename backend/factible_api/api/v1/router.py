from fastapi import APIRouter

from factible_api.api.v1.endpoints import fact_check, health

api_router = APIRouter()

api_router.include_router(health.router, prefix="", tags=["health"])
api_router.include_router(fact_check.router, prefix="", tags=["fact-check"])
