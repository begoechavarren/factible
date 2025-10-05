import asyncio
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from factible.run import run_factible
from factible_api.core.config import Settings, get_settings
from factible_api.schemas.v1.requests import FactCheckRequest
from factible_api.schemas.v1.responses import ErrorResponse, ProgressUpdate

router = APIRouter(tags=["fact-check"])
_logger = logging.getLogger(__name__)


async def fact_check_stream(
    request: FactCheckRequest, settings: Settings
) -> AsyncGenerator[str, None]:
    """
    Stream fact-checking progress updates using Server-Sent Events (SSE).

    Thin wrapper around the factible pipeline that converts progress callbacks
    into SSE-formatted messages.
    """
    # Queue to collect progress updates from callback
    updates_queue: asyncio.Queue[ProgressUpdate | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_handler(step: str, message: str, progress: int, data: dict | None):
        """Callback that collects updates for SSE streaming."""
        asyncio.run_coroutine_threadsafe(
            updates_queue.put(
                ProgressUpdate(step=step, message=message, progress=progress, data=data)
            ),
            loop,
        )

    # Run pipeline in thread with callback
    async def run_pipeline():
        try:
            await asyncio.to_thread(
                run_factible,
                str(request.video_url),
                max_claims=request.max_claims,
                max_queries_per_claim=request.max_queries_per_claim,
                max_results_per_query=request.max_results_per_query,
                headless_search=settings.headless_search,
                progress_callback=progress_handler,
            )
        except Exception as exc:
            _logger.exception("Fact-checking pipeline failed: %s", exc)
            asyncio.run_coroutine_threadsafe(
                updates_queue.put(
                    ProgressUpdate(
                        step="error",
                        message=f"Fact-checking failed: {str(exc)}",
                        progress=100,
                        data={"error": str(exc)},
                    )
                ),
                loop,
            )
        finally:
            asyncio.run_coroutine_threadsafe(updates_queue.put(None), loop)

    # Start pipeline in background
    asyncio.create_task(run_pipeline())

    # Stream updates as they arrive
    while True:
        update = await updates_queue.get()
        if update is None:
            break
        yield _format_sse(update)
        if update.step in ("complete", "error"):
            break


def _format_sse(update: ProgressUpdate) -> str:
    """Format a progress update as an SSE message."""
    return f"data: {update.model_dump_json()}\n\n"


@router.post(
    "/fact-check/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Stream of progress updates",
            "content": {"text/event-stream": {"example": "data: {...}\n\n"}},
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def fact_check_stream_endpoint(
    request: FactCheckRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Fact-check a YouTube video with real-time progress updates via Server-Sent Events.

    Returns a stream of progress updates as the fact-checking pipeline processes the video.
    Each update includes the current step, a human-readable message, and progress percentage.

    The final update includes the complete fact-check results.
    """
    return StreamingResponse(
        fact_check_stream(request, settings),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )
