import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Extracts or generates a unique correlation ID (X-Request-ID).
    2. Attaches request_id to request.state.
    3. Measures request execution duration.
    4. Logs structured request details with request ID and timing.
    5. Appends X-Request-ID and X-Process-Time headers to the response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Resolve Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Let global exception handlers capture and format the response
            raise
        finally:
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Append correlation headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time_ms}ms"

        # Log completion
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({process_time_ms}ms) [request_id={request_id}]"
        )

        return response
