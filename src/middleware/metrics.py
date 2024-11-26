from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, prometheus_logger):
        super().__init__(app)
        self.prometheus_logger = prometheus_logger

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            process_time = time.time() - start_time

            self.prometheus_logger.process("api_requests", {
                "endpoint": request.url.path,
                "method": request.method,
                "status": status_code,
                "response_time": process_time
            })

        return response