import os
import sys
from loguru import logger
import litserve as ls
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
from litserve.middlewares import MaxSizeMiddleware
from core.config import config
from api.classifier_serve import CervicalCellClassifierAPI
from src.monitoring.prometheus import PrometheusLogger
from middleware.metrics import MetricsMiddleware

def setup_prometheus():
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
    if not os.path.exists("/tmp/prometheus_multiproc_dir"):
        os.makedirs("/tmp/prometheus_multiproc_dir")

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )


def cleanup_metric_files():
    multiprocess_dir = "/tmp/prometheus_multiproc_dir"
    if os.path.exists(multiprocess_dir):
        for file in os.listdir(multiprocess_dir):
            os.remove(os.path.join(multiprocess_dir, file))


def main():
    try:
        cleanup_metric_files()
        registry = setup_prometheus()
        setup_logging()

        prometheus_logger = PrometheusLogger(registry)

        api = CervicalCellClassifierAPI()
        server = ls.LitServer(
            api,
            api_path="/api/v1/predict",
            accelerator="auto",
            max_batch_size=4,
            timeout=1,
            track_requests=True,
            devices="auto",
            loggers=[prometheus_logger]
        )

        server.app.mount("/api/v1/metrics", make_asgi_app(registry=registry))
        server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)
        server.app.add_middleware(MetricsMiddleware, prometheus_logger=prometheus_logger)

        logger.info("Starting server on port 8000")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()