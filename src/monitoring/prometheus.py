import litserve
from prometheus_client import CollectorRegistry, Histogram, Gauge, Counter
import torch.cuda as cuda
from loguru import logger

class PrometheusLogger(litserve.Logger):
    def __init__(self, registry: CollectorRegistry = None):
        super().__init__()
        self.registry = registry or CollectorRegistry()

        self.function_duration = Histogram(
            "request_processing_seconds",
            "Time spent processing request",
            ["function_name"], registry=registry
        )

        self.api_requests = Counter(
            "api_requests_total",
            "Total API requests handled",
            ["endpoint", "method", "status"], registry=registry
        )

        self.api_response_time = Histogram(
            "api_response_time_seconds",
            "Response time of API endpoints",
            ["endpoint"], registry=registry
        )

        self.model_predictions = Counter(
            "model_predictions_total",
            "Total number of model predictions",
            ["predicted_class", "confidence_level"], registry=registry
        )

        self.model_memory_usage = Gauge(
            "model_memory_usage_bytes",
            "Memory used by the model",
            ["device_type"], registry=registry
        )

        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "Overall system memory usage",
            ["memory_type"], registry=registry
        )

        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage by the process and system",
            ["cpu_type"], registry=registry
        )

    def process(self, key, value):
        try:
            if key == "model_prediction":
                predicted_class, confidence = value
                confidence_bucket = self._confidence_bucket(confidence)

                self.model_predictions.labels(
                    predicted_class=predicted_class,
                    confidence_level=confidence_bucket
                ).inc(1)
            elif key == "model_memory_allocated":
                self.model_memory_usage.labels(device_type="gpu" if cuda.is_available() else "cpu").set(value)
            elif key == "model_memory_peak":
                self.model_memory_usage.labels(device_type="gpu_peak" if cuda.is_available() else "cpu_peak").set(value)
            elif key.startswith("system_memory_"):
                memory_type = key.split("_")[-1]
                self.system_memory_usage.labels(memory_type=memory_type).set(value)
            elif key == "cpu_usage":
                self.cpu_usage.labels(cpu_type="system").set(value["system"])
                self.cpu_usage.labels(cpu_type="process").set(value["process"])
            elif key == "api_requests":
                endpoint, method, status = value["endpoint"], value["method"], value["status"]
                self.api_requests.labels(
                    endpoint=endpoint,
                    method=method,
                    status=str(status)
                ).inc(1)

                if "response_time" in value:
                    self.api_response_time.labels(
                        endpoint=endpoint
                    ).observe(value["response_time"])
            else:
                self.function_duration.labels(function_name=key).observe(value)
        except Exception as e:
            logger.error(
                f"PrometheusLogger ran into an error while processing log for key {key} and value {value}: {e}")

    @staticmethod
    def _confidence_bucket(confidence):
        if confidence < 0.5:
            return "low"
        elif confidence < 0.8:
            return "medium"
        else:
            return "high"