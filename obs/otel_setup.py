from __future__ import annotations
 
import logging
import os
 
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pythonjsonlogger.json import JsonFormatter

_configured = False


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "json").lower() == "json":
        handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        handlers=[handler],
        force=True,
    )


def _resolve_connection_string() -> str | None:
    explicit = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if explicit:
        return explicit

    instrumentation_key = os.getenv("AML_APP_INSIGHTS_KEY")
    endpoint = os.getenv("AML_APP_INSIGHTS_ENDPOINT")
    if instrumentation_key and endpoint:
        return f"InstrumentationKey={instrumentation_key};IngestionEndpoint={endpoint}"
    return None
 
 
def configure_observability(service_name: str) -> None:
    """Configure logs and tracing once per process."""
    global _configured
    _configure_logging()
    if _configured:
        return

    connection_string = _resolve_connection_string()
    if connection_string:
        configure_azure_monitor(connection_string=connection_string)
        _configured = True
        return
 
    # Local fallback: console spans make development visible without Azure resources.
    resource = Resource.create({"service.name": service_name})

    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(metric_provider)
    _configured = True
