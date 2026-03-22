import os
import json
import logging
from functools import wraps
from typing import Callable, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource

_telemetry_initialized = False

class JsonLoggerSpanExporter(SpanExporter):
    """
    OpenTelemetry Span'larını yakalayıp, JSON formatında standart Python Logger'ına 
    aktaran özel Exporter. Bu sayede FileBeat/Logstash veya Kibana directly JSON okuyabilir.
    """
    def __init__(self, logger_name: str = "bio_ml_agent.telemetry"):
        self.logger = logging.getLogger(logger_name)

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            span_data = {
                "event_type": "otel_span",
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
                "name": span.name,
                "start_time_ns": span.start_time,
                "end_time_ns": span.end_time,
                "latency_ms": (span.end_time - span.start_time) / 1000000.0 if span.end_time and span.start_time else 0.0,
                "attributes": dict(span.attributes) if span.attributes else {},
                "status": span.status.status_code.name if span.status else "UNSET"
            }
            # INFO seviyesinde JSON dökümü (Formatter kendisi dict string basabilir veya string json_dumps basılır)
            self.logger.info(json.dumps(span_data))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

def setup_telemetry(service_name: str = "bio_ml_agent"):
    global _telemetry_initialized
    if _telemetry_initialized:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    # Custom Exporter to structured log
    processor = SimpleSpanProcessor(JsonLoggerSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    _telemetry_initialized = True

def get_tracer():
    return trace.get_tracer(__name__)

def otel_trace(span_name: str) -> Callable:
    """Metotları OpenTelemetry trace (Span) ile sarmalar."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not _telemetry_initialized:
                setup_telemetry()
                
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator

# Otomatik başlatma
setup_telemetry()
