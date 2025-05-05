from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.tortoiseorm import TortoiseORMInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

resource = Resource(attributes={
    SERVICE_NAME: 'oaas-products'
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)
tracer = trace.get_tracer(__name__)

TortoiseORMInstrumentor().instrument(tracer_provider=provider)
RedisInstrumentor().instrument(tracer_provider=provider)
HTTPXClientInstrumentor().instrument(tracer_provider=provider)