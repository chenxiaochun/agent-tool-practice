import 'dotenv/config';
import { initLangfuseTracing, shutdownLangfuseTracing } from './tracing.mjs';
import { startActiveObservation } from '@langfuse/tracing';

initLangfuseTracing();

const traceId = await startActiveObservation('langfuse-connectivity-check', async (span) => {
  span.update({
    input: { message: 'Langfuse tracing smoke test' },
    output: { ok: true },
    metadata: { source: 'agent-tool-practice' },
  });
  return span.otelSpan.spanContext().traceId;
});

console.log(`Connectivity check trace id: ${traceId}`);
await shutdownLangfuseTracing();
console.log('Langfuse flush completed');
