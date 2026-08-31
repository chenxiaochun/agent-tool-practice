import dotenv from 'dotenv';

dotenv.config();

import { NodeSDK } from '@opentelemetry/sdk-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { CallbackHandler } from '@langfuse/langchain';
import { propagateAttributes, startActiveObservation } from '@langfuse/tracing';

let sdk;
let spanProcessor;

/**
 * Initialize OpenTelemetry + Langfuse. Must run before any LangChain/LangGraph calls.
 * Safe to call multiple times.
 */
export function initLangfuseTracing() {
  if (sdk) 
    return { sdk, spanProcessor };
  }

  spanProcessor = new LangfuseSpanProcessor({
    environment: process.env.LANGFUSE_TRACING_ENVIRONMENT ?? process.env.NODE_ENV ?? 'development',
    exportMode: 'immediate',
  });

  sdk = new NodeSDK({
    spanProcessors: [spanProcessor],
  });
  sdk.start();

  return { sdk, spanProcessor };
}

/**
 * Create a LangChain CallbackHandler with project defaults.
 */
export function createLangfuseHandler(options = {}) {
  const { tags = [], ...rest } = options;

  return new CallbackHandler({
    tags: ['agent-tool-practice', ...tags],
    ...rest,
  });
}

/**
 * Wrap a LangGraph/LangChain run with trace-level input/output and attributes.
 */
export async function traceLangChainRun(name, input, run, options = {}) {
  initLangfuseTracing();

  const { tags = [], sessionId, userId, metadata = {} } = options;

  return startActiveObservation(name, async (span) => {
    span.update({ input, metadata });

    return propagateAttributes({ sessionId, userId, tags }, async () => {
      const handler = createLangfuseHandler({ sessionId, userId, tags });
      const output = await run(handler);
      span.update({ output });
      return { output, traceId: handler.last_trace_id };
    });
  });
}

/**
 * Flush and shut down tracing. Required for short-lived CLI scripts.
 */
export async function shutdownLangfuseTracing() {
  if (spanProcessor) {
    await spanProcessor.forceFlush();
    await spanProcessor.shutdown();
  }

  if (sdk) {
    await sdk.shutdown();
  }
}
