# Langfuse Observability

Instrument LLM applications with Langfuse tracing, following best practices and tailored to your use case.

## Workflow

### 1. Assess Current State

Check the project:

- Is Langfuse SDK installed?
- What LLM frameworks are used? (OpenAI SDK, LangChain, LlamaIndex, Vercel AI SDK, etc.)
- Is there existing instrumentation?

**No integration yet:** Set up Langfuse using a framework integration if available. Integrations capture more context automatically and require less code than manual instrumentation.

**Integration exists:** Audit against baseline requirements below.

### 2. Verify Baseline Requirements

Every trace should have these fundamentals:

| Requirement | Check | Why |
| ------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Model name | Is the LLM model captured? | Enables model comparison and filtering |
| Token usage | Are input/output tokens tracked? | Enables automatic cost calculation |
| Good trace names | Are names descriptive? (`chat-response`, not `trace-1`) | Makes traces findable and filterable |
| Span hierarchy | Are multi-step operations nested properly? | Shows which step is slow or failing |
| Correct observation types | Are generations marked as generations? | Enables model-specific analytics |
| Sensitive data masked | Is PII/confidential data excluded or masked? | Prevents data leakage |
| Trace input/output | Does the trace capture meaningful input/output? | Makes traces readable in the UI |

Framework integrations (OpenAI, LangChain, etc.) handle model name, tokens, and observation types automatically. Prefer integrations over manual instrumentation.

### 3. Run and Self-Audit the Traces (required)

**a.** Execute the instrumented path end-to-end so a trace is actually sent.

**b.** Fetch the trace(s) you just created from Langfuse.

**c.** Audit the trace against: https://langfuse.com/docs/observability/best-practices

**d.** Fix every gap you find, then re-run and re-fetch to confirm.

## Common Mistakes

| Mistake | Problem | Fix |
| ---------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| No `flush()`/`shutdown()` in scripts | Traces never sent | Call `sdk.shutdown()` before exit |
| Flat traces | Can't see which step failed | Use nested spans for distinct steps |
| Generic trace names | Hard to filter | Use descriptive names: `chat-response`, `doc-summary` |
| Langfuse import before env vars loaded | Wrong credentials | Import Langfuse AFTER loading environment variables |
| Manual instrumentation when integration exists | More code, less context | Use framework integration |
