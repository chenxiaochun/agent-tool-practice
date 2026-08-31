---
name: langfuse
description: Interact with Langfuse and access its documentation. Use when needing to (1) query or modify Langfuse data programmatically via the CLI — traces, prompts, datasets, scores, sessions, and any other API resource, (2) look up Langfuse documentation, concepts, integration guides, or SDK usage, or (3) understand how any Langfuse feature works. This skill covers CLI-based API access (via npx) and multiple documentation retrieval methods.
---

# Langfuse

This skill helps you use Langfuse effectively across all common workflows: instrumenting applications, migrating prompts, debugging traces, and accessing data programmatically.

## Core Principles

Follow these principles for ALL Langfuse work:

1. **Documentation First**: NEVER implement based on memory. Always fetch current docs before writing code (Langfuse updates frequently) See the section below on how to access documentation.
2. **CLI for Data Access**: Use `langfuse-cli` when querying/modifying Langfuse data. See the section below on how to use the CLI.
3. **Best Practices by Use Case**: Check the relevant reference file below for use-case-specific guidelines before implementing
4. **Use latest Langfuse versions**: Unless the user specified otherwise or there's a good reason, always use the latest version of Langfuse SDKs/APIs. Even if you're only creating a plan for another agent to execute, be explicit about the exact version to use.
5. **If you guide the user through UI** and are unsure about a label or location, inspect the user's screenshots or ask to see the relevant screen. Do not assume UI labels have the exact same names as API, SDK, or CLI fields.

## Use case specific references

- instrumenting an existing function/application: references/instrumentation.md
- migrating prompts from a codebase into Langfuse: references/prompt-migration.md
- creating a prompt or changing any part of an existing prompt, including small edits and debugging/tuning: references/prompt-engineering.md
- capturing user feedback (thumbs, ratings, implicit signals) as scores on traces: references/user-feedback.md
- further tips on using the Langfuse CLI: references/cli.md

## 1. Langfuse API via CLI

Use the `langfuse-cli` to interact with the full Langfuse REST API from the command line. Run via npx (no install required):

```bash
npx langfuse-cli api __schema
npx langfuse-cli api <resource> --help
npx langfuse-cli api <resource> <action> --help
```

### Credentials

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

If `LANGFUSE_BASE_URL` is used instead of `LANGFUSE_HOST`, run `export LANGFUSE_HOST="$LANGFUSE_BASE_URL"`.

## 2. Langfuse Documentation

1. Start with **llms.txt** — `https://langfuse.com/llms.txt`
2. **Fetch specific pages** — append `.md` to doc URLs
3. Fall back to **search** — `https://langfuse.com/api/search-docs?query=...`
