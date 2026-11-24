# @vasikmansuri/calling-agent-toolkit

TypeScript distribution of the Universal Calling Agent Toolkit: provider-neutral telephony adapters, audio pipeline, email capture, STT and RAG building blocks for an **existing** AI calling agent. It has zero runtime dependencies and needs no Python.

```ts
import { AudioPipeline, EmailCapture } from "@vasikmansuri/calling-agent-toolkit";
import { TwilioMediaTransport } from "@vasikmansuri/calling-agent-toolkit/twilio";
```

Entry points: `.`, `./twilio`, `./plivo`, `./telnyx`, `./vonage`, `./deepgram`, `./openai`.

Requires Node 20 or later. ESM only.

See the repository's `docs/` for architecture, provider compatibility (including what is **not** supported), and security guidance.
