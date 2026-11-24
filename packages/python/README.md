# calling-agent-toolkit (Python)

Python distribution of the Universal Calling Agent Toolkit: provider-neutral telephony adapters, audio pipeline, email capture, STT and RAG building blocks for an **existing** AI calling agent. It has zero runtime dependencies and needs no Node.js. Requires Python 3.10 or later; typed (`py.typed`).

```python
from calling_agent_toolkit.audio_pipeline import AudioPipeline
from calling_agent_toolkit.email_capture import EmailCapture
from calling_agent_toolkit.providers.twilio import TwilioMediaTransport
```

Provider modules: `calling_agent_toolkit.providers.{twilio,plivo,telnyx,vonage,deepgram,openai}`.

See the repository's `docs/` for architecture, provider compatibility (including what is **not** supported), and security guidance.
