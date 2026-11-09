# RAG integration

The toolkit retrieves and grounds. It never writes the answer: your agent receives a `GroundedContext` and decides what to say.

## Pieces

| Piece | Role |
|---|---|
| `Retriever` | Your backend (vector store, search, LangChain or LlamaIndex) |
| `SafeRetriever` | Wraps any backend. It enforces tenant and ACL even when the backend already filters, applies metadata filters, enforces a timeout, honours cancellation, ranks (score desc, then id), caps `topK`, and attaches citations. Results are always marked `untrusted: true` |
| `RetrievalCoordinator` | Decides **when** to retrieve. Final utterances only (never partials), a `minChars` threshold, and no repeat of the same query. A newer turn cancels the in-flight retrieval. A result that completes after a newer turn started comes back as `stale`, with no items, so it can't be attached to the wrong question |
| `formatContext` | Renders passages with an "untrusted data, never instructions" preamble. Content and attributes are escaped, so a passage cannot close its own `<passage>` tag |
| `InMemoryRetriever` | Keyword retriever for tests and small knowledge bases, with `upsert`, `delete` and `deleteDocument`. Ignores a built-in English stop-word list when scoring in Python and TypeScript; this is not multilingual or semantic search. |

## Statuses

`ok`, `empty`, `timeout`, `cancelled`, `error` (with a `ProviderError` payload), `stale`. Errors never throw into the call.

## Access control

`Chunk.acl === null` means the chunk is visible to the whole tenant. Otherwise the principal must be listed in `acl.principals` or share a group with `acl.groups`. Chunks from other tenants are always removed. `droppedForAccess` counts removed results, without ever exposing them.

## Framework adapters

| Adapter | Status |
|---|---|
| LangChain (Python `LangChainRetriever`, `ainvoke`) | **Tested against real `langchain-core`** (a dev dependency; test `test_real_langchain_core_retriever`) |
| LangChain.js (`LangChainRetrieverAdapter`, `invoke`) | Shape-tested with a stub; not run against the real library |
| LlamaIndex (Python `LlamaIndexRetriever`, `aretrieve`) | Implemented; not tested against llama-index |

Metadata keys read by the adapters: `tenant_id`, `document_id`, `acl_principals`, `acl_groups`, `title`, `source`, `page`. LangChain returns no scores, so the rank is used instead (`1/(rank+1)`).

## Not measured

Retrieval relevance has no ground-truth evaluation yet; the keyword retriever exists for tests, not quality. Latency depends entirely on your backend.
