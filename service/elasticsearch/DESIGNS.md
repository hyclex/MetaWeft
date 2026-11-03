# Elasticsearch service design

This is a new small REST wrapper over the existing prepared Elasticsearch 9.2
runtime. It uses Requests already available in the AppWorld environment, avoiding
installation into the immutable environment or a second memory-service process.
Service startup, isolation, credentials, and durable artifact paths belong to the
experiment layer. This package has no AppWorld or ReMe imports.

The wrapper does not inherit ReMe's fixed content/vector mapping, automatic
embeddings, metadata filters, silent bulk-failure handling, or 10,000-record export
limit. Caller-provided mappings and query bodies permit richer fields and future
multiple-index or hybrid designs without embedding their policy in this service.

Writes use caller-assigned stable IDs and `refresh=wait_for`. Every write is checked
by reading its complete source. After a transport failure, matching readback plus
an explicit refresh confirms the intended outcome; otherwise `StoreWriteError`
marks uncertainty. The caller must stop dependent updates and retain this state.
No write is retried automatically. Concurrent writers to one index are outside the
v1 protocol: independent streams must use distinct indices.

Exact source comparison requires storing the submitted document representation.
For Elasticsearch 9.2 vector fields, the base variant sets
`index.mapping.exclude_source_vectors=false`; schema configuration remains in the
variant. Export preserves full `_source`, reads every scroll page, and fails on
incomplete page retrieval. Evidence references are ordinary stored fields; only
variant code resolves them and enforces its feedback protocol.
