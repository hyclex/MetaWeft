# Elasticsearch service API

`ElasticsearchStore` uses `requests` and Elasticsearch's REST API. The caller owns
the index name, mappings, document shape, embedding generation, and query policy.

```python
from MetaWeft.service.elasticsearch import ElasticsearchStore

store = ElasticsearchStore("http://127.0.0.1:9200", "unique-run-and-split-index")
store.create({"mappings": {"properties": {"content": {"type": "text"}}}})
store.put("m1", {"content": "procedure", "extra": {"neighbors": ["m2"]}})
record = store.get("m1")
response = store.query({"query": {"match": {"content": "procedure"}}, "size": 5})
snapshot = store.export()  # [{"id": "m1", "document": {...}}, ...]
store.delete("m1")
store.close()
```

- `create(body)` creates a fresh index using the complete settings/mappings body.
  Existing-index errors propagate; it never resets an index.
- `get(id)` returns its unmodified `_source`, or `None` for an absent document.
  A missing index remains an error.
- `put(id, document)` replaces that stable ID, waits for search visibility, then
  reads back and returns the confirmed document. Documents are not modified.
- `delete(id)` waits for search visibility and confirms absence.
- `query(body)` returns the complete search response, including IDs and scores.
  Timed-out searches and partial shard results raise instead of becoming retrievals.
- `export(page_size=1000)` scrolls through all documents and returns `{id,
  document}` entries sorted by ID. A failed page raises; no partial snapshot is
  returned. The count is checked against Elasticsearch's exact total. The scroll
  context is closed, or allowed to expire if cleanup fails.
- `StoreWriteError` means a mutation could not be confirmed. Its `operation`,
  `document_id`, and `uncertain=True` identify the unresolved outcome. Do not assume
  the write failed to commit. Preserve the uncertainty in run artifacts.
- Constructor options are `session` (injected Requests-compatible session) and
  `timeout` (30 seconds). No automatic retries or embedding calls are performed.

For exact vector snapshots/readback on Elasticsearch 9.2, the variant's index
settings must preserve original vectors in `_source`; see [DESIGNS.md](DESIGNS.md).
