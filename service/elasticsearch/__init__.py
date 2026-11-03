"""Policy-neutral Elasticsearch document operations."""

from .client import ElasticsearchStore, StoreWriteError

__all__ = ["ElasticsearchStore", "StoreWriteError"]
