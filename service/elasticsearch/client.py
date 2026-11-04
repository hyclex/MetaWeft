"""REST storage operations with caller-owned mappings and search queries."""

from urllib.parse import quote

import requests


class StoreSearchError(RuntimeError):
    """A search request or response failed; no result list is usable."""


class StoreWriteError(RuntimeError):
    """A mutation could not be confirmed; its actual store outcome is uncertain."""

    def __init__(self, operation, document_id):
        super().__init__(f"Could not confirm Elasticsearch {operation} for {document_id}")
        self.operation = operation
        self.document_id = document_id
        self.uncertain = True


class ElasticsearchStore:
    def __init__(self, url, index, *, session=None, timeout=30):
        self.url = url.rstrip("/")
        self.index = index
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout

    def _request(self, method, path, **kwargs):
        response = self.session.request(
            method, self.url + path, timeout=self.timeout, **kwargs,
        )
        response.raise_for_status()
        return response.json()

    @property
    def _index_path(self):
        return "/" + quote(self.index, safe="")

    def create(self, body):
        """Create a fresh index with the caller's settings and mappings."""
        return self._request("PUT", self._index_path, json=body)

    def get(self, document_id):
        """Return the unmodified source document, or None when its ID is absent."""
        path = self._index_path + "/_doc/" + quote(document_id, safe="")
        response = self.session.request("GET", self.url + path, timeout=self.timeout)
        if response.status_code == 404:
            body = response.json()
            if body.get("found") is False:
                return None
        response.raise_for_status()
        return response.json()["_source"]

    def put(self, document_id, document):
        """Write a fixed ID and confirm its complete source before returning."""
        path = self._index_path + "/_doc/" + quote(document_id, safe="")
        failure = None
        try:
            self._request("PUT", path, params={"refresh": "wait_for"}, json=document)
        except (requests.RequestException, ValueError) as error:
            failure = error
        try:
            observed = self.get(document_id)
            if observed == document:
                # A timed-out write may have committed; make it searchable now.
                if failure is not None:
                    self._request("POST", self._index_path + "/_refresh")
                return observed
        except (requests.RequestException, ValueError, KeyError) as error:
            failure = error
        raise StoreWriteError("put", document_id) from failure

    def delete(self, document_id):
        """Delete a fixed ID and confirm absence before returning."""
        path = self._index_path + "/_doc/" + quote(document_id, safe="")
        failure = None
        try:
            self._request("DELETE", path, params={"refresh": "wait_for"})
        except (requests.RequestException, ValueError) as error:
            failure = error
        try:
            if self.get(document_id) is None:
                if failure is not None:
                    self._request("POST", self._index_path + "/_refresh")
                return
        except (requests.RequestException, ValueError, KeyError) as error:
            failure = error
        raise StoreWriteError("delete", document_id) from failure

    def query(self, body):
        """Execute the caller's search body and return the complete ES response."""
        try:
            response = self._request(
                "POST", self._index_path + "/_search",
                params={"allow_partial_search_results": "false"}, json=body,
            )
        except (requests.RequestException, ValueError) as error:
            raise StoreSearchError(f"Elasticsearch search failed: {type(error).__name__}: {error}") from error
        self._check_search(response)
        return response

    @staticmethod
    def _check_search(response):
        if not isinstance(response, dict) or not isinstance(response.get("_shards", {}), dict):
            raise StoreSearchError("Elasticsearch search returned an invalid response")
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise StoreSearchError("Elasticsearch search returned incomplete results")

    def export(self, *, page_size=1000):
        """Return every {id, document} without a result-window truncation."""
        records = []
        scroll_id = None
        try:
            response = self._request(
                "POST", self._index_path + "/_search",
                params={"scroll": "1m", "allow_partial_search_results": "false"},
                json={"query": {"match_all": {}}, "sort": ["_doc"], "size": page_size,
                      "track_total_hits": True},
            )
            total = response["hits"].get("total")
            while True:
                scroll_id = response.get("_scroll_id", scroll_id)
                self._check_search(response)
                hits = response["hits"]["hits"]
                records.extend({"id": hit["_id"], "document": hit["_source"]} for hit in hits)
                if not hits:
                    break
                if not scroll_id:
                    raise ValueError("Elasticsearch omitted the scroll ID; export may be incomplete")
                response = self._request(
                    "POST", "/_search/scroll", json={"scroll": "1m", "scroll_id": scroll_id},
                )
            if isinstance(total, dict) and (total.get("relation") != "eq" or total.get("value") != len(records)):
                raise ValueError("Elasticsearch export count differs from its exact total")
            if isinstance(total, int) and total != len(records):
                raise ValueError("Elasticsearch export count differs from its exact total")
        finally:
            if scroll_id:
                try:
                    self._request("DELETE", "/_search/scroll", json={"scroll_id": [scroll_id]})
                except requests.RequestException:
                    # Scroll contexts expire; cleanup cannot turn partial data into a snapshot.
                    pass
        return sorted(records, key=lambda item: item["id"])

    def close(self):
        self.session.close()
