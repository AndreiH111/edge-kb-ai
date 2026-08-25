"""Endpoint tests for DELETE /ai/files.

``indexed_records`` comes from ``len(embed_service.documents)`` (delete.py:36); the
session fake ``embed_service`` exposes a ``documents`` list that tests set per case.
The router imports ``file_service`` helpers by name, so patches target the route
module (``delete_route``). DELETE carries a JSON body, so requests use
``client.request("DELETE", ...)`` (httpx's ``.delete`` takes no body).
"""


class TestDeleteFiles:
    def test_delete_mixed_deleted_and_not_found(
        self, client, delete_route, monkeypatch
    ):
        existing = {"a.xlsx"}
        monkeypatch.setattr(delete_route, "file_exists", lambda name: name in existing)
        monkeypatch.setattr(delete_route, "delete_file_from_storage", lambda name: None)
        monkeypatch.setattr(
            delete_route.file_metadata_store, "remove_metadata", lambda name: None
        )
        monkeypatch.setattr(delete_route.embed_service, "documents", [1, 2, 3])

        resp = client.request(
            "DELETE", "/ai/files", json={"filenames": ["a.xlsx", "b.pdf"]}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == ["a.xlsx"]
        assert body["not_found"] == ["b.pdf"]
        assert body["indexed_records"] == 3
        assert body["message"]

    def test_delete_all_not_found(self, client, delete_route, monkeypatch):
        monkeypatch.setattr(delete_route, "file_exists", lambda name: False)
        monkeypatch.setattr(delete_route.embed_service, "documents", [])

        resp = client.request("DELETE", "/ai/files", json={"filenames": ["x.xlsx"]})

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == []
        assert body["not_found"] == ["x.xlsx"]
        assert body["indexed_records"] == 0

    def test_delete_error_returns_500(self, client, delete_route, monkeypatch):
        def boom(name):
            raise RuntimeError("backend failure")

        monkeypatch.setattr(delete_route, "file_exists", boom)

        resp = client.request("DELETE", "/ai/files", json={"filenames": ["a.xlsx"]})

        assert resp.status_code == 500

    def test_delete_empty_filenames_returns_422(self, client):
        """``DeleteFilesRequest.filenames`` has ``min_length=1``."""
        resp = client.request("DELETE", "/ai/files", json={"filenames": []})

        assert resp.status_code == 422

    def test_delete_missing_body_returns_422(self, client):
        resp = client.request("DELETE", "/ai/files")

        assert resp.status_code == 422
