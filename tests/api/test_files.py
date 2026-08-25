"""Endpoint tests for GET /ai/files."""

from types import SimpleNamespace


class TestListFiles:
    def test_list_populated(self, client, files_route, monkeypatch):
        details = [
            SimpleNamespace(
                filename="a.xlsx",
                uploaded_by="alice",
                uploaded_at="2020-01-01T00:00:00+00:00",
            ),
            SimpleNamespace(filename="b.pdf", uploaded_by=None, uploaded_at=None),
        ]
        monkeypatch.setattr(
            files_route, "get_storage_destination", lambda: ("data", "/srv/data")
        )
        monkeypatch.setattr(files_route, "list_stored_file_details", lambda: details)

        resp = client.get("/ai/files")

        assert resp.status_code == 200
        body = resp.json()
        assert body["destination"] == "data"
        assert body["location"] == "/srv/data"
        assert body["files"] == [
            {
                "filename": "a.xlsx",
                "uploaded_by": "alice",
                "uploaded_at": "2020-01-01T00:00:00+00:00",
            },
            {"filename": "b.pdf", "uploaded_by": None, "uploaded_at": None},
        ]

    def test_list_empty(self, client, files_route, monkeypatch):
        monkeypatch.setattr(
            files_route, "get_storage_destination", lambda: ("aws", "s3://bucket")
        )
        monkeypatch.setattr(files_route, "list_stored_file_details", lambda: [])

        resp = client.get("/ai/files")

        assert resp.status_code == 200
        body = resp.json()
        assert body["destination"] == "aws"
        assert body["files"] == []

    def test_list_error_returns_500(self, client, files_route, monkeypatch):
        def boom():
            raise RuntimeError("s3 unreachable")

        monkeypatch.setattr(
            files_route, "get_storage_destination", lambda: ("data", "/srv/data")
        )
        monkeypatch.setattr(files_route, "list_stored_file_details", boom)

        resp = client.get("/ai/files")

        assert resp.status_code == 500
