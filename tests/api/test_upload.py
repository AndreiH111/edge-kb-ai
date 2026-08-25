"""Endpoint tests for /ai/upload and /ai/reindex.

Without the lifespan, ``job_queue`` has no worker thread, so ``submit`` is patched
per-test and no background indexing runs — every response is deterministic. The
routers import ``file_service`` helpers by name, so patches target the route
module (``upload_route``). ``is_supported_file`` / ``sanitize_filename`` are left
real (pure extension/basename logic).
"""

from types import SimpleNamespace

# Excel content type for the multipart uploads.
XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


class TestUpload:
    def test_upload_happy_path(self, client, upload_route, monkeypatch):
        recorded = []
        submitted = []

        async def fake_save(upload_file, filename):
            return f"data/{filename}"

        def fake_record(filename, uploaded_by=None):
            recorded.append((filename, uploaded_by))

        def fake_submit(job_type, filename, task_fn):
            submitted.append((job_type, filename))
            return SimpleNamespace(id="job-42")

        monkeypatch.setattr(upload_route, "file_exists", lambda name: False)
        monkeypatch.setattr(upload_route, "save_upload_file", fake_save)
        monkeypatch.setattr(upload_route.file_metadata_store, "record_upload", fake_record)
        monkeypatch.setattr(upload_route.job_queue, "submit", fake_submit)

        resp = client.post(
            "/ai/upload",
            files={"file": ("report.xlsx", b"binary-bytes", XLSX_CONTENT_TYPE)},
            data={"uploaded_by": "alice"},
        )

        assert resp.status_code == 202
        assert resp.json() == {
            "job_id": "job-42",
            "filename": "report.xlsx",
            "path": "data/report.xlsx",
            "status": "queued",
        }
        assert recorded == [("report.xlsx", "alice")]  # metadata recorded
        assert submitted == [("index", "report.xlsx")]  # index job submitted

    def test_upload_empty_filename_returns_400(self, client, upload_route, monkeypatch):
        # httpx omits the filename when it is "", so Starlette would treat the part
        # as a plain form field (422). Hand-roll a multipart body with an explicit
        # ``filename=""`` to actually reach the route's ``if not file.filename`` guard.
        monkeypatch.setattr(upload_route, "file_exists", lambda name: False)
        boundary = "edgekbboundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename=""\r\n'
            "Content-Type: text/plain\r\n"
            "\r\n"
            "data\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        resp = client.post(
            "/ai/upload",
            content=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        assert resp.status_code == 400

    def test_upload_unsupported_extension_returns_400(
        self, client, upload_route, monkeypatch
    ):
        monkeypatch.setattr(upload_route, "file_exists", lambda name: False)
        resp = client.post(
            "/ai/upload", files={"file": ("notes.txt", b"data", "text/plain")}
        )

        assert resp.status_code == 400

    def test_upload_conflict_returns_409(self, client, upload_route, monkeypatch):
        monkeypatch.setattr(upload_route, "file_exists", lambda name: True)
        resp = client.post(
            "/ai/upload",
            files={"file": ("report.xlsx", b"data", XLSX_CONTENT_TYPE)},
        )

        assert resp.status_code == 409

    def test_upload_save_failure_returns_500(self, client, upload_route, monkeypatch):
        async def boom(upload_file, filename):
            raise RuntimeError("disk full")

        monkeypatch.setattr(upload_route, "file_exists", lambda name: False)
        monkeypatch.setattr(upload_route, "save_upload_file", boom)
        resp = client.post(
            "/ai/upload",
            files={"file": ("report.xlsx", b"data", XLSX_CONTENT_TYPE)},
        )

        assert resp.status_code == 500

    def test_upload_missing_file_returns_422(self, client):
        resp = client.post("/ai/upload")

        assert resp.status_code == 422


class TestReindex:
    def test_reindex_returns_202(self, client, upload_route, monkeypatch):
        submitted = []

        def fake_submit(job_type, filename, task_fn):
            submitted.append((job_type, filename))
            return SimpleNamespace(id="job-reindex")

        monkeypatch.setattr(upload_route.job_queue, "submit", fake_submit)
        resp = client.post("/ai/reindex")

        assert resp.status_code == 202
        assert resp.json() == {"job_id": "job-reindex", "status": "queued"}
        assert submitted == [("reindex", None)]
