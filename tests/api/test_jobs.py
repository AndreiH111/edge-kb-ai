"""Endpoint tests for GET /ai/jobs and GET /ai/jobs/{job_id}."""

from datetime import datetime, timezone


class TestListJobs:
    def test_list_jobs(self, client, jobs_route, make_job, monkeypatch):
        job = make_job(
            job_id="j1",
            status="complete",
            job_type="index",
            filename="a.xlsx",
            indexed_records=5,
            created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2020, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(jobs_route.job_queue, "get_all_jobs", lambda: [job])

        resp = client.get("/ai/jobs")

        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j1"
        assert jobs[0]["status"] == "complete"
        assert jobs[0]["type"] == "index"
        assert jobs[0]["indexed_records"] == 5
        assert jobs[0]["created_at"] == "2020-01-01T00:00:00+00:00"
        assert jobs[0]["completed_at"] == "2020-01-01T00:00:05+00:00"

    def test_list_jobs_empty(self, client, jobs_route, monkeypatch):
        monkeypatch.setattr(jobs_route.job_queue, "get_all_jobs", lambda: [])

        resp = client.get("/ai/jobs")

        assert resp.status_code == 200
        assert resp.json()["jobs"] == []


class TestGetJob:
    def test_get_job_found(self, client, jobs_route, make_job, monkeypatch):
        job = make_job(job_id="j2", status="running", completed_at=None)
        monkeypatch.setattr(
            jobs_route.job_queue,
            "get_job",
            lambda job_id: job if job_id == "j2" else None,
        )

        resp = client.get("/ai/jobs/j2")

        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "j2"
        assert body["status"] == "running"
        assert body["completed_at"] is None

    def test_get_job_not_found_returns_404(self, client, jobs_route, monkeypatch):
        monkeypatch.setattr(jobs_route.job_queue, "get_job", lambda job_id: None)

        resp = client.get("/ai/jobs/unknown-id")

        assert resp.status_code == 404
