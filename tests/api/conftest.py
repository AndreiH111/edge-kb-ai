"""Fixtures for the HTTP-level endpoint suite.

Importing ``main`` transitively imports ``services.embed_service``, whose module
body builds real embeddings + a Chroma store and runs ``initialize_on_startup``
(see CLAUDE.md). The ``app`` fixture injects a lightweight fake ``embed_service``
into ``sys.modules`` *before* importing ``main`` — mirroring the
``inject_services`` philosophy in ``tests/unit/test_agent_service.py`` — so the
import stays cheap and offline. ``rag_service`` (the agent's fallback) becomes
cheap once ``embed_service`` is faked; nothing else needs faking.

The ``client`` uses ``TestClient`` *without* the ``with`` block, so the app
lifespan never runs: no job-queue worker thread and no agent/MCP startup. Every
scenario is driven per-test with ``monkeypatch`` on the specific service function.
"""

import contextlib
import sys
import types
from contextvars import ContextVar
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

_MISSING = object()


def _make_fake_embed_service() -> types.ModuleType:
    """Build a stand-in for ``services.embed_service`` (all heavy work removed)."""
    module = types.ModuleType("services.embed_service")
    module.documents = []
    module.remove_file = MagicMock(name="remove_file")
    module.index_file = MagicMock(name="index_file", return_value=1)
    module.refresh_vector_store = MagicMock(name="refresh_vector_store", return_value=0)
    module.retrieve_from_all_sources = MagicMock(
        name="retrieve_from_all_sources", return_value=[]
    )
    module.get_indexed_sources = MagicMock(name="get_indexed_sources", return_value=[])
    module.pinned_source = ContextVar("pinned_source", default=None)
    module.use_source = lambda source: contextlib.nullcontext()
    module.initialize_on_startup = MagicMock(name="initialize_on_startup")
    return module


@pytest.fixture(scope="session")
def app():
    """Import the FastAPI app with a fake ``embed_service`` in place.

    Sets both the ``sys.modules`` entry and the ``services`` package attribute so a
    lazy ``from services import embed_service`` resolves to the fake, then restores
    the original state on teardown.
    """
    import services as services_pkg

    fake = _make_fake_embed_service()
    saved_module = sys.modules.get("services.embed_service", _MISSING)
    saved_attr = getattr(services_pkg, "embed_service", _MISSING)
    sys.modules["services.embed_service"] = fake
    services_pkg.embed_service = fake

    try:
        from main import app as fastapi_app

        yield fastapi_app
    finally:
        if saved_module is _MISSING:
            sys.modules.pop("services.embed_service", None)
        else:
            sys.modules["services.embed_service"] = saved_module

        if saved_attr is _MISSING:
            if hasattr(services_pkg, "embed_service"):
                delattr(services_pkg, "embed_service")
        else:
            services_pkg.embed_service = saved_attr


@pytest.fixture
def client(app) -> TestClient:
    """A ``TestClient`` with no lifespan (no worker thread, no agent startup)."""
    return TestClient(app)


# ---- Route-module fixtures --------------------------------------------------
# The routers import service helpers *by name* (``from services.file_service
# import file_exists``), so tests must patch the binding on the route module, not
# on ``services.*``. These fixtures import the route module — safe only because
# they depend on ``app``, which has already imported ``main`` with the fake in
# place; a top-level import would pull in the real ``embed_service``.


@pytest.fixture
def upload_route(app):
    import api.routes.upload as module

    return module


@pytest.fixture
def delete_route(app):
    import api.routes.delete as module

    return module


@pytest.fixture
def files_route(app):
    import api.routes.files as module

    return module


@pytest.fixture
def jobs_route(app):
    import api.routes.jobs as module

    return module


# ---- Factory fixtures -------------------------------------------------------


@pytest.fixture
def make_job():
    """Build a fake ``Job`` (as the routers see it) with real ``datetime``s."""

    def _make(
        job_id="job-1",
        status="complete",
        job_type="index",
        filename="report.xlsx",
        error=None,
        indexed_records=3,
        created_at=None,
        completed_at=None,
    ):
        return SimpleNamespace(
            id=job_id,
            status=status,
            type=job_type,
            filename=filename,
            error=error,
            indexed_records=indexed_records,
            created_at=created_at or datetime(2020, 1, 1, tzinfo=timezone.utc),
            completed_at=completed_at,
        )

    return _make


@pytest.fixture
def make_file_detail():
    """Build a fake ``FileDetail`` (duck-typed via ``SimpleNamespace``)."""

    def _make(filename="report.xlsx", uploaded_by=None, uploaded_at=None):
        return SimpleNamespace(
            filename=filename, uploaded_by=uploaded_by, uploaded_at=uploaded_at
        )

    return _make
