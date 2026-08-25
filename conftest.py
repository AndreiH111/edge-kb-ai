"""Session-wide offline wiring, loaded by pytest before any test module.

This is the earliest hook pytest runs, so it executes *before* ``core.config``
builds its frozen ``settings`` singleton. It neutralizes the working-tree ``.env``
(which points at live AWS credentials and enables the agent/Jira) and pins safe,
network-free settings, guaranteeing the whole suite stays offline no matter which
test module is collected first. See CLAUDE.md for why importing the app is
otherwise heavy and non-hermetic.
"""

import os
import tempfile

# Stop core.config from loading the real .env. core.config does
# ``from dotenv import load_dotenv`` *inside* a function, so replacing the
# attribute here — before core.config is imported — makes that call a no-op and
# the working-tree .env is never read.
import dotenv

dotenv.load_dotenv = lambda *args, **kwargs: None

# Pin offline settings. Set in os.environ before the first
# ``from core.config import settings`` so ``Settings()`` reads these values.
os.environ.update(
    {
        "FILE_STORAGE_BACKEND": "local",
        "DATA_DIR": tempfile.mkdtemp(prefix="edge-kb-test-data-"),
        "VECTOR_DB_BACKEND": "chroma",
        "CHROMA_MODE": "local",
        "EMBEDDING_MODE": "custom",
        "CHUNKING_MODE": "local",
        "AGENT_ENABLED": "false",
        "JIRA_ENABLED": "false",
    }
)
