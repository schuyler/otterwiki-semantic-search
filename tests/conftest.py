"""Test fixtures for otterwiki-semantic-search."""

import os

import pytest
import otterwiki.gitstorage

# Env vars we set during tests — saved and restored in fixture teardown
_MANAGED_ENV_VARS = [
    "OTTERWIKI_SETTINGS",
    "CHROMADB_MODE",
    "CHROMADB_PATH",
    "CHROMA_SYNC_STATE_PATH",
    "CHROMA_SYNC_INTERVAL",
    "OTTERWIKI_API_KEY",
]


@pytest.fixture(scope="session")
def create_app(tmp_path_factory):
    # Save original env state
    saved_env = {k: os.environ.get(k) for k in _MANAGED_ENV_VARS}

    tmpdir = tmp_path_factory.mktemp("data")
    repo_dir = tmpdir / "repo"
    repo_dir.mkdir()
    _storage = otterwiki.gitstorage.GitStorage(
        path=str(repo_dir), initialize=True
    )
    settings_cfg = str(tmpdir / "settings.cfg")
    with open(settings_cfg, "w") as f:
        f.writelines(
            [
                "REPOSITORY = '{}'\n".format(str(_storage.path)),
                "SITE_NAME = 'TEST WIKI'\n",
                "DEBUG = True\n",
                "TESTING = True\n",
                "MAIL_SUPPRESS_SEND = True\n",
                "SECRET_KEY = 'Testing Testing Testing'\n",
            ]
        )
    os.environ["OTTERWIKI_SETTINGS"] = settings_cfg
    # Force local ChromaDB mode for tests
    os.environ["CHROMADB_MODE"] = "local"
    os.environ["CHROMADB_PATH"] = str(tmpdir / "chroma")
    os.environ["CHROMA_SYNC_STATE_PATH"] = str(tmpdir / "chroma_sync_state.json")
    os.environ["CHROMA_SYNC_INTERVAL"] = "3600"  # Don't run sync during tests
    os.environ["OTTERWIKI_API_KEY"] = "test-secret-key"

    from otterwiki.server import app, db, mail, storage

    app._otterwiki_tempdir = storage.path
    app.storage = storage
    app.test_mail = mail
    app.config["TESTING"] = True
    app.config["DEBUG"] = True
    yield app

    # Teardown: stop sync thread and restore env
    from otterwiki_semantic_search import reset_state

    reset_state()

    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def test_client(create_app):
    return create_app.test_client()


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-secret-key"}
