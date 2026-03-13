"""Background sync thread to keep the vector index in sync with git changes."""

import json
import logging
import os
import threading

from otterwiki_semantic_search import index

log = logging.getLogger(__name__)


class SyncThread(threading.Thread):
    """Daemon thread that periodically syncs the vector index with git changes."""

    def __init__(self, app, storage, interval=60, state_path="/app-data/chroma_sync_state.json"):
        super().__init__(daemon=True)
        self.app = app
        self.storage = storage
        self.interval = interval
        self.state_path = state_path
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                with self.app.app_context():
                    self._sync()
            except Exception:
                log.exception("Sync cycle failed")
            self._stop_event.wait(self.interval)

    def _sync(self):
        from otterwiki_semantic_search import _state

        backend = _state.get("backend")
        if backend is None:
            return

        # Don't start a reindex if one is already running
        if index.is_reindex_in_progress():
            return

        last_sha = self._read_state()

        # If no state or index is empty, do a full reindex
        if last_sha is None or backend.count() == 0:
            log.info("No sync state or empty index, running full reindex")
            index.reindex_all(self.storage, self.app.config)
            self._write_state()
            return

        # Get current HEAD
        try:
            current_sha = self.storage.repo.head.commit.hexsha
        except Exception:
            log.exception("Failed to get HEAD commit")
            return

        # Nothing changed
        if current_sha == last_sha:
            return

        # Find changed .md files between last_sha and HEAD
        try:
            raw = self.storage.repo.git.log(
                f"{last_sha}..HEAD",
                "--name-only",
                "--pretty=format:",
            )
        except Exception:
            log.exception("Failed to get git log for sync")
            return

        if not raw.strip():
            self._write_state()
            return

        # Parse unique changed filenames
        changed = set()
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".md"):
                changed.add(line)

        retain_case = self.app.config.get("RETAIN_PAGE_NAME_CASE", False)

        for filepath in changed:
            pagepath = index.filepath_to_pagepath(filepath, retain_case)
            if self.storage.exists(filepath):
                try:
                    content = self.storage.load(filepath)
                    index.upsert_page(pagepath, content)
                except Exception:
                    log.warning("Failed to sync %s", filepath)
            else:
                index.delete_page(pagepath)

        self._write_state()
        if changed:
            log.info("Synced %d changed files", len(changed))

    def _read_state(self):
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
                return data.get("last_sha")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _write_state(self):
        try:
            current_sha = self.storage.repo.head.commit.hexsha
        except Exception:
            log.exception("Failed to get HEAD commit for state write")
            return

        try:
            dirname = os.path.dirname(self.state_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"last_sha": current_sha}, f)
        except Exception:
            log.exception("Failed to write sync state")
