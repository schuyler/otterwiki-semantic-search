"""ChromaDB index operations."""

import logging
import threading

from otterwiki_semantic_search.chunking import chunk_page

log = logging.getLogger(__name__)

_write_lock = threading.Lock()
_reindex_lock = threading.Lock()

MAX_SEARCH_RESULTS = 50
_UPSERT_BATCH_SIZE = 5000


def _get_collection():
    from otterwiki_semantic_search import _state

    return _state.get("collection")


def upsert_page(pagepath, content):
    """Chunk a page and upsert all chunks into ChromaDB."""
    collection = _get_collection()
    if collection is None:
        return

    chunks = chunk_page(pagepath, content)
    if not chunks:
        # Page has no content worth indexing; just delete existing
        delete_page(pagepath)
        return

    try:
        with _write_lock:
            # Delete existing chunks for this page
            try:
                collection.delete(where={"page_path": pagepath})
            except Exception:
                log.debug("Delete before upsert failed for %s (may be empty)", pagepath)

            # Upsert new chunks in one batch
            collection.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
        log.debug("Indexed %d chunks for %s", len(chunks), pagepath)
    except Exception:
        log.exception("Failed to upsert page %s", pagepath)


def delete_page(pagepath):
    """Delete all chunks for a page from ChromaDB."""
    collection = _get_collection()
    if collection is None:
        return

    try:
        with _write_lock:
            collection.delete(where={"page_path": pagepath})
        log.debug("Deleted chunks for %s", pagepath)
    except Exception:
        log.exception("Failed to delete page %s", pagepath)


def search(query, n=5):
    """Search for pages similar to query. Returns list of result dicts."""
    collection = _get_collection()
    if collection is None:
        return []

    n = max(1, min(n, MAX_SEARCH_RESULTS))

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n * 3,
        )
    except Exception:
        # ChromaDB may throw if n_results > document count in some versions;
        # fall back to a smaller query.
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n,
            )
        except Exception:
            log.exception("Search failed for query: %s", query)
            return []

    if not results or not results["ids"] or not results["ids"][0]:
        return []

    # Deduplicate by page_path, keeping the chunk with lowest distance
    seen = {}
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, doc_id in enumerate(ids):
        page_path = metadatas[i]["page_path"]
        distance = distances[i]
        if page_path not in seen or distance < seen[page_path]["distance"]:
            snippet = documents[i]
            if len(snippet) > 150:
                # Truncate on word boundary
                truncated = snippet[:150].rsplit(" ", 1)[0]
                snippet = truncated + "..."
            seen[page_path] = {
                "name": page_path,
                "path": page_path,
                "snippet": snippet,
                "distance": distance,
            }

    # Sort by distance and return top n
    results_list = sorted(seen.values(), key=lambda x: x["distance"])
    return results_list[:n]


def is_reindex_in_progress():
    return _reindex_lock.locked()


def _batch_upsert(collection, chunks):
    """Upsert chunks in batches to stay within ChromaDB limits."""
    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def reindex_all(storage, app_config=None):
    """Delete and rebuild the entire index from storage."""
    from otterwiki_semantic_search import _state

    collection = _state.get("collection")
    if collection is None:
        return {"pages_indexed": 0, "chunks_created": 0}

    acquired = _reindex_lock.acquire(blocking=False)
    if not acquired:
        log.info("Reindex already in progress, skipping")
        return {"pages_indexed": 0, "chunks_created": 0, "skipped": True}

    client = _state.get("client")
    collection_name = _state.get("collection_name", "otterwiki_pages")

    try:
        # Phase 1: Read and chunk all pages OUTSIDE the write lock
        files, _ = storage.list()
        md_files = [f for f in files if f.endswith(".md")]

        retain_case = app_config.get("RETAIN_PAGE_NAME_CASE", False) if app_config else False

        all_chunks = []
        pages_indexed = 0

        for filepath in md_files:
            try:
                content = storage.load(filepath)
            except Exception:
                log.warning("Could not load %s, skipping", filepath)
                continue

            pagepath = filepath_to_pagepath(filepath, retain_case)
            chunks = chunk_page(pagepath, content)
            if chunks:
                all_chunks.extend(chunks)
                pages_indexed += 1

        # Phase 2: Swap the collection under the write lock
        with _write_lock:
            client.delete_collection(collection_name)
            collection = client.get_or_create_collection(collection_name)
            _state["collection"] = collection

            if all_chunks:
                _batch_upsert(collection, all_chunks)

        chunks_created = len(all_chunks)
        log.info("Reindex complete: %d pages, %d chunks", pages_indexed, chunks_created)
        return {"pages_indexed": pages_indexed, "chunks_created": chunks_created}
    except Exception:
        log.exception("Reindex failed")
        return {"pages_indexed": 0, "chunks_created": 0}
    finally:
        _reindex_lock.release()


def filepath_to_pagepath(filepath, retain_case=False):
    """Convert a filepath like 'some/page.md' to a pagepath."""
    if filepath.endswith(".md"):
        filepath = filepath[:-3]
    if not retain_case:
        parts = filepath.split("/")
        parts = [p.title() for p in parts]
        filepath = "/".join(parts)
    return filepath
