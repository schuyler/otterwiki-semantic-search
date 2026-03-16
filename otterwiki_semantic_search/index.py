"""Vector index operations — backend-agnostic."""

import logging
import threading

from otterwiki_semantic_search.chunking import chunk_page

log = logging.getLogger(__name__)

_reindex_lock = threading.Lock()

MAX_SEARCH_RESULTS = 50
_UPSERT_BATCH_SIZE = 5000


def _get_backend():
    """Get the backend, preferring registry resolution for multi-tenant."""
    from otterwiki_semantic_search import _state

    registry = _state.get("registry")
    if registry is not None:
        try:
            return registry.get_for_current_request()
        except RuntimeError:
            pass
    return _state.get("backend")


def _get_embedding_fn():
    from otterwiki_semantic_search import _state

    # Check registry first (multi-tenant shares embedding_fn)
    registry = _state.get("registry")
    if registry is not None:
        return registry.embedding_fn
    return _state.get("embedding_fn")


def upsert_page(pagepath, content, backend=None):
    """Chunk a page and upsert all chunks into the vector backend.

    Args:
        pagepath: Page path identifier.
        content: Page content text.
        backend: Optional explicit backend. If None, resolved from registry.
    """
    if backend is None:
        backend = _get_backend()
    if backend is None:
        return

    chunks = chunk_page(pagepath, content)
    if not chunks:
        delete_page(pagepath)
        return

    try:
        embedding_fn = _get_embedding_fn()
        embeddings = None
        if embedding_fn is not None:
            texts = [c["text"] for c in chunks]
            embeddings = embedding_fn.embed(texts)

        # Delete existing chunks for this page
        try:
            backend.delete(where={"page_path": pagepath})
        except Exception:
            log.debug("Delete before upsert failed for %s (may be empty)", pagepath)

        backend.upsert(
            ids=[c["id"] for c in chunks],
            texts=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=embeddings,
        )
        log.debug("Indexed %d chunks for %s", len(chunks), pagepath)
    except Exception:
        log.exception("Failed to upsert page %s", pagepath)


def delete_page(pagepath, backend=None):
    """Delete all chunks for a page from the vector backend.

    Args:
        pagepath: Page path identifier.
        backend: Optional explicit backend. If None, resolved from registry.
    """
    if backend is None:
        backend = _get_backend()
    if backend is None:
        return

    try:
        backend.delete(where={"page_path": pagepath})
        log.debug("Deleted chunks for %s", pagepath)
    except Exception:
        log.exception("Failed to delete page %s", pagepath)


def search(query, n=5, backend=None, max_chunks_per_page=2):
    """Search for pages similar to query. Returns list of result dicts.

    Args:
        query: Search query text.
        n: Maximum number of results.
        backend: Optional explicit backend. If None, resolved from registry.
        max_chunks_per_page: Max chunks returned per page (1-5, default 2).
    """
    if backend is None:
        backend = _get_backend()
    if backend is None:
        return []

    n = max(1, min(n, MAX_SEARCH_RESULTS))
    max_chunks_per_page = max(1, min(max_chunks_per_page, 5))

    prefetch = min(n * max_chunks_per_page * 2, MAX_SEARCH_RESULTS)

    try:
        embedding_fn = _get_embedding_fn()
        if embedding_fn is not None:
            # FAISS path: embed the query externally
            query_embeddings = embedding_fn.embed([query])
            results = backend.query(
                query_embeddings=query_embeddings,
                n_results=prefetch,
            )
        else:
            # ChromaDB path: backend handles embedding internally
            results = backend.query(
                query_texts=[query],
                n_results=prefetch,
            )
    except Exception:
        # Fall back to smaller query
        try:
            if embedding_fn is not None:
                query_embeddings = embedding_fn.embed([query])
                results = backend.query(
                    query_embeddings=query_embeddings,
                    n_results=n,
                )
            else:
                results = backend.query(
                    query_texts=[query],
                    n_results=n,
                )
        except Exception:
            log.exception("Search failed for query: %s", query)
            return []

    if not results or not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Deduplicate by page_path, keeping up to max_chunks_per_page best chunks
    page_chunks = {}  # page_path -> list of result dicts (sorted by distance)
    for i, doc_id in enumerate(ids):
        meta = metadatas[i]
        page_path = meta["page_path"]
        distance = distances[i]
        text = documents[i]

        # Strip the [section_path] prefix before computing snippet
        section_path = meta.get("section_path")
        if section_path:
            prefix = f"[{section_path}] "
            snippet_text = text[len(prefix):] if text.startswith(prefix) else text
        else:
            snippet_text = text
        snippet = snippet_text
        if len(snippet) > 150:
            truncated = snippet[:150].rsplit(" ", 1)[0]
            snippet = truncated + "..."

        entry = {
            "name": page_path,
            "path": page_path,
            "text": text,
            "snippet": snippet,
            "distance": distance,
            "section": meta.get("section"),
            "section_path": meta.get("section_path"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "page_word_count": meta.get("page_word_count"),
        }

        if page_path not in page_chunks:
            page_chunks[page_path] = []
        if len(page_chunks[page_path]) < max_chunks_per_page:
            page_chunks[page_path].append(entry)

    # Flatten and sort by distance
    all_results = []
    for chunks in page_chunks.values():
        all_results.extend(chunks)

    all_results.sort(key=lambda x: x["distance"])
    return all_results[:n]


def is_reindex_in_progress():
    return _reindex_lock.locked()


def _batch_upsert(backend, chunks, embedding_fn=None):
    """Upsert chunks in batches."""
    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]
        embeddings = None
        if embedding_fn is not None:
            embeddings = embedding_fn.embed([c["text"] for c in batch])
        backend.upsert(
            ids=[c["id"] for c in batch],
            texts=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embeddings,
        )


def reindex_all(storage, app_config=None, backend=None):
    """Delete and rebuild the entire index from storage.

    Args:
        storage: Git storage instance.
        app_config: Flask app config dict.
        backend: Optional explicit backend. If None, resolved from registry.
    """
    if backend is None:
        backend = _get_backend()
    if backend is None:
        return {"pages_indexed": 0, "chunks_created": 0}

    acquired = _reindex_lock.acquire(blocking=False)
    if not acquired:
        log.info("Reindex already in progress, skipping")
        return {"pages_indexed": 0, "chunks_created": 0, "skipped": True}

    try:
        embedding_fn = _get_embedding_fn()

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

        # Phase 2: Reset and rebuild
        backend.reset()
        if all_chunks:
            _batch_upsert(backend, all_chunks, embedding_fn)

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
